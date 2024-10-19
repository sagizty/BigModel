import gc
import sys
import math
import torch
from torch import nn, topk
from pathlib import Path
from torch.nn.functional import silu
from torch.utils.cpp_extension import load


ROOT_PATH = Path(__file__).resolve().parent
sys.path.append(str(ROOT_PATH.parent))
from gigapath.pos_embed import get_2d_sincos_pos_embed

wkv6_cuda = load(
    name="wkv6",
    sources=[str(ROOT_PATH / "cuda" / "wkv6_op.cpp"), str(ROOT_PATH / "cuda" / "wkv6_cuda.cu")],
    verbose=False,
    extra_cuda_cflags=[
        "-res-usage",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        "--extra-device-vectorization",
        "-D_N_=64",
        "-D_T_=16384",
    ],
)


class WKV_6(torch.autograd.Function):
    @staticmethod
    def create_tensor(shape, device, requires_grad=False):
        return torch.empty(
            shape,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=requires_grad,
            memory_format=torch.contiguous_format,
        )

    @staticmethod
    # Forward: r, k, v, w, u => y
    def forward(ctx, b, t, c, h, r, k, v, w, u):
        with torch.no_grad():
            ctx.b, ctx.t, ctx.c, ctx.h = b, t, c, h
            ctx.save_for_backward(r, k, v, w, u)
            y = WKV_6.create_tensor((b, t, c), r.device, True)
            wkv6_cuda.forward(b, t, c, h, r, k, v, w, u, y)
            return y

    @staticmethod
    # Backward: gy => gr, gk, gv, gw, gu
    def backward(ctx, gy):
        with torch.no_grad():
            b, t, c, h = ctx.b, ctx.t, ctx.c, ctx.h
            r, k, v, w, u = ctx.saved_tensors
            gr, gk, gv, gw = [WKV_6.create_tensor((b, t, c), gy.device) for _ in range(4)]
            gu = WKV_6.create_tensor((b, c), gy.device)
            wkv6_cuda.backward(b, t, c, h, r, k, v, w, u, gy, gr, gk, gv, gw, gu)
            gu = torch.sum(gu, 0).view(h, c // h)
            gradients = (None, None, None, None, gr, gk, gv, gw, gu)
            return gradients


def cuda_wkv_6(b, t, c, h, r, k, v, w, u):
    return WKV_6.apply(b, t, c, h, r, k, v, w, u)


class TimeMix(nn.Module):
    def __init__(self, embed_dim, n_blocks, layer_id, head_size=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_blocks = n_blocks
        self.layer_id = layer_id
        self.head_size = head_size
        self.n_head = self.embed_dim // self.head_size

        ratio_0_to_1 = layer_id / (self.n_blocks - 1)  # 0 to 1
        ratio_1_to_almost0 = 1.0 - (layer_id / self.n_blocks)  # 1 to ~0
        ddd = torch.ones(1, 1, self.embed_dim)
        for i in range(self.embed_dim):
            ddd[0, 0, i] = i / self.embed_dim

        # Time mix params
        self.miu_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
        self.lambda_ = nn.Parameter(
            torch.stack(
                [
                    1.0 - torch.pow(ddd, ratio_1_to_almost0),  # lambda_w
                    1.0 - torch.pow(ddd, ratio_1_to_almost0),  # lambda_k
                    1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0),  # lambda_r
                    1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0),  # lambda_g
                    1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1),  # lambda_v
                ]
            )
        )

        self.A = nn.Parameter(torch.zeros(self.embed_dim, 32 * 5))
        self.B = nn.Parameter(torch.zeros(5, 32, self.embed_dim).uniform_(-0.01, 0.01))

        # Time decay params
        decay_speed = torch.ones(self.embed_dim)
        for n in range(self.embed_dim):
            decay_speed[n] = -6 + 5 * (n / (self.embed_dim - 1)) ** (0.7 + 1.3 * ratio_0_to_1)

        self.time_decay_miu = nn.Parameter(decay_speed.reshape(1, 1, self.embed_dim))
        self.time_decay_A = nn.Parameter(torch.zeros(self.embed_dim, 64))
        self.time_decay_B = nn.Parameter(torch.zeros(64, self.embed_dim).uniform_(-0.01, 0.01))

        # Bonus
        tmp = torch.zeros(self.embed_dim)
        for n in range(self.embed_dim):
            zigzag = ((n + 1) % 3 - 1) * 0.1
            tmp[n] = ratio_0_to_1 * (1 - (n / (self.embed_dim - 1))) + zigzag
        self.u = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.W_r, self.W_k, self.W_v, self.W_o, self.W_g = [
            nn.Linear(self.embed_dim, self.embed_dim, bias=False) for _ in range(5)
        ]
        self.ln_x = nn.GroupNorm(self.n_head, self.embed_dim, eps=1e-5 * self.n_head)

    @staticmethod
    def lerp(a, b_minus_a, miu):
        return a + b_minus_a * miu

    @staticmethod
    def lora(x, A, B, lambda_):
        b, t, _ = x.size()
        x = torch.tanh(x @ A).view(b * t, 5, -1).transpose(0, 1)
        x = torch.bmm(x, B).view(5, b, t, -1)
        x = lambda_ + x
        return x

    @staticmethod
    def ddlerp(a, b, miu_x, A, B, lambda_):
        b_minus_a = b - a
        x = TimeMix.lerp(a, b_minus_a, miu_x)
        miu = TimeMix.lora(x, A, B, lambda_)
        x = TimeMix.lerp(a, b_minus_a, miu)
        return x

    def jit_func(self, x):
        x_last = self.time_shift(x)
        x_ddlerp = self.ddlerp(x, x_last, self.miu_x, self.A, self.B, self.lambda_)
        w, k, v, r, g = x_ddlerp.unbind(dim=0)

        k = self.W_k(k)
        v = self.W_v(v)
        r = self.W_r(r)
        g = silu(self.W_g(g))
        w = self.time_decay_miu + torch.tanh(w @ self.time_decay_A) @ self.time_decay_B
        return r, k, v, g, w

    def jit_func_2(self, x, g):
        b, t, c = x.size()
        x = x.view(b * t, c)
        x = self.ln_x(x).view(b, t, c)
        x = self.W_o(x * g)
        return x

    def forward(self, x):
        b, t, c = x.size()
        r, k, v, g, w = self.jit_func(x)
        x = cuda_wkv_6(b, t, c, self.n_head, r, k, v, w, self.u)
        x = self.jit_func_2(x, g)
        return x


class ChannelMix(nn.Module):
    def __init__(self, embed_dim, n_blocks, layer_id, dim_ffn):
        super().__init__()
        self.layer_id = layer_id

        # Params
        ratio_1_to_almost0 = 1.0 - (layer_id / n_blocks)  # 1 to ~0
        ddd = torch.ones(1, 1, embed_dim)
        for i in range(embed_dim):
            ddd[0, 0, i] = i / embed_dim

        self.miu_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
        self.miu_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.W_k = nn.Linear(embed_dim, dim_ffn, bias=False)
        self.W_v = nn.Linear(dim_ffn, embed_dim, bias=False)
        self.W_r = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        x_last = self.time_shift(x)

        k = TimeMix.lerp(x, x_last, self.miu_k)
        k = self.W_k(k)
        k = torch.relu(k) ** 2
        kv = self.W_v(k)

        r = TimeMix.lerp(x, x_last, self.miu_r)
        r = torch.sigmoid(self.W_r(r))
        return r * kv


class Block(nn.Module):
    def __init__(self, embed_dim, n_blocks, block_id, dim_ffn):
        super().__init__()
        self.block_id = block_id
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        # if self.block_id == 0:
        #     self.ln0 = nn.LayerNorm(embed_dim)

        self.time_mix = TimeMix(embed_dim, n_blocks, block_id)
        self.channel_mix = ChannelMix(embed_dim, n_blocks, block_id, dim_ffn)

    def forward(self, x):
        # x = self.ln0(x) if self.block_id == 0 else x
        x = x + self.time_mix(self.ln1(x))
        x = x + self.channel_mix(self.ln2(x))
        return x


class PathRWKV(nn.Module):
    def __init__(
        self,
        depth=24,
        embed_dim=768,
        slide_ngrids=1000,
        ROI_feature_dim=1536,
    ):
        super().__init__()
        self.n_blocks = depth
        self.embed_dim = embed_dim
        self.slide_ngrids = slide_ngrids
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer("pos_embed", torch.zeros(1, slide_ngrids**2 + 1, embed_dim), persistent=False)
        self.emb = nn.Linear(ROI_feature_dim, embed_dim, bias=False)
        self.blocks = nn.ModuleList(
            [Block(embed_dim, depth, blk, int((embed_dim * 3.5) // 32 * 32)) for blk in range(depth)]
        )
        self.ln_out = nn.LayerNorm(embed_dim)

    def coords_to_pos(self, coords):
        coords_ = torch.floor(coords / 256.0)
        pos = coords_[..., 0] * self.slide_ngrids + coords_[..., 1]
        return pos.long() + 1  # add 1 for the cls token

    def forward(self, x, coords=None):
        x, coords = x.bfloat16(), coords.bfloat16()
        x = self.emb(x)

        # Get position indices
        pos = self.coords_to_pos(coords)  # [b, t]
        x = x + self.pos_embed[:, pos, :].squeeze(0)

        # Append class token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for block in self.blocks:
            x = block(x)

        x = self.ln_out(x)
        x, _ = torch.max(x, dim=1)
        return [x]

    def init_params(self):
        state_dict = self.state_dict()
        n_params = 0

        for name, param in state_dict.items():
            n_params += param.numel()

            if "ln_x.weight" in name:
                state_dict[name] = ((1 + int(name.split(".")[1])) / self.n_blocks) ** 0.7

            elif name.endswith(".weight") and "ln" not in name:
                scale = (
                    0
                    if any(n in name for n in ["time_mix.W_o", "channel_mix.W_v", "channel_mix.W_r"])
                    else 0.1 if any(n in name for n in ["time_mix.W_k", "time_mix.W_g"]) else 1.0
                )
                nn.init.zeros_(state_dict[name]) if scale == 0 else nn.init.orthogonal_(state_dict[name], gain=scale)

        pos_embed = get_2d_sincos_pos_embed(self.embed_dim, self.slide_ngrids, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        print("Model Params", n_params)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    model = PathRWKV()
    model.init_params()
    model.bfloat16().cuda()
    x = torch.randn((1, 10000, 1536), device=torch.device("cuda"))
    coords = torch.randn((1, 10000, 2), device=torch.device("cuda"))
    out = model.forward(x, coords)
    loss = out[0].sum()
    loss.backward()
    print("Test sucessful!")
