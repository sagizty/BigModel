import gc
import sys
import torch
from torch import nn
from pathlib import Path
from torch.utils.cpp_extension import load
from torch.nn.functional import softplus, normalize

ROOT_PATH = Path(__file__).resolve().parent
sys.path.append(str(ROOT_PATH.parent))
from gigapath.pos_embed import get_2d_sincos_pos_embed

load(
    name="wkv7",
    sources=[str(ROOT_PATH / "cuda" / "wkv7_op.cpp"), str(ROOT_PATH / "cuda" / "wkv7.cu")],
    is_python_module=False,
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


class WKV_7(torch.autograd.Function):
    @staticmethod
    def create_tensor(shape, device, requires_grad=False):
        return torch.empty(
            shape,
            device=device,
            dtype=torch.half,
            requires_grad=requires_grad,
            memory_format=torch.contiguous_format,
        )

    @staticmethod
    def forward(ctx, r, w, k, v, a, b):
        with torch.no_grad():
            B, T, C = r.size()
            H = C // 64
            ctx.B, ctx.T, ctx.C, ctx.H = B, T, C, H
            ctx.save_for_backward(r, k, v, w, u)
            y = WKV_7.create_tensor((B, T, C), r.device, True)
            torch.ops.wkv7.forward(B, T, C, H, r, w, k, v, a, b, y)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            B, T, C, H = ctx.B, ctx.T, ctx.C, ctx.H
            r, w, k, v, a, b = ctx.saved_tensors
            gr, gk, gv, gw = [WKV_7.create_tensor((B, T, C), r.device) for _ in range(4)]
            ga, gb = [WKV_7.create_tensor((B, C), r.device) for _ in range(2)]
            torch.ops.wkv7.backward(B, T, C, H, r, w, k, v, a, b, gy, gr, gw, gk, gv, ga, gb)
            ga, gb = torch.sum(ga, 0).view(H, C // H), torch.sum(gb, 0).view(H, C // H)
            gradients = (None, None, None, None, gr, gk, gv, gw, ga, gb)
            return gradients


def cuda_wkv_7(r, w, k, v, a, b):
    return WKV_7.apply(r, w, k, v, a, b)


class TimeMix(nn.Module):
    def __init__(self, embed_dim, head_size=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_size = head_size
        self.n_head = self.embed_dim // self.head_size

        self.lambdas = nn.Parameter(torch.empty(6, 1, 1, self.embed_dim))

        self.k_A = nn.Parameter(torch.empty(self.embed_dim, 64))
        self.k_B = nn.Parameter(torch.empty(64, self.embed_dim))

        self.g_A = nn.Parameter(torch.empty(self.embed_dim, 128))
        self.g_B = nn.Parameter(torch.empty(128, self.embed_dim))

        self.w_A = nn.Parameter(torch.empty(self.embed_dim, 64))
        self.w_B = nn.Parameter(torch.empty(64, self.embed_dim))
        self.w_miu = nn.Parameter(torch.empty(1, 1, self.embed_dim))

        self.a_A = nn.Parameter(torch.empty(self.embed_dim, 64))
        self.a_B = nn.Parameter(torch.empty(64, self.embed_dim))
        self.a_time = nn.Parameter(torch.empty(1, 1, self.embed_dim))

        self.x_A = nn.Parameter(torch.empty(self.embed_dim, 32 * 6))
        self.x_B = nn.Parameter(torch.empty(6, 32, self.embed_dim))
        self.x_miu = nn.Parameter(torch.empty(1, 1, self.embed_dim))
        self.x_time = nn.Parameter(torch.empty(self.n_head, self.head_size))
        self.x_ln = nn.GroupNorm(self.n_head, self.embed_dim, eps=1e-5 * self.n_head)

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.r_W, self.k_W, self.v_W, self.o_W = [
            nn.Linear(self.embed_dim, self.embed_dim, bias=False) for _ in range(4)
        ]

    @staticmethod
    def lerp(a, b_minus_a, miu):
        return a + b_minus_a * miu

    @staticmethod
    def lora(x, A, B, lambda_=0):
        return lambda_ + torch.tanh(x @ A) @ B

    @staticmethod
    def lora_m(x, A, B, lambdas):
        b, t, _ = x.size()
        x = torch.tanh(x @ A).view(b * t, 6, -1).transpose(0, 1)
        x = torch.bmm(x, B).view(6, b, t, -1)
        x = lambdas + x
        return x

    @staticmethod
    def ddlerp(a, b, miu, A, B, lambdas):
        x = TimeMix.lerp(a, b - a, miu)
        miu = TimeMix.lora_m(x, A, B, lambdas)
        x = TimeMix.lerp(a, b - a, miu)
        return x

    def forward(self, x):
        B, T, C = x.size()
        x_last = self.time_shift(x)
        x_ddlerp = self.ddlerp(x, x_last, self.x_miu, self.x_A, self.x_B, self.lambdas)
        r, w, k, v, a, g = x_ddlerp.unbind(dim=0)

        v = self.v_W(v)
        r = self.r_W(r)
        g = TimeMix.lora(g, self.g_A, self.g_B)
        w = -softplus(-TimeMix.lora(w, self.w_A, self.w_B, self.w_miu)) - 0.5
        k = normalize(TimeMix.lora(k, self.k_A, self.k_B, self.k_W(k)), dim=-1, p=2.0)
        a = torch.sigmoid(self.a_time + (a @ self.a_A) @ self.a_B) * 2.0

        x = cuda_wkv_7(r, w, k * torch.clamp(w * 0.5, max=0).exp(), v, -k, k * a)

        x = self.x_ln(x.view(B * T, C)).view(B, T, C)
        r, k, v = r.view(B, T, self.n_head, -1), k.view(B, T, self.n_head, -1), v.view(B, T, self.n_head, -1)
        x = x + ((r * k * self.x_time).sum(dim=-1, keepdim=True) * v).view(B, T, C)
        x = self.o_W(x * g)

        return x


class ChannelMix(nn.Module):
    def __init__(self, embed_dim, dim_ffn):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.v_W = nn.Linear(dim_ffn, embed_dim, bias=False)

        self.k_W = nn.Linear(embed_dim, dim_ffn, bias=False)
        self.k_miu = nn.Parameter(torch.empty(1, 1, embed_dim))

        self.r_W = nn.Linear(embed_dim, embed_dim, bias=False)
        self.r_miu = nn.Parameter(torch.empty(1, 1, embed_dim))

    def forward(self, x):
        x_last = self.time_shift(x)

        k = TimeMix.lerp(x, x_last, self.k_miu)
        k = self.k_W(k)
        k = torch.relu(k) ** 2
        kv = self.v_W(k)

        r = TimeMix.lerp(x, x_last, self.r_miu)
        r = torch.sigmoid(self.r_W(r))
        return r * kv


class Block(nn.Module):
    def __init__(self, embed_dim, dim_ffn):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.time_mix = TimeMix(embed_dim)
        self.channel_mix = ChannelMix(embed_dim, dim_ffn)

    def forward(self, x):
        x = x + self.time_mix(self.ln1(x))
        x = x + self.channel_mix(self.ln2(x))
        return x


class PathRWKVv7(nn.Module):
    def __init__(
        self,
        depth=24,
        embed_dim=768,
        slide_ngrids=1000,
        ROI_feature_dim=1536,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.slide_ngrids = slide_ngrids
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer("pos_embed", torch.zeros(1, slide_ngrids**2 + 1, embed_dim), persistent=False)
        self.emb = nn.Linear(ROI_feature_dim, embed_dim, bias=False)
        self.blocks = nn.ModuleList([Block(embed_dim, int((embed_dim * 3.5) // 32 * 32)) for _ in range(depth)])
        self.ln_out = nn.LayerNorm(embed_dim)

    def coords_to_pos(self, coords):
        coords_ = torch.floor(coords / 256.0)
        pos = coords_[..., 0] * self.slide_ngrids + coords_[..., 1]
        return pos.long() + 1  # add 1 for the cls token

    def forward(self, x, coords=None):
        x, coords = x.half(), coords.half()
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

        for param in state_dict.values():
            n_params += param.numel()

        pos_embed = get_2d_sincos_pos_embed(self.embed_dim, self.slide_ngrids, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        print("Model Params", n_params)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    model = PathRWKVv7()
    model.init_params()
    model.half().cuda()
    x = torch.randn((1, 10000, 1536), device=torch.device("cuda"))
    coords = torch.randn((1, 10000, 2), device=torch.device("cuda"))
    out = model.forward(x, coords)
    print(out[0].shape)
    loss = out[0].sum()
    loss.backward()
    print("Test sucessful!")
