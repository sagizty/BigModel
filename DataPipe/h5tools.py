"""
WSI h5 file tools   Script  ver： July 27th 01:00


"""
import os
import h5py
import numpy as np
from tqdm import tqdm
from PIL import Image
import random


def hdf5_save_a_patch(h5_file_path, patch, patch_type='features'):
    """
    add a patch and its information into a h5 file

    h5_file_path: loc to save h5 file
    patch: 'features' format of numpy
    patch_type: 'features'

    h5file[patch_type] is a list of numpy
    (images) None, they are too big in h5, we don't save them in h5. we keep them in jpeg at the same folder of h5
    (features) format of numpy
    """
    assert h5_file_path.endswith('.h5')
    assert type(patch_type) is str
    assert patch_type != 'images'

    # init a new hdf5 file if not exist
    if not os.path.exists(h5_file_path):
        file = h5py.File(h5_file_path, "w")
        # we keep things in h5 in numpy format
        patch = np.array(patch)[np.newaxis, ...]  # add a dim of index
        dtype = patch.dtype

        # Initialize a resizable dataset to hold the output
        patch_shape = patch.shape  # [1, ...]  each data keep in this shape, and also chunk as this shape
        maxshape = (None,) + patch_shape[1:]  # this is for universally applicable
        # maximum dimensions to (None, ...) (None means unlimited)
        patch_dset = file.create_dataset(patch_type, shape=patch_shape, maxshape=maxshape, chunks=patch_shape,
                                         dtype=dtype)

        patch_dset[:] = patch

        file.close()

    # otherwise, check the current h5 and maybe add to current h5 file
    else:
        patch = np.array(patch)[np.newaxis, ...]
        patch_shape = patch.shape  # [1, ...]  each data keep in this shape, and also chunk as this shape

        file = h5py.File(h5_file_path, "a")

        try:
            patch_dset = file[patch_type]
        except:
            dtype = patch.dtype
            # Initialize a resizable dataset to hold the output
            maxshape = (None,) + patch_shape[1:]  # this is for universally applicable
            # maximum dimensions to (None, ...) (None means unlimited)
            patch_dset = file.create_dataset(patch_type, shape=patch_shape, maxshape=maxshape, chunks=patch_shape,
                                             dtype=dtype)
            patch_dset[:] = patch
        else:
            # make new space and append
            patch_dset.resize(len(patch_dset) + patch_shape[0], axis=0)
            patch_dset[-patch_shape[0]:] = patch

        file.close()


def hdf5_save_a_patch_coord(h5_file_path, coord_y, coord_x):
    """
    add a patch and its information into a h5 file

    h5_file_path: loc to save h5 file
    coord_y: Y is patch index in WSI height
    coord_x: X is patch index in WSI width

    h5file['coords_yx'] is a list of coordinates, each item is a [Y, X], Y, X is patch index in WSI
    """
    assert h5_file_path.endswith('.h5')

    # init a new hdf5 file if not exist
    if not os.path.exists(h5_file_path):
        file = h5py.File(h5_file_path, "w")

        patch_coord = np.array([coord_y, coord_x])[np.newaxis, ...]
        # save coordinate
        coord_dset = file.create_dataset('coords_yx', shape=(1, 2), maxshape=(None, 2), chunks=(1, 2), dtype=np.int32)
        coord_dset[:] = patch_coord

        file.close()

    # otherwise, check the current h5 and maybe add to current h5 file
    else:
        patch_coord = np.array([coord_y, coord_x])[np.newaxis, ...]
        patch_coord_shape = patch_coord.shape

        file = h5py.File(h5_file_path, "a")

        try:
            coord_dset = file['coords_yx']
        except:
            coord_dset = file.create_dataset('coords_yx', shape=(1, 2), maxshape=(None, 2), chunks=(1, 2), dtype=np.int32)
            coord_dset[:] = patch_coord
        else:
            # make new space and append
            coord_dset.resize(len(coord_dset) + patch_coord_shape[0], axis=0)
            coord_dset[-patch_coord_shape[0]:] = patch_coord

        file.close()


def hdf5_save_a_wsi_coord_npy(h5_file_path):
    """
    build the coordinate index numpy of a WSI h5

    h5_file_path:
    """

    # build the index coordinate numpy
    file = h5py.File(h5_file_path, "a")

    # load the coordinate information based on the coordinate list
    # h5file['coords'] is a list of coordinates, each item is a [Y,X], Y,X is patch index in WSI
    coord_read = file['coords_yx']  # N,2

    Y_max = coord_read[:, 0].max() + 1  # +1 is to make the correct shape of index matrix (start with 0)
    X_max = coord_read[:, 1].max() + 1

    coord_npy = np.ones([Y_max, X_max]) * -1  # set default to -1 (no patch)

    for index in range(len(coord_read)):
        coord_npy[coord_read[index, 0], coord_read[index, 1]] = index

    # save coordinate in numpy matrix, only have one matrix for the whole WSI
    coord_dset = file.create_dataset('coord_npy', shape=(1, Y_max, X_max), maxshape=(1, Y_max, X_max),
                                     chunks=(1, Y_max, X_max), dtype=np.int8)
    coord_dset[:] = coord_npy

    file.close()


def hdf5_save_a_WSI(h5_file_path, patch_list=None, patch_loc_list=None, patch_type='images'):
    """
    h5_file_to_save: loc to save h5
    patch_list: a list of patches in numpy (feature), default is None for patch images
    patch_loc_list: length of patch list, each item is [Y,X], Y,X is patch index in WSI
    patch_type: 'images' or 'features'

    h5file[patch_type] is a list of numpy
    (images) None, they are too big in h5, we don't save them in h5. we keep them in jpeg at the same folder of h5
    (features) format of numpy

    output:
    coord_npy is a numpy size of Y_max, X_max, each value is the patch index of patch at [Y,X], or -1 for missing
    the whole WSI only have one coord_npy, keep as [1, Y_max, X_max]
    """
    assert type(patch_type) is str
    assert patch_loc_list is not None
    assert (patch_list is None and patch_type == 'images') or patch_type != 'images'

    # at this stage, clear the previous WSI h5 file
    if os.path.isfile(h5_file_path):
        os.remove(h5_file_path)

    WSI_name = os.path.split(h5_file_path)[-1]

    # build the data part of patch and coord_idx
    for idx in range(len(patch_loc_list)):
        if patch_type != 'images':
            hdf5_save_a_patch(h5_file_path, patch_list[idx], patch_type=patch_type)
        else:
            # we don't keep image patches in h5
            # if we need to save patch images, they have been saved in WSI cropping at the same folder of h5 file
            # patch_path = os.path.join(os.path.split(h5_file_path)[0], str(idx)+'.jpeg')
            pass
        hdf5_save_a_patch_coord(h5_file_path, coord_y=patch_loc_list[idx][0], coord_x=patch_loc_list[idx][1])

    # build the coordinate index numpy
    hdf5_save_a_wsi_coord_npy(h5_file_path)


def hdf5_check_all_coord(h5_file_path):
    """
    coord_list is a list of coordinates, each item is [Y,X], Y,X is coordinates
    coord_npy is a numpy size of Y_max, X_max, each value is the patch index of patch at [Y,X], or -1 for missing
    """
    # get a list of coordinates, each item is [Y,X], Y,X is coordinates
    # can select patch based on coordinates.
    file = h5py.File(h5_file_path, "r")
    coord_dset = file['coords_yx']  # N,2

    # the whole WSI only have one coord_npy, keep as [1, Y_max, X_max]
    coord_npy = file['coord_npy'][0]

    coord_list = []
    for coord in coord_dset:
        coord_list.append(coord.tolist())
    file.close()

    return coord_list, coord_npy


def hdf5_load_a_list_of_patches_by_id(h5_file_path, patch_idx_list, patch_type='images'):
    """
    each patch have a coordinate of [Y,X], Y,X is patch index in WSI

    h5_file_path: loc to save h5
    patch_idx_list: list of [patch order index]
    patch_type: 'images' or 'features'

    h5file[patch_type] is a list of numpy
    (images are to big in h5, we don't return them)
    (features) format of numpy

    h5file['coords'] is a list of coordinates, each item is a [Y,X], Y,X is patch index in WSI

    """
    # load patch based on coordinate index
    assert h5_file_path.endswith('.h5')
    assert type(patch_idx_list) is list

    file = h5py.File(h5_file_path, "r")

    patch_list = []
    patch_coord_list = []

    for patch_idx in patch_idx_list:
        patch_list.append(file[patch_type][patch_idx]) if patch_type != 'images' else patch_list.append(None)
        patch_coord_list.append(file['coords_yx'][patch_idx])

    file.close()

    return patch_list, patch_coord_list


def print_h5_dataset_sizes(h5_file_path):
    with h5py.File(h5_file_path, 'r') as f:
        def visit_func(name, node):
            if isinstance(node, h5py.Dataset):
                # Get the dataset
                dataset = f[name]
                # Calculate the size of the dataset in bytes
                dataset_size = dataset.dtype.itemsize
                for dim in dataset.shape:
                    dataset_size *= dim
                print(f"Dataset '{name}' dtype: {dataset.dtype} size: {dataset_size // 1024} KB")
                return dataset_size
            else:
                return 0

        # Initialize total size variable
        total_size = 0
        # Visit each object in the HDF5 file
        for name, item in f.items():
            total_size += visit_func(name, item)

        print(f"Total data size: {total_size // 1024} KB")


if __name__ == '__main__':
    h5_file_path = './sample/data.h5'
    patch_type = 'images'  # or features

    if not os.path.exists(os.path.split(h5_file_path)[0]):
        os.mkdir(os.path.split(h5_file_path)[0])

    # generate a pseudo patch list and location lost
    generated_patch_list = []
    generated_patch_loc_list = []
    generated_index_list = []
    y_num = 50
    x_num = 50

    patch_num = y_num * x_num
    y_list = np.argsort(np.random.rand(y_num))
    x_list = np.argsort(np.random.rand(x_num))

    for idx in range(patch_num):
        # fake patch imgs (PIL numpy: 0-255, RGB, hwc) or features
        patch = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        generated_patch_list.append(patch)
        generated_patch_loc_list.append((y_list[idx // y_num], x_list[idx - idx // y_num * y_num]))
        generated_index_list.append(idx)

    # pick and save a pseudo patch list and location list in to WSI path
    random_indices = random.sample(range(len(generated_index_list)), k=patch_num // 2)
    saved_patch_list = [generated_patch_list[i] for i in random_indices]
    saved_patch_loc_list = [generated_patch_loc_list[i] for i in random_indices]

    # save samples
    patch_list = []
    patch_loc_list = []
    for idx in tqdm(range(len(saved_patch_list))):
        if patch_type != 'images':  # represents embedded features
            patch_list.append(saved_patch_list[idx])
        else:
            patch_path = os.path.join(os.path.split(h5_file_path)[0], str(idx)+'.jpeg')
            image = Image.fromarray(saved_patch_list[idx]).convert("RGB")
            image.save(patch_path)

        patch_loc_list.append(saved_patch_loc_list[idx])

    # Then, make h5 file
    patch_list = None if patch_type == 'images' else patch_list  # set patch_list to None if it's images
    hdf5_save_a_WSI(h5_file_path, patch_list, patch_loc_list, patch_type=patch_type)

    # Then, read all valid coordinates from current wsi
    coord_list, coord_npy = hdf5_check_all_coord(h5_file_path)

    # we can take a patch_idx by its coordinate with coord_npy[Y, X]

    # (play) get valid coordinate index list
    # coord_valid_list_idx = [coord_npy[2, 4], coord_npy[2, 7], coord_npy[9, 4]]  # let's say I want idx 2,4,5
    coord_valid_list_idx = random.choices(range(len(saved_patch_list)), k=5)

    patch_list, patch_coord_list = hdf5_load_a_list_of_patches_by_id(
        h5_file_path, coord_valid_list_idx, patch_type=patch_type)

    for i in range(len(coord_valid_list_idx)):
        if patch_type != 'images':
            print(patch_list[i], patch_coord_list[i])
        else:
            WSI_path = os.path.split(h5_file_path)[0]
            save_patch_path = os.path.join(WSI_path, str(i) + '.jpeg')
            patch = Image.open(save_patch_path)
            print(patch, patch_coord_list[i])

    coord_valid_list_coord = [coord_list[x] for x in coord_valid_list_idx]  # just for example

