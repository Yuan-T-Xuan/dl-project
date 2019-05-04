import numpy as np
import torch
import random

pool_x = None
pool_y = None

def add_to_generated_pool(pool_name, new_imgs, b=1, B=50):
    global pool_x
    global pool_y
    assert pool_name in ["pool_x", "pool_y"]
    #
    if pool_name == "pool_x" and type(pool_x) == type(None):
        pool_x = np.array(new_imgs.cpu().detach().numpy())
        return
    if pool_name == "pool_y" and type(pool_y) == type(None):
        pool_y = np.array(new_imgs.cpu().detach().numpy())
        return
    if pool_name == "pool_x" and pool_x.shape[0] < B:
        new_imgs_numpy = np.array(new_imgs.cpu().detach().numpy())
        pool_x = np.vstack((pool_x, new_imgs_numpy))
        if pool_x.shape[0] > B:
            np.random.shuffle(pool_x)
            pool_x = pool_x[:B]
        return
    if pool_name == "pool_y" and pool_y.shape[0] < B:
        new_imgs_numpy = np.array(new_imgs.cpu().detach().numpy())
        pool_y = np.vstack((pool_y, new_imgs_numpy))
        if pool_y.shape[0] > B:
            np.random.shuffle(pool_y)
            pool_y = pool_y[:B]
        return
    #
    pool = pool_x
    if pool_name == "pool_y":
        pool = pool_y
    new_imgs_numpy = np.array(new_imgs.cpu().detach().numpy())
    if b > 1:
        np.random.shuffle(new_imgs_numpy)
        new_imgs_numpy = new_imgs_numpy[:int(b/2)]
    np.random.shuffle(pool)
    if pool_name == "pool_x":
        if b > 1:
            pool_x = np.vstack((pool[int(b/2):], new_imgs_numpy))
        else:
            pool_x = np.vstack((pool[1:], new_imgs_numpy))
    else:
        if b > 1:
            pool_y = np.vstack((pool[int(b/2):], new_imgs_numpy))
        else:
            pool_y = np.vstack((pool[1:], new_imgs_numpy))

def get_batch_from_pool(pool_name, b=1):
    global pool_x
    global pool_y
    assert pool_name in ["pool_x", "pool_y"]
    pool = pool_x
    if pool_name == "pool_y":
        pool = pool_y
    index_to_select = list(range(len(pool)))
    random.shuffle(index_to_select)
    index_to_select = index_to_select[:b]
    tensor_to_return = torch.tensor(pool[index_to_select]).cuda()
    return tensor_to_return


