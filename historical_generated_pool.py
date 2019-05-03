import numpy as np
import torch
import random

pool_x = None
pool_y = None

def add_to_generated_pool(pool_name, new_imgs, b=100, B=200):
    global pool_x
    global pool_y
    assert pool_name in ["pool_x", "pool_y"]
    #
    if pool_name == "pool_x" and pool_x == None:
        pool_x = np.array(new_imgs.detach().numpy())
        return
    if pool_name == "pool_y" and pool_y == None:
        pool_y = np.array(new_imgs.detach().numpy())
        return
    if pool_name == "pool_x" and pool_x.shape[0] < B:
        new_imgs_numpy = np.array(new_imgs.detach().numpy())
        pool_x = np.vstack((pool_x, new_imgs_numpy))
        if pool_x.shape[0] > B:
            np.random.shuffle(pool_x)
            pool_x = pool_x[:B]
        return
    if pool_name == "pool_y" and pool_y.shape[0] < B:
        new_imgs_numpy = np.array(new_imgs.detach().numpy())
        pool_y = np.vstack((pool_y, new_imgs_numpy))
        if pool_y.shape[0] > B:
            np.random.shuffle(pool_y)
            pool_y = pool_y[:B]
        return
    #
    pool = pool_x
    if pool_name == "pool_y":
        pool = pool_y
    new_imgs_numpy = np.array(new_imgs.detach().numpy())
    np.random.shuffle(new_imgs_numpy)
    new_imgs_numpy = new_imgs_numpy[:(b/2)]
    np.random.shuffle(pool)
    if pool_name == "pool_x":
        pool_x = np.vstack((pool[(b/2):], new_imgs_numpy))
    else:
        pool_y = np.vstack((pool[(b/2):], new_imgs_numpy))
