import torch
import model
from model import init_weights
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import optim
from historical_generated_pool import *

def loss_D(real_imgs, generated_imgs, D):
    part1 = torch.mean((D(real_imgs) - 1.0).pow(2))
    part2 = torch.mean(D(generated_imgs).pow(2))
    return (part1 + part2) / 2.0

def loss_G(generated_imgs, D):
    return torch.mean((D(generated_imgs) - 1.0).pow(2))

def loss_cyc(generated_x, generated_y, imgs_x, imgs_y, G, F):
    part1 = torch.mean(torch.abs(F(generated_y) - imgs_x))
    part2 = torch.mean(torch.abs(G(generated_x) - imgs_y))
    return part1 + part2

def train(G, F, Dx, Dy, lr, data_loader_x, data_loader_y, epochs, lmbda, e_offset=0):
    optimizer_G = optim.Adam(G.parameters(), lr = lr)
    optimizer_F = optim.Adam(F.parameters(), lr = lr)
    optimizer_Dx = optim.Adam(Dx.parameters(), lr = lr)
    optimizer_Dy = optim.Adam(Dy.parameters(), lr = lr)
    for_plot_G = list()
    for_plot_F = list()
    for_plot_Dx = list()
    for_plot_Dy = list()
    for e in range(epochs):
        print(e)
        if e % 2 == 0:
            torch.save(G, "/content/gdrive/My Drive/model3_test1/saved_G_" + str(e_offset+e) + ".pt")
            torch.save(F, "/content/gdrive/My Drive/model3_test1/saved_F_" + str(e_offset+e) + ".pt")
            torch.save(Dx, "/content/gdrive/My Drive/model3_test1/saved_Dx_" + str(e_offset+e) + ".pt")
            torch.save(Dy, "/content/gdrive/My Drive/model3_test1/saved_Dy_" + str(e_offset+e) + ".pt")
            #
        _iter_y = data_loader_y.__iter__()
        for _imgs_x, _ in tqdm(data_loader_x):
            _imgs_y = _iter_y.__next__()[0]
            _imgs_x = _imgs_x.cuda()
            _imgs_y = _imgs_y.cuda()
            # optimize G
            generated_y = G(_imgs_x)
            generated_x = F(_imgs_y)
            G.zero_grad()
            loss = 10.0 * loss_cyc(generated_x, generated_y, _imgs_x, _imgs_y, G, F) + loss_G(generated_y, Dy)
            for_plot_G.append(loss.item())
            loss.backward()
            optimizer_G.step()
            add_to_generated_pool("pool_y", generated_y)
            # optimize Dy
            generated_y = get_batch_from_pool("pool_y")
            Dy.zero_grad()
            loss = loss_D(_imgs_y, generated_y, Dy)
            for_plot_Dy.append(loss.item())
            loss.backward()
            optimizer_Dy.step()
            # optimize F
            generated_y = G(_imgs_x)
            generated_x = F(_imgs_y)
            F.zero_grad()
            loss = 10.0 * loss_cyc(generated_x, generated_y, _imgs_x, _imgs_y, G, F) + loss_G(generated_x, Dx)
            for_plot_F.append(loss.item())
            loss.backward()
            optimizer_F.step()
            add_to_generated_pool("pool_x", generated_x)
            # optimize Dx
            generated_x = get_batch_from_pool("pool_x")
            Dx.zero_grad()
            loss = loss_D(_imgs_x, generated_x, Dx)
            for_plot_Dx.append(loss.item())
            loss.backward()
            optimizer_Dx.step()
        plt.plot(for_plot_Dx)
        plt.show()
        plt.plot(for_plot_Dy)
        plt.show()
        plt.plot(for_plot_G)
        plt.show()
        plt.plot(for_plot_F)
        plt.show()
