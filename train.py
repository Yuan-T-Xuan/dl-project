import torch
import model
from model import init_weights
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import optim

def loss_D(real_imgs, generated_imgs, dis):
    part1 = torch.mean((dis(real_imgs) - 1.0).pow(2))
    part2 = torch.mean(dis(generated_imgs).pow(2))
    return (part1 + part2) / 2.0

def loss_G(generated_imgs, dis):
    return torch.mean((dis(generated_imgs) - 1.0).pow(2))

def loss_cyc(generated_x, generated_y, imgs_x, imgs_y, G, F):
    part1 = torch.mean((F(generated_y) - imgs_x).pow(2))
    part2 = torch.mean((G(generated_x) - imgs_y).pow(2))
    return part1 + part2

def train(G, F, Dx, Dy, lr, data_loader_x, data_loader_y, epochs, lmbda, e_offset=0):
    optimizer_G = optim.Adam(G.parameters(), lr = lr)
    optimizer_F = optim.Adam(F.parameters(), lr = lr)
    optimizer_Dx = optim.Adam(Dx.parameters(), lr = lr)
    optimizer_Dy = optim.Adam(Dy.parameters(), lr = lr)
    for e in range(epochs):
        print(e)
        if e % 2 == 0:
            torch.save(G, "/content/gdrive/My Drive/model2_test2/saved_G_" + str(e_offset+e) + ".pt")
            torch.save(F, "/content/gdrive/My Drive/model2_test2/saved_F_" + str(e_offset+e) + ".pt")
            torch.save(Dx, "/content/gdrive/My Drive/model2_test2/saved_Dx_" + str(e_offset+e) + ".pt")
            torch.save(Dy, "/content/gdrive/My Drive/model2_test2/saved_Dy_" + str(e_offset+e) + ".pt")
            #
        _iter_y = data_loader_y.__iter__()
        for_plot_GF = list()
        for_plot_Dx = list()
        for_plot_Dy = list()
        _iter_x2 = data_loader_x.__iter__()
        _iter_y2 = data_loader_y.__iter__()
        for _imgs_x, _ in tqdm(data_loader_x):
            _imgs_y = _iter_y.__next__()[0]
            _imgs_x = _imgs_x.cuda()
            _imgs_y = _imgs_y.cuda()
            generated_x = F(_imgs_y)
            generated_y = G(_imgs_x)
            # train Ds
            Dx.zero_grad()
            loss_x = loss_D(_imgs_x, generated_x, Dx)
            for_plot_Dx.append(loss_x.item())
            loss_x.backward()
            optimizer_Dx.step()
            Dy.zero_grad()
            loss_y = loss_D(_imgs_y, generated_y, Dy)
            for_plot_Dy.append(loss_y.item())
            loss_y.backward()
            optimizer_Dy.step()
            # train G and F
            _imgs_x = _iter_x2.__next__()[0]
            _imgs_y = _iter_y2.__next__()[0]
            _imgs_x = _imgs_x.cuda()
            _imgs_y = _imgs_y.cuda()
            generated_x = F(_imgs_y)
            generated_y = G(_imgs_x)
            G.zero_grad()
            F.zero_grad()
            loss = loss_G(generated_x, Dx) + loss_G(generated_y, Dy) + lmbda * loss_cyc(generated_x, generated_y, _imgs_x, _imgs_y, G, F)
            for_plot_GF.append(loss.item())
            loss.backward()
            optimizer_G.step()
            optimizer_F.step()
        plt.plot(for_plot_Dx)
        plt.show()
        plt.plot(for_plot_Dy)
        plt.show()
        plt.plot(for_plot_GF)
        plt.show()
