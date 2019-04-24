import torch
import model
from torch import optim

def loss_D(real_imgs, generated_imgs, dis):
    part1 = torch.mean((dis(real_imgs) - 1.0).pow(2))
    part2 = torch.mean(dis(generated_imgs).pow(2))
    return part1 + part2

def loss_G(generated_imgs, dis):
    return torch.mean((dis(generated_imgs) - 1.0).pow(2))

def loss_cyc(imgs_x, imgs_y, G, F):
    part1 = torch.mean((F(G(imgs_x)) - imgs_x).pow(2))
    part2 = torch.mean((G(F(imgs_y)) - imgs_y).pow(2))
    return part1 + part2

def train(G, F, Dx, Dy, lr, data_loader_x, data_loader_y, epochs, lmbda):
    optimizer_G = optim.Adam(G.parameters(), lr = lr)
    optimizer_F = optim.Adam(F.parameters(), lr = lr)
    optimizer_Dx = optim.Adam(Dx.parameters(), lr = lr)
    optimizer_Dy = optim.Adam(Dy.parameters(), lr = lr)
    for e in range(epochs):
        print(e)
        _iter_y = data_loader_y.__iter__()
        for _imgs_x, _ in data_loader_x:
            _imgs_y = _iter_y.__next__()[0]
            generated_x = F(_imgs_y)
            generated_y = G(_imgs_x)
            # train Ds
            Dx.zero_grad()
            loss_x = loss_D(_imgs_x, generated_x, Dx)
            loss_x.backward()
            optimizer_Dx.step()
            Dy.zero_grad()
            loss_y = loss_D(_imgs_y, generated_y, Dy)
            loss_y.backward()
            optimizer_Dy.step()
            # train G and F
            G.zero_grad()
            F.zero_grad()
            loss = loss_G(generated_x, Dx) + loss_G(generated_y, Dy) + lmbda * loss_cyc(_imgs_x, _imgs_y, G, F)
            loss.backward()
            optimizer_G.step()
            optimizer_F.step()

