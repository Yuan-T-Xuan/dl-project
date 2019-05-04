import numpy as np
import matplotlib.pyplot as plt
from data_loader import *
from model import *

dl_manga = get_dataloader(path="/Users/xuan/Downloads/manga_faces", batch_size=6)
dl_photo = get_dataloader(path="/Users/xuan/Downloads/photo_faces", batch_size=6)

g = torch.load("/Users/xuan/Downloads/saved_G_2.pt", map_location='cpu')
f = torch.load("/Users/xuan/Downloads/saved_F_2.pt", map_location='cpu')

manga_sample_y = dl_manga.__iter__().__next__()[0]
photo_sample_x = dl_photo.__iter__().__next__()[0]
generated_y = g(photo_sample_x)
generated_x = f(manga_sample_y)
re_generated_x = f(generated_y)
re_generated_y = g(generated_x)

for i in range(6):
    plt.imshow(photo_sample_x[i].permute(1, 2, 0).detach().numpy() * 0.5 + 0.5)
    plt.show()
    plt.imshow(re_generated_x[i].permute(1, 2, 0).detach().numpy() * 0.5 + 0.5)
    plt.show()

for i in range(6):
    plt.imshow(manga_sample_y[i].permute(1, 2, 0).detach().numpy() * 0.5 + 0.5)
    plt.show()
    plt.imshow(re_generated_y[i].permute(1, 2, 0).detach().numpy() * 0.5 + 0.5)
    plt.show()

