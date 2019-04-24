import torch
from torchvision import datasets, transforms

def get_dataloader(path, batch_size, num_workers=4):
    trans = transforms.Compose([
        transforms.Resize(96),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    dataset = datasets.ImageFolder(path, trans)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

