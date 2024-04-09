from dataset import FashionMNISTDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from tqdm import tqdm




def create_dataloaders(train, test, NUMWORKERS, batch_size=16):

    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(56),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) #mean and std of fashion mnist
    ])

    train_dataset = FashionMNISTDataset(train, transform=data_transform)
    test_dataset = FashionMNISTDataset(test, transform=data_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=NUMWORKERS, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=NUMWORKERS, shuffle=False)

    print(""" Training and testing with:
            Autoencoder traning set size: {}
            Autoencoder testing set size: {}
            """.format(len(train_loader.dataset), len(test_loader.dataset)))

    return train_loader, test_loader

def train_epoch(model, dataloader, loss_fn, optimizer):
    model.train()
    total_loss = 0.0
    for batch in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset)/dataloader.batch_size)):
        batch = batch['image'].to(model.device)
        _h = model(batch)
        loss = loss_fn(_h, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader.dataset)

def val_epoch(model, dataloader, loss_fn):
    with torch.no_grad():
        model.eval()
        total_loss = 0.0
        for batch in dataloader:
            batch = batch['image'].to(model.device)
            _h = model(batch)
            loss = loss_fn(_h, batch)
            total_loss += loss.item()
            
    return total_loss / len(dataloader.dataset)

