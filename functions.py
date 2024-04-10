from dataset import FashionMNISTDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from tqdm import tqdm
from pathlib import Path
from model_AE import AE

from loss_history import LossHistory
from torchvision import datasets

def dataloader_test():
    transform = transforms.Compose(
        [transforms.Resize(56), 
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))] # (0.1307,), (0.3081,) -> fits the data into -0.42, 2.82 which doesnt compatible with sigmoid, so I switched to 0.5,0.5 and tanh
    )

    trainset = datasets.FashionMNIST(
        "C:\\Users\\Robinumut\\Desktop\\archive", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(
        trainset, batch_size=64, shuffle=True
    )

    testset = datasets.FashionMNIST(
        "C:\\Users\\Robinumut\\Desktop\\archive", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(
        testset, batch_size=64, shuffle=False
    )

    return train_loader, test_loader


def create_dataloaders(train, test, NUMWORKERS, batch_size, use_gpu):

    data_transform = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.Resize(56),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) #mean and std of fashion mnist
    ])

    train_dataset = FashionMNISTDataset(train, transform=data_transform)
    test_dataset = FashionMNISTDataset(test, transform=data_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=NUMWORKERS, shuffle=True, pin_memory=use_gpu)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=NUMWORKERS, shuffle=False, pin_memory=use_gpu)

    print(""" Training and testing with:
            Autoencoder traning set size: {}
            Autoencoder testing set size: {}
            """.format(len(train_loader.dataset), len(test_loader.dataset)))

    return train_loader, test_loader

def train_epoch(model, dataloader, loss_fn, optimizer, vae=False):
    model.train()
    total_loss = 0.0
    for idx, (data, _) in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset)/dataloader.batch_size)):
        #img, _ = batch['image'], batch['label']
        #for idx, batch in
        img = data.to(model.device)
        
        if vae:
            _h, m, v = model(img)
            loss = loss_fn(_h, img, m, v)
        else:
            _h = model(img)
            loss = loss_fn(_h, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader.dataset)

def val_epoch(model, dataloader, loss_fn):
    with torch.no_grad():
        model.eval()
        total_loss = 0.0
        for data, _ in dataloader:
            #for batch in dataloader:
            #batch = batch['image'].to(model.device)
            img = data.to(model.device)
            _h = model(img)
            loss = loss_fn(_h, img)
            total_loss += loss.item()
            
    return total_loss / len(dataloader.dataset)

def init_model(model_type, models_dir, device, lat_space, variational):

    if model_type == 'ae':
        MODEL = AE
    elif model_type == 'vae':
        MODEL = AE
    model_name = '{}_ls{}.pt'.format(model_type, lat_space)
    metrics_name = '{}_ls{}.mt'.format(model_type, lat_space)

    modelpath = models_dir.joinpath(model_name)
    metricpath = models_dir.joinpath(metrics_name)
    if modelpath.exists() and metricpath.exists():
        model = load_model(modelpath, model_type, device)
        metrics = LossHistory.load(metricpath)
    else:
        model = MODEL(lat_space=lat_space, device=device, variational=variational)
        metrics = LossHistory(metricpath)
    
    return model, metrics

def load_model(modelpath, model_type, device, verbose=True):
    if model_type == 'ae':
        MODEL = AE
    elif model_type == 'vae':
        #MODEL = VAE
        MODEL = AE
    chk_dict = torch.load(modelpath)
    mstate = chk_dict['mstate']
    lat_space = chk_dict['lat-space']
    model = MODEL(lat_space, device)
    model.load_state_dict(mstate)

    if verbose:
        print('Loaded model from {}.'.format(modelpath))
    return model

if __name__ == '__main__':
    train_csv = "C:\\Users\\Robinumut\\Desktop\\archive\\fashion-mnist_train.csv"
    test_csv = "C:\\Users\\Robinumut\\Desktop\\archive\\fashion-mnist_test.csv"
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(56),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) #mean and std of fashion mnist
    ])

    train_dataset = FashionMNISTDataset(train_csv, transform=data_transform)
    test_dataset = FashionMNISTDataset(test_csv, transform=data_transform)

    print(len(train_dataset[5]['image']))
    print(len(train_dataset))