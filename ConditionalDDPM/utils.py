import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass, field
from typing import List, Tuple
from tqdm import tqdm
from DDPM import ConditionalDDPM
from torchvision.utils import make_grid
import torch
from torchvision import datasets
from torch.utils.data import DataLoader


torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class DMConfig:
    '''
    Define the model and experiment settings here
    '''
    input_dim: Tuple[int, int] = (28, 28) # input image size
    num_channels: int = 1                 # input image channels
    condition_mask_value: int = -1          # unconditional condition mask value 
    num_classes: int = 10                 # number of classes in the dataset
    T: int = 400                          # diffusion and denoising steps
    beta_1 = 1e-4                         # variance schedule
    beta_T = 2e-2
    mask_p = 0.1                          # unconditional condition drop ratio
    num_feat = 128                        # feature size of the UNet model
    omega: float = 2.0                    # conditional guidance weight

    batch_size: int = 256                 # training batch size
    epochs: int = 20                      # training epochs
    learning_rate: float = 1e-4           # training learning rate
    multi_lr_milestones: List[int] = field(default_factory=lambda: [20]) # learning rate decay milestone
    multi_lr_gamma: float = 0.1           # learning rate decay ratio
    
def make_dataloader(transform, batch_size, dir = './data', train = True):
    dataset = datasets.MNIST(root = dir, train = train, transform = transform, download = True)
    data_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = train)
    return data_loader

def make_dataloader_cifar10(transform, batch_size, dir = './data', train = True):
    dataset = datasets.CIFAR10(root = dir, train = train, transform=transform, download = True)
    data_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = train)
    return data_loader
        
def save_checkpoint(save_path, epoch, model, optimizer, dmconfig):
    save_ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'dmconfig': dmconfig
    }
    torch.save(save_ckpt, save_path)

class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v
    
def check_forward(dataloader, dmconfig, device):
    model = ConditionalDDPM(dmconfig).to(device)
    optimizer = optim.Adam(model.network.parameters(), lr = dmconfig.learning_rate)
    model.train()
    noise_loss_list = []
    for images, conditions in tqdm(dataloader, leave = False, desc = 'training'):
        images, conditions = images.to(device), conditions.to(device)
        noise_loss = model(images, conditions)

        noise_loss_list.append(noise_loss.item())
        optimizer.zero_grad()
        noise_loss.backward()
        optimizer.step()
    plt.figure(figsize = (6, 4))
    plt.plot(noise_loss_list)
    plt.xlabel('Iterations') 
    plt.ylabel('Noise Loss') 
    plt.title('Noise Loss Curve')
    plt.show()

    return model

def check_sample(model, dmconfig, device):
    model.eval()
    num_classes = dmconfig.num_classes
    omega = dmconfig.omega
    T = dmconfig.T
    conditions =  torch.arange(0, num_classes).to(device)
    conditions = torch.tile(conditions, [10, 1]).T.reshape((-1,))
    conditions_1hot = F.one_hot(conditions, num_classes = num_classes)
    generated_images = model.sample(conditions_1hot, omega).cpu()
    
    fig, axes = plt.subplots(num_classes, 10, figsize = (6, 6), gridspec_kw = {'hspace': 0, 'wspace': 0})
    plt.subplots_adjust(wspace=0, hspace=0)
    axes = axes.flatten()
    for i, _ in enumerate(generated_images):
        ax = axes[i]
        ax.imshow(generated_images[i].permute(1,2,0), cmap = 'gray')
        ax.axis('off')
    fig.tight_layout()
    return fig

def sample_images(config, checkpoint_path):
    num_classes = config.num_classes
    omega = config.omega
    T = config.T
    model = ConditionalDDPM(dmconfig = config).to(device)
    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    conditions =  torch.arange(0, num_classes).to(device)
    conditions = torch.tile(conditions, [10, 1]).T.reshape((-1,))
    conditions_1hot = F.one_hot(conditions, num_classes = num_classes)
    generated_images = model.sample(conditions_1hot, omega)
    generated_images = make_grid(generated_images, nrow=10, padding=0)
    generated_images = generated_images.permute(1, 2, 0).cpu().numpy()
    return generated_images
    

def plot_images(model, num_classes, omega, T):
    model.eval()
    conditions =  torch.arange(0, num_classes).to(device)
    conditions = torch.tile(conditions, [10, 1]).T.reshape((-1,))
    conditions_1hot = F.one_hot(conditions, num_classes = num_classes)

    generated_images = model.sample(conditions_1hot, omega).cpu()

    fig, axes = plt.subplots(num_classes, 10, figsize = (6, 6), gridspec_kw = {'hspace': 0, 'wspace': 0})
    plt.subplots_adjust(wspace=0, hspace=0)
    axes = axes.flatten()
    for i, _ in enumerate(generated_images):
        ax = axes[i]
        ax.imshow(generated_images[i].permute(1,2,0), cmap = 'gray')
        ax.axis('off')
    fig.tight_layout()
    return fig

def train(train_loader, model, optimizer):
    model.train()
    train_noise_loss = Averager()
    for images, conditions in tqdm(train_loader, leave = False, desc = 'training'):
        images, conditions = images.to(device), conditions.to(device)
        noise_loss = model(images, conditions)

        train_noise_loss.add(noise_loss.item())

        optimizer.zero_grad()
        noise_loss.backward()
        optimizer.step()

        noise_loss = None

    return train_noise_loss.item()

def test(test_loader, model):
    model.eval()
    test_noise_loss = Averager()
    with torch.no_grad():
        for images, conditions in tqdm(test_loader, leave = False, desc = 'test'):
            images, conditions = images.to(device), conditions.to(device)
            noise_loss = model(images, conditions)

            test_noise_loss.add(noise_loss.item())
            noise_loss = None

    return test_noise_loss.item()

def solver(dmconfig, exp_name, train_loader, test_loader):
    if not os.path.exists('./save'): os.mkdir('./save')
    exp_dir = os.path.join('./save', exp_name)
    if not os.path.exists(exp_dir): os.mkdir(exp_dir)
    image_dir = os.path.join(exp_dir, 'images')
    if not os.path.exists(image_dir): os.mkdir(image_dir)

    model = ConditionalDDPM(dmconfig).to(device)
    optimizer = optim.Adam(model.network.parameters(), lr = dmconfig.learning_rate)
    lr_scheduler = MultiStepLR(optimizer, 
                               milestones = dmconfig.multi_lr_milestones, 
                               gamma = dmconfig.multi_lr_gamma,)
    best_test_loss = 1e10
    
    for i in range(dmconfig.epochs):
        epoch = i + 1
        print('epoch {}/{}'.format(epoch, dmconfig.epochs))
        train_noise_loss = train(train_loader, model, optimizer)
        
        lr_scheduler.step()

        test_noise_loss = test(test_loader, model)
        print('train: train_noise_loss = {:.4f}'.format(train_noise_loss),
              'test: test_noise_loss = {:.4f}'.format(test_noise_loss))
        if test_noise_loss < best_test_loss:
            save_best_path = os.path.join(exp_dir, 'best_checkpoint.pth')
            save_checkpoint(save_best_path, epoch, model, optimizer, dmconfig)
        
        fig = plot_images(model, dmconfig.num_classes, dmconfig.omega, dmconfig.T)
        fig.savefig(os.path.join(image_dir, f'generate_epoch_{epoch}.png'))
        plt.close(fig)


