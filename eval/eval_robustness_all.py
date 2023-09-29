# settings
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import argparse
import utils
import json
from tqdm import tqdm
import os

import torchvision
from torchvision import transforms

from corruption.cifar10c import CIFAR10C, SplitInDataset, SplitOutDataset

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--epochs', default=50, type=int, metavar='epochs', help='number of epochs')
parser.add_argument('--label', default=0, type=int, help='The normal class')
parser.add_argument('--lr', type=float, default=1e-1, help='The initial learning rate.')
parser.add_argument('--resnet_type', default=152, type=int, help='which resnet to use')
parser.add_argument('--latent_dim_size', default=2048, type=int)
parser.add_argument('--batch_size', default=32, type=int)
args = parser.parse_args()

# corruption_names = get_corruption_names()
# corruption_names = ['clean'] + corruption_names
corruption_names = ['clean', 'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 
                    'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression', 'saturate', 'spatter']

# print(corruption_names)


transform_color = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def get_score(model, device, train_feature_space, testloader, class_idx=0):
    model.to(device)
    model.eval()
    test_labels = []
    test_feature_space = []
    with torch.no_grad():
        for (imgs, labels) in tqdm(testloader, desc='Test set feature extracting'):
            imgs = imgs.to(device)
            _, features = model(imgs)
            test_feature_space.append(features)
            batch_size = imgs.shape[0]
            for j in range(batch_size):
                test_labels.append(labels[j])
        test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()

    distances = utils.knn_score(train_feature_space, test_feature_space)

    # auc = roc_auc_score(test_labels, distances)

    return train_feature_space, distances

# torch.nn.Module.dump_patches = True
#---------------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    
for i in range(10):
    # pretrained_path = f'eval_model/normal/class{i}.ckpt' #ここをしっかり設定する
    pretrained_path = f'eval_model/oe/class{i}.ckpt' #ここをしっかり設定する

    model = torch.load(pretrained_path)
    
    transform = transform_color
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=False, transform=transform)
    # trainset = torch.utils.data.Subset(trainset, list(range(50000)))
    idx = np.array(trainset.targets) == i 
    trainset.data = trainset.data[idx]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1500, shuffle=False, num_workers=4)
    
    train_feature_space = []
    with torch.no_grad():
        for (imgs, _) in tqdm(trainloader, desc='Train set feature extracting'):
            imgs = imgs.to(device)
            _, features = model(imgs)
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()

    for s in range(1, 6):
        for k in corruption_names:
            print(f'Class {i}, corruption {k}, severity {s}')
            if k == "clean":
                testset_c = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform)
            else:
                testset_c = CIFAR10C(root='../data', corruption_name=k, severity=s, transform=transform)
                
            #in
            testset_in = SplitInDataset(testset_c, i)
            testloader_in = torch.utils.data.DataLoader(testset_in, batch_size=1500, shuffle=False, num_workers=4)
            model.eval()                
            _ , score_in = get_score(model, device, train_feature_space, testloader_in, class_idx=i)
            print(score_in.shape)
            # make save path
            save_dir = f'anomaly_score_PANDA-OE/{i}/{s}/in/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # save score list
            np.save(os.path.join(save_dir, f'{k}.npy'), score_in)
            
            #out
            testset_out = SplitOutDataset(testset_c, i)
            testloader_out = torch.utils.data.DataLoader(testset_out, batch_size=1500, shuffle=False, num_workers=4)
            model.eval()                
            _ , score_out = get_score(model, device, train_feature_space, testloader_out, class_idx=i)
            print(score_out.shape)
            # convert to numpy 
            # make save path
            save_dir = f'anomaly_score_PANDA-OE/{i}/{s}/out'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # save score list
            np.save(os.path.join(save_dir, f'{k}.npy'), score_out)

