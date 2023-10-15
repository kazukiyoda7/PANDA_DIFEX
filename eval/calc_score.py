# settings
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import argparse
from utils import fix_seed
import utils
import json

import torchvision
from torchvision import transforms

import csv

from corruption.cifar10c import CIFAR10C, SplitInDataset, SplitOutDataset

from panda_difex import Net

# 指定したクラス，ドメイン，深刻度のデータローダーを取得する関数
def get_testloader(root, transform, corruption_name, severity, label):
    if corruption_name == "clean":
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=False, transform=transform)
    else:
        testset = CIFAR10C(root=root, corruption_name=corruption_name, severity=severity, transform=transform)
    testset = SplitInDataset(testset, label)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=4)
    return testloader

corruption_list = ['clean', 'gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'gaussian_blur', 'snow', 
                    'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression', 'saturate', 'spatter']

cifar10_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] 

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--epochs', default=50, type=int, metavar='epochs', help='number of epochs')
parser.add_argument('--label', default=0, type=int, help='The normal class')
parser.add_argument('--latent_dim_size', default=2048, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--severity', type=int, default=1)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--feature_path', type=str, required=True)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--eval_domain', type=str, default='clean')
parser.add_argument('--output_dir', type=str, default='.')
parser.add_argument('--data_root', type=str, default='~/data')
parser.add_argument('--method', type=str, default='vanilla')
args = parser.parse_args()

fix_seed(args.seed)

transform_color = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def get_score(model, device, train_feature_space, testloader, method, class_idx=0):
    model.to(device)
    model.eval()
    test_labels = []
    test_feature_space = []
    with torch.no_grad():
        for (imgs, labels) in testloader:
            imgs = imgs.to(device)
            features, _ = model(imgs)
            test_feature_space.append(features)
            batch_size = imgs.shape[0]
            for j in range(batch_size):
                test_labels.append(labels[j])
        test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()

    distances = utils.knn_score(train_feature_space, test_feature_space)

    return train_feature_space, distances

#---------------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def calc_score(model_path, train_feature_path, args, device):
    model = torch.load(model_path)
    model.eval()
    train_feature_space = np.load(train_feature_path)
    transform = transform_color
    
    corruption_names = args.eval_domain.split('-')
    if 'all' in corruption_names:
        corruption_names = corruption_list
    else:
        assert all(c in corruption_list for c in corruption_names), "corruption name is incorrect"
    print(corruption_names)
    
    for s in range(1, 6):
        for i in range(10):
            for k in corruption_list:
                print(f'severity {s}, Class {i}, corruption {k}')
                testloader = get_testloader(root=args.data_root, transform=transform, corruption_name=k, severity=s, label=i)
                model.eval()                
                _ , score_out = get_score(model, device, train_feature_space, testloader, args.method, class_idx=i)
                score_save_dir = os.path.join(args.output_dir, 'score', str(s), str(i))
                if not os.path.exists(score_save_dir):
                    os.makedirs(score_save_dir)
                np.save(os.path.join(score_save_dir, f'{k}.npy'), score_out)

calc_score(args.model_path, args.feature_path, args, device)
