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
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--eval_domain', type=str, default='clean')
parser.add_argument('--output_dir', type=str, default='.')
parser.add_argument('--data_root', type=str, default='~/data')
args = parser.parse_args()

fix_seed(args.seed)

transform_color = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#---------------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def eval_semantic_shift(id_class, args, device, corruption_list):
    corruption_names = args.eval_domain.split('-')
    if 'all' in corruption_names:
        corruption_names = corruption_list
    else:
        assert all(c in corruption_list for c in corruption_names), "corruption name is incorrect"
    print(corruption_names)
    
    corruption_list = corruption_names
    
    transform = transform_color
    
    for s in range(1, 6):
        ood_labels = list(filter(lambda x: x!=id_class, range(10)))

        results = {}

        for ood_label in ood_labels:
            score_path = os.path.join(args.output_dir, 'score', str(s), str(ood_label), f'clean.npy')
            score_out = np.load(score_path)
            results[ood_label] = {}
            
            for k in corruption_names:
                    
                #in
                score_path = os.path.join(args.output_dir, 'score', str(s), str(id_class), f'{k}.npy')
                score_in = np.load(score_path)
                
                #calc auc of each domain
                scores = np.concatenate([score_in, score_out])
                labels = np.concatenate([np.zeros(score_in.shape[0]), np.ones(score_out.shape[0])])
                auroc = roc_auc_score(labels, scores)
                
                results[ood_label][k] = auroc
                
            # 平均値の計算と追加
            avg_auroc = sum(results[ood_label].values()) / len(corruption_names)
            results[ood_label]["average"] = avg_auroc

        # csvの保存
        csv_save_dir = os.path.join(args.output_dir, 'csv', str(id_class))
        if not os.path.exists(csv_save_dir):
            os.makedirs(csv_save_dir)
        filename = os.path.join(csv_save_dir, f"auroc_results_{cifar10_class[id_class]}_severity_{s}.csv")
        
        with open(filename, "w", newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

            # ヘッダーを書き込み
            headers = ["OOD/Corruption"] + corruption_names + ['average']
            csvwriter.writerow(headers)
            
            # 各行のデータを書き込み
            for ood_label, aurocs in results.items():
                row = [cifar10_class[ood_label]] + [f"{aurocs[c]:.4f}" for c in corruption_names+['average']]
                csvwriter.writerow(row)
                
            print('result_outputed')
        
eval_semantic_shift(args.label, args, device ,corruption_list)