import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import argparse
from losses import CompactnessLoss, EWCLoss
import utils
from copy import deepcopy
from tqdm import tqdm
from logger import Logger
import sys
import os
import json
import datetime
import os.path as osp
from utils import fix_seed
from torchvision.utils import save_image

def train_model(model, train_loader, test_loader, device, args, ewc_loss):
    model.eval()
    auc, feature_space = get_score(model, device, train_loader, test_loader)
    print('Epoch: {}, AUROC is: {}'.format(0, auc))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.00005, momentum=0.9)
    center = torch.FloatTensor(feature_space).mean(dim=0) # 5000*512
    # torch.save(center, 'train_center.pth')
    criterion = CompactnessLoss(center.to(device)) # 512
    model_save_dir = osp.join(args.output_dir, 'model')
    feature_save_dir = osp.join(args.output_dir, 'feature')
    best_auc, best_epoch = 0, 0
    if not osp.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if not osp.exists(feature_save_dir):
        os.makedirs(feature_save_dir)
    for epoch in range(args.epochs):
        running_loss = run_epoch(model, train_loader, optimizer, criterion, device, args.ewc, ewc_loss)
        print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
        auc, feature_space = get_score(model, device, train_loader, test_loader)
        print('Epoch: {}, AUROC is: {}'.format(epoch + 1, auc))
        if auc > best_auc:
            best_epoch = epoch
            best_auc = auc
            print('best_auc is updated')
            torch.save(model, osp.join(model_save_dir, 'model_best.pth'))
            np.save(osp.join(feature_save_dir, 'train_feature_best.npy'), feature_space)
        if (epoch+1) % args.interval == 0:
            torch.save(model, osp.join(model_save_dir, f'model_{epoch+1}.pth'))
            np.save(osp.join(feature_save_dir, f'train_feature_{epoch+1}.npy'), feature_space)
        
    print(f'best_eposh is {best_epoch}')
        


def run_epoch(model, train_loader, optimizer, criterion, device, ewc, ewc_loss):
    running_loss = 0.0
    for i, (imgs, _) in enumerate(train_loader):

        images = imgs.to(device)

        optimizer.zero_grad()

        features = model(images)

        loss = criterion(features)

        if ewc:
            loss += ewc_loss(model)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / (i + 1)


def get_score(model, device, train_loader, test_loader):
    train_feature_space = []
    with torch.no_grad():
        for (imgs, _) in tqdm(train_loader, desc='Train set feature extracting'):
            imgs = imgs.to(device)
            features = model(imgs)
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
    test_feature_space = []
    with torch.no_grad():
        for (imgs, _) in tqdm(test_loader, desc='Test set feature extracting'):
            imgs = imgs.to(device)
            features = model(imgs)
            test_feature_space.append(features)
        test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
        test_labels = test_loader.dataset.targets

    distances = utils.knn_score(train_feature_space, test_feature_space)
    auc = roc_auc_score(test_labels, distances)

    return auc, train_feature_space

def main(args):
    print('Dataset: {}, Normal Label: {}, LR: {}'.format(args.dataset, args.label, args.lr))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = utils.get_resnet_model(resnet_type=args.resnet_type)
    model = model.to(device)

    ewc_loss = None

    # Freezing Pre-trained model for EWC
    if args.ewc:
        frozen_model = deepcopy(model).to(device)
        frozen_model.eval()
        utils.freeze_model(frozen_model)
        fisher = torch.load(args.diag_path)
        # for name, _ in model.named_parameters():
        #     print(name, _.shape, fisher[name].shape)
        ewc_loss = EWCLoss(frozen_model, fisher)

        
        

    utils.freeze_parameters(model)
    train_loader, test_loader = utils.get_loaders(dataset=args.dataset, label_class=args.label, batch_size=args.batch_size, args=args)
    train_model(model, train_loader, test_loader, device, args, ewc_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--diag_path', default='../data/fisher_diagonal.pth', help='fim diagonal path')
    parser.add_argument('--ewc', action='store_true', help='Train with EWC')
    parser.add_argument('--epochs', default=15, type=int, metavar='epochs', help='number of epochs')
    parser.add_argument('--label', default=0, type=int, help='The normal class')
    parser.add_argument('--lr', type=float, default=1e-2, help='The initial learning rate.')
    parser.add_argument('--resnet_type', default=152, type=int, help='which resnet to use')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--output_dir', type=str, default=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise', type=str, default=None)
    parser.add_argument('--severity', type=int, default=3)
    parser.add_argument('--interval', type=int, default=5)
    parser.add_argument('--data_root', type=str, default='~/data')
    
    args = parser.parse_args()
    
    fix_seed(args.seed)
    
    save_dir = args.output_dir
    now = datetime.datetime.now()
    args.output_dir = osp.join(args.output_dir, f"{now.year}-{now.month}-{now.day}", f"{now.hour}-{now.minute}-{now.second}")
    # args.output_dir = os.path.join(save_dir, str(args.label))
    print(f"Output Directory : {args.output_dir}")
    sys.stdout = Logger(fpath=args.output_dir)
    save_json_path = os.path.join(args.output_dir, "config_args.json")
    with open(save_json_path, "w") as f:
        json.dump(vars(args), f, indent=2)
        
        main(args)
