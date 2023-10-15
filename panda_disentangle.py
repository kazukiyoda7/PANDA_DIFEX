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
from matplotlib import pyplot as plt

from torch.utils.tensorboard import SummaryWriter

def train_model(model, model_dz, train_loader, test_loader, device, args, ewc_loss):
    model.eval()
    model_dz.eval()
    auc, feature_space = get_score(model, device, train_loader, test_loader, args.domain_list)
    print('Epoch: {}, AUROC is: {}'.format(0, auc))
    params_model = list(model.parameters())
    params_model_dz = list(model_dz.parameters())
    # optimizer = optim.SGD(params_model_dz, lr=args.lr, weight_decay=0.00005, momentum=0.9)
    optimizer = optim.SGD(params_model+params_model_dz, lr=args.lr, weight_decay=0.00005, momentum=0.9)
    center = torch.FloatTensor(feature_space).mean(dim=0) # 5000*512
    criterion = CompactnessLoss(center.to(device)) # 512
    criterion_ds = torch.nn.CrossEntropyLoss()
    criteiron_disentangle = torch.nn.CosineSimilarity(dim=1)
    model_save_dir = osp.join(args.output_dir, 'model')
    feature_save_dir = osp.join(args.output_dir, 'feature')
    loss_save_dir = osp.join(args.output_dir, 'loss')
    best_auc, best_epoch = 0, 0
    loss_list = []
    if not osp.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if not osp.exists(feature_save_dir):
        os.makedirs(feature_save_dir)
    if not osp.exists(loss_save_dir):
        os.makedirs(loss_save_dir)
    for epoch in range(args.epochs):
        running_loss, running_loss_dict, running_domain_loss, running_disentangle_loss = run_epoch(model, model_dz, train_loader, optimizer, criterion, criterion_ds, criteiron_disentangle, device, args.ewc, ewc_loss, args.domain_list)
        print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
        print(running_loss_dict)
        print('domain_loss:', running_domain_loss)
        print('disentangle_loss:', running_disentangle_loss)
        loss_list.append(running_loss_dict)
        auc, feature_space = get_score(model, device, train_loader, test_loader, args.domain_list)
        print('Epoch: {}, AUROC is: {}'.format(epoch + 1, auc))
        
        # domain classificationの精度を計算
        acc_dict, all_acc = eval_domain_classification(model_dz, device, args.domain_list)
        print(acc_dict, all_acc)
        
        if auc > best_auc:
            best_epoch = epoch
            best_auc = auc
            print('best_auc is updated')
            torch.save(model, osp.join(model_save_dir, 'model_best.pth'))
            np.save(osp.join(feature_save_dir, 'train_feature_best.npy'), feature_space)
        if (epoch+1) % args.interval == 0:
            torch.save(model, osp.join(model_save_dir, f'model_{epoch+1}.pth'))
            np.save(osp.join(feature_save_dir, f'train_feature_{epoch+1}.npy'), feature_space)
        args.writer.add_scalar('total loss', running_loss, epoch)
        args.writer.add_scalar('domain loss', running_domain_loss, epoch)
        args.writer.add_scalars('each loss', running_loss_dict, epoch)
        args.writer.add_scalars('each acc', acc_dict, epoch)
        args.writer.add_scalar('disentangle loss', running_disentangle_loss, epoch)
    # plot_loss_evolution(loss_list, osp.join(loss_save_dir, 'loss.png'))
        
    print(f'best_eposh is {best_epoch}')
        


def run_epoch(model, model_dz, train_loader, optimizer, criterion, criterion_ds, criterion_disentangle, device, ewc, ewc_loss, domain_list):
    running_loss = 0.0
    running_loss_dict = {}
    running_domain_loss = 0.0
    runnig_disentangle_loss = 0.0
    
    for domain in domain_list:
        running_loss_dict[domain] = 0.0
    for i, img_dict in enumerate(train_loader):
        
        loss_dict = {}
        for domain in domain_list:
            loss_dict[domain] = 0.0
        
        img_list = []
        label_list = []
        
        for domain in domain_list:
            img_list.append(img_dict[domain][0])
            label_list.append(img_dict[domain][1])      
            
        imgs = torch.cat(img_list, dim=0)

        images = imgs.to(device)
        labels = torch.cat(label_list, dim=0).to(device)

        optimizer.zero_grad()

        features, logits = model(images)
        
        num_per_class = features.shape[0]//len(domain_list)
        for i, domain in enumerate(domain_list):
            features_in_domain = features[num_per_class*i:num_per_class*(i+1)]
            loss_dict[domain] += criterion(features_in_domain)
            
            
        loss = sum(loss_dict.values())
        
        if len(domain_list) > 1:
            features_ds, logits_ds = model_dz(images)
            loss_ds = criterion_ds(logits_ds, labels)
            running_domain_loss += loss_ds.item()*args.alpha
            loss += loss_ds

        loss_disentangle = criterion_disentangle(features, features_ds).mean()*args.beta
        runnig_disentangle_loss += loss_disentangle.item()
        loss += loss_disentangle

        if ewc:
            loss += ewc_loss(model)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)

        optimizer.step()

        running_loss += loss.item()
        
        for domain in domain_list:
            running_loss_dict[domain] += loss_dict[domain].item()
            
    for domain in domain_list:
        running_loss_dict[domain] /= i + 1
    return running_loss / (i + 1), running_loss_dict, running_domain_loss / (i + 1), runnig_disentangle_loss / (i + 1)


def get_score(model, device, train_loader, test_loader, domain_list):
    train_feature_space = []
    with torch.no_grad():
        for img_dict in tqdm(train_loader, desc='Train set feature extracting'):
            img_list = []
            for domain in domain_list:
                img_list.append(img_dict[domain][0])
            imgs = torch.cat(img_list, dim=0)
            imgs = imgs.to(device)
            features, logit = model(imgs)
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
    test_feature_space = []
    with torch.no_grad():
        for (imgs, _) in tqdm(test_loader, desc='Test set feature extracting'):
            imgs = imgs.to(device)
            features, logit = model(imgs)
            test_feature_space.append(features)
        test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
        test_labels = test_loader.dataset.targets

    distances = utils.knn_score(train_feature_space, test_feature_space)
    auc = roc_auc_score(test_labels, distances)

    return auc, train_feature_space

def plot_loss_evolution(loss_data_list, save_path='loss_evolution.png'):
    # Assuming all dictionaries have the same keys/domains
    domains = list(loss_data_list[0].keys())

    for domain in domains:
        losses = [data[domain] for data in loss_data_list]
        plt.plot(losses, label=domain)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Evolution over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()
    
def eval_domain_classification(model_ds, device, domain_list):
    model_ds.eval()
    acc_dict = {}
    dataloaders = utils.get_domain_loaders(domain_list, args)
    all_acc = 0
    for domain, dataloader in dataloaders.items():
        n_samples = 0
        n_hits = 0
        for img, domain_label in dataloader:
            img = img.to(device)
            domain_label = domain_label.to(device)
            _, logit = model_ds(img)
            pred = torch.argmax(logit, dim=1)
            hits = (pred == domain_label).sum().item()
            n_hits += hits
            samples = len(domain_label)
            n_samples += samples
        acc = n_hits / n_samples
        acc_dict[domain] = acc
        all_acc += acc
    all_acc /= len(domain_list)
    return acc_dict, all_acc

def main(args):
    print('Dataset: {}, Normal Label: {}, LR: {}'.format(args.dataset, args.label, args.lr))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train_loader, test_loader = utils.get_loaders(dataset=args.dataset, label_class=args.label, batch_size=args.batch_size, args=args)
    
    print(device)
    model = utils.get_resnet_model(resnet_type=args.resnet_type)
    model = model.to(device)
    
    model_ds = utils.get_resnet_model(resnet_type=args.resnet_type, pretrained=False)
    model_ds.fc = torch.nn.Linear(model_ds.fc.in_features, len(args.domain_list))
    model_ds = model_ds.to(device)

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
    train_model(model, model_ds, train_loader, test_loader, device, args, ewc_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--diag_path', default='~/data/fisher_diagonal.pth', help='fim diagonal path')
    parser.add_argument('--ewc', action='store_true', help='Train with EWC')
    parser.add_argument('--epochs', default=15, type=int, metavar='epochs', help='number of epochs')
    parser.add_argument('--label', default=0, type=int, help='The normal class')
    parser.add_argument('--lr', type=float, default=1e-2, help='The initial learning rate.')
    parser.add_argument('--resnet_type', default=152, type=int, help='which resnet to use')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--batch_size_test', default=100, type=int)
    parser.add_argument('--output_dir', type=str, default=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise', type=str, default=None)
    parser.add_argument('--severity', type=int, default=3)
    parser.add_argument('--interval', type=int, default=5)
    parser.add_argument('--lr_fourier', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=1e-1)
    parser.add_argument('--beta', type=float, default=1e-1)
    parser.add_argument('--epochs_fourier', type=int, default=5)
    parser.add_argument('--data_root', type=str, default='~/data')
    parser.add_argument('--domain', type=str, default='clean')
    
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
        
    tensorboard_dir = osp.join(args.output_dir, 'tensorboard_logs')
    if not osp.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    writer = SummaryWriter(tensorboard_dir)
    args.writer = writer
        
    main(args)
