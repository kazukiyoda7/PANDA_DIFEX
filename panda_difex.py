import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import argparse
from losses import CompactnessLoss, EWCLoss, coral
import torch.nn.functional as F
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
from matplotlib import pyplot as plt

from difex.network import common_network
from torch import nn

from torch.utils.tensorboard import SummaryWriter


def train_model(teacher_net, student_net, train_loader, test_loader, device, args, ewc_loss):
    teacher_net.eval()
    student_net.eval()
    auc, feature_space = get_score(student_net, device, train_loader, test_loader, args.domain_list)
    print('Epoch: {}, AUROC is: {}'.format(0, auc))
    optimizer = optim.SGD(student_net.parameters(), lr=args.lr, weight_decay=0.00005, momentum=0.9)
    center = torch.FloatTensor(feature_space).mean(dim=0) # 5000*512
    criterion = CompactnessLoss(center.to(device)) # 512
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
        loss, loss1, loss2, loss3, loss4 = run_epoch_fix(teacher_net, student_net, train_loader, optimizer, criterion, device, args.ewc, ewc_loss, args.batch_size, args.domain_list)
        print('Epoch: {}, Loss: {}, Loss1: {}, Loss2: {}, Loss3: {}, Loss4: {}'.format(epoch + 1, loss, loss1, loss2, loss3, loss4))
        auc, feature_space = get_score(student_net, device, train_loader, test_loader, args.domain_list)
        print('Epoch: {}, AUROC is: {}'.format(epoch + 1, auc))
        if auc > best_auc:
            best_epoch = epoch
            best_auc = auc
            print('best_auc is updated')
            torch.save(student_net, osp.join(model_save_dir, 'model_best.pth'))
            np.save(osp.join(feature_save_dir, 'train_feature_best.npy'), feature_space)
        if (epoch+1) % args.interval == 0:
            torch.save(student_net, osp.join(model_save_dir, f'model_{epoch+1}.pth'))
            np.save(osp.join(feature_save_dir, f'train_feature_{epoch+1}.npy'), feature_space)
        args.writer.add_scalar('total loss', loss, epoch)
        args.writer.add_scalars('losses', {'loss1': loss1, 'loss2':loss2, 'loss3': loss3, 'loss4': loss4}, epoch)
    print(f'best_epoch is {best_epoch}')
    
    
def train_model_fourier(model, train_loader, test_loader, device, args, ewc_loss):
    model.eval()
    auc, feature_space = get_score_fourier(model, device, train_loader, test_loader, args.domain_list
                                           )
    print('Epoch: {}, AUROC is: {}'.format(0, auc))
    optimizer = optim.SGD(model.parameters(), lr=args.lr_fourier, weight_decay=0.00005, momentum=0.9)
    center = torch.FloatTensor(feature_space).mean(dim=0)
    criterion = CompactnessLoss(center.to(device))
    # model_save_dir = osp.join(args.output_dir, 'model')
    # feature_save_dir = osp.join(args.output_dir, 'feature')
    best_auc, best_epoch = 0, 0
    # if not osp.exists(model_save_dir):
    #     os.makedirs(model_save_dir)
    # if not osp.exists(feature_save_dir):
    #     os.makedirs(feature_save_dir)
    for epoch in range(args.epochs_fourier):
        running_loss, running_loss_dict = run_epoch_fourier(model, train_loader, optimizer, criterion, device, args.ewc, ewc_loss, args.domain_list)
        print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
        print(running_loss_dict)
        auc, feature_space = get_score(model, device, train_loader, test_loader, args.domain_list)
        print('Epoch: {}, AUROC is: {}'.format(epoch + 1, auc))
        if auc > best_auc:
            best_epoch = epoch+1
            best_auc = auc
            print('best_auc is updated')
        #     torch.save(model, osp.join(model_save_dir, 'model_best.pth'))
        #     np.save(osp.join(feature_save_dir, 'train_feature_best.npy'), feature_space)
        # if (epoch+1) % args.interval == 0:
        #     torch.save(model, osp.join(model_save_dir, f'model_{epoch+1}.pth'))
        #     np.save(osp.join(feature_save_dir, f'train_feature_{epoch+1}.npy'), feature_space)
        args.writer.add_scalar('fourier loss', running_loss, epoch)
        
    print(f'best_epoch is {best_epoch}')


def run_epoch(model, train_loader, optimizer, criterion, device, ewc, ewc_loss):
    running_loss = 0.0
    for i, (imgs, _) in enumerate(train_loader):
        images = imgs.to(device)
        optimizer.zero_grad()
        features, _ = model(images)
        loss = criterion(features)
        if ewc:
            loss += ewc_loss(model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)
        optimizer.step()
        running_loss += loss.item()
    return running_loss / (i + 1)

def run_epoch_fix(teachernet, student_net, train_loader, optimizer, criterion, device, ewc, ewc_loss, batch_size, domain_list):
    running_loss, running_loss1, running_loss2, running_loss3, running_loss4 = 0.0, 0.0, 0.0, 0.0, 0.0
    running_loss_dict = {}
    for domain in domain_list:
        running_loss_dict[domain] = 0.0
    teachernet.eval()
    for i, img_dict in enumerate(train_loader):
        
        img_list = []
        for domain in domain_list:
            img_list.append(img_dict[domain])    
        imgs = torch.cat(img_list, dim=0)
        
        with torch.no_grad():
            imgs = imgs.to(device)
            imgs_phase = torch.angle(torch.fft.fftn(imgs, dim=(2, 3)))
            _, teacher_output = teachernet(imgs_phase)
            teacher_output = teacher_output.detach()
        images = imgs.to(device)
        
        optimizer.zero_grad()
        student_features, student_output = student_net(images)
        
        num_per_domain = student_features.shape[0]//len(domain_list)
        
        loss4 = None            
        if len(domain_list) > 1:
            loss4 = 0
            for i in range(len(domain_list) - 1):
                for j in range(i+1, len(domain_list)):
                    loss4 += coral(student_features[i*num_per_domain:(i+1)*num_per_domain, teachernet.bottleneck_output:],
                                    student_features[j*num_per_domain:(j+1)*num_per_domain, teachernet.bottleneck_output:])
            loss4 = loss4*2/(len(domain_list)*(len(domain_list)-1))*args.theta        
        else:
            x = student_features[:batch_size//2, teachernet.bottleneck_output:]
            y = student_features[batch_size//2:, teachernet.bottleneck_output:]
            if len(x) > 1 and len(y) > 1:
                loss4 = coral(student_features[:batch_size//2, teachernet.bottleneck_output:], student_features[batch_size//2:, teachernet.bottleneck_output:])*args.theta
            
        loss1 = criterion(student_features)
        loss2 = F.mse_loss(teacher_output, student_output[:, :256])*args.alpha
        loss3 = -F.mse_loss(student_output[:, :teachernet.bottleneck_output], student_output[:, teachernet.bottleneck_output:])*args.beta
        
        if loss4 is not None:
            loss = loss1 + loss2 + loss3 + loss4
        else:
            loss = loss1 + loss2 + loss3
        
        if ewc:
            loss += ewc_loss(student_features)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student_net.parameters(), 1e-3)
        optimizer.step()
        running_loss1 += loss1.item()
        running_loss2 += loss2.item()
        running_loss3 += loss3.item()
        running_loss += loss1.item() + loss2.item() + loss3.item()
        if loss4 is not None:
            running_loss4 += loss4.item()
            running_loss += loss4.item()
    return running_loss/(i+1), running_loss1/(i+1), running_loss2/(i+1), running_loss3/(i+1), running_loss4/(i+1)

def run_epoch_fourier(model, train_loader, optimizer, criterion, device, ewc, ewc_loss, domain_list):
    running_loss = 0.0
    running_loss_dict = {}
    for domain in domain_list:
        running_loss_dict[domain] = 0.0
    for i, img_dict in enumerate(train_loader):
        img_list = []
        for domain in domain_list:
            img_list.append(img_dict[domain])            
        imgs = torch.cat(img_list, dim=0)
        images = imgs.to(device)
        images = torch.angle(torch.fft.fftn(images, dim=(2, 3)))
        optimizer.zero_grad()
        features, _ = model(images)
        loss_dict = {}
        for domain in domain_list:
            loss_dict[domain] = 0.0
        num_per_class = features.shape[0]//len(domain_list)
        for i, domain in enumerate(domain_list):
            features_in_domain = features[num_per_class*i:num_per_class*(i+1)]
            loss_dict[domain] += criterion(features_in_domain)
        loss = sum(loss_dict.values())
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
    return running_loss / (i + 1), running_loss_dict


def get_score(model, device, train_loader, test_loader, domain_list):
    model.eval()
    train_feature_space = []
    with torch.no_grad():
        for img_dict in tqdm(train_loader, desc='Train set feature extracting'):
            img_list = []
            for domain in domain_list:
                img_list.append(img_dict[domain])            
            imgs = torch.cat(img_list, dim=0)
            imgs = imgs.to(device)
            features, _ = model(imgs)
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
    test_feature_space = []
    with torch.no_grad():
        for (imgs, _) in tqdm(test_loader, desc='Test set feature extracting'):
            imgs = imgs.to(device)
            features, _ = model(imgs)
            test_feature_space.append(features)
        test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
        test_labels = test_loader.dataset.targets
    distances = utils.knn_score(train_feature_space, test_feature_space)
    auc = roc_auc_score(test_labels, distances)
    return auc, train_feature_space


def get_score_fourier(model, device, train_loader, test_loader, domain_list):
    model.eval()
    train_feature_space = []
    with torch.no_grad():
        for img_dict in tqdm(train_loader, desc='Fourier Train set featire extracting'):
            img_list = []
            for domain in domain_list:
                img_list.append(img_dict[domain])     
            imgs = torch.cat(img_list, dim=0)

            imgs = imgs.to(device)
            imgs = torch.angle(torch.fft.fftn(imgs, dim=(2, 3)))
            features, _ = model(imgs)
            train_feature_space.append(features)
        train_feature_space = (torch.cat(train_feature_space, dim=0)).contiguous().cpu().numpy()
    test_feature_space = []
    with torch.no_grad():
        for (imgs, _) in tqdm(test_loader, desc='Fourier Test set feature extracting'):
            imgs = imgs.to(device)
            imgs = torch.angle(torch.fft.fftn(imgs, dim=(2, 3)))
            features, _ = model(imgs)
            test_feature_space.append(features)
        test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
        test_labels = test_loader.dataset.targets
    distances = utils.knn_score(train_feature_space, test_feature_space)
    auc = roc_auc_score(test_labels, distances)
    return auc, train_feature_space


class Net(nn.Module):
    def __init__(self, resnet_type, bottleneck_input, bottleneck_output, layer_type='bn'):
        super(Net, self).__init__()
        self.featurizer = utils.get_resnet_model(resnet_type=resnet_type)
        self.bottleneck = common_network.feat_bottleneck(bottleneck_input, bottleneck_output, layer_type)
        self.bottleneck_output = bottleneck_output
        
    def forward(self, x):
        feature = self.featurizer(x)
        output = self.bottleneck(feature)
        return feature, output
        

def main(args):
    print('Dataset: {}, Normal Label: {}, LR: {}'.format(args.dataset, args.label, args.lr))
    ewc_loss = None
    
    # データローダーの取得
    train_loader, test_loader = utils.get_loaders(dataset=args.dataset, label_class=args.label, batch_size=args.batch_size, args=args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # teacher_net
    teacher_net = Net(args.resnet_type, 512, 256, 'bn')
    teacher_net.to(device)
    
    # teacher_featurizerの訓練
    train_model_fourier(teacher_net, train_loader, test_loader, device, args, ewc_loss)
    
    # student_net
    student_net = Net(args.resnet_type, 512, 512, 'bn')
    student_net.to(device)
    
    # Freezing Pre-trained model for EWC
    # if args.ewc:
    #     frozen_model = deepcopy(student_featurizer).to(device)
    #     frozen_model.eval()
    #     utils.freeze_model(frozen_model)
    #     fisher = torch.load(args.diag_path)
    #     ewc_loss = EWCLoss(frozen_model, fisher)
    # utils.freeze_parameters(student_featurizer)
    
    train_model(teacher_net, student_net, train_loader, test_loader, device, args, ewc_loss)

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
    parser.add_argument('--output_dir', type=str, default=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise', type=str, default=None)
    parser.add_argument('--severity', type=int, default=3)
    parser.add_argument('--interval', type=int, default=5)
    parser.add_argument('--lr_fourier', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=1e-1)
    parser.add_argument('--beta', type=float, default=1e-1)
    parser.add_argument('--theta', type=float, default=1e-1)
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
