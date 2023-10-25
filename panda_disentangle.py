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

import optuna

def train_model(model, model_ds, model_bc, train_loader, test_loader, device, args, ewc_loss):
    model.eval()
    model_ds.eval()
    # auc, feature_space = get_score(model, device, train_loader, test_loader, args.domain_list)
    auc_mix, feature_space = get_score_in_mix_domain(model, device, train_loader, args.domain_list, args)
    print('Epoch: {}, AUROC is: {}'.format(0, auc_mix))
    params_model = list(model.parameters())
    params_model_ds = list(model_ds.parameters())
    params_model_bc = list(model_bc.parameters())
    # optimizer = optim.SGD(params_model_ds, lr=args.lr, weight_decay=0.00005, momentum=0.9)
    optimizer = optim.SGD(params_model, lr=args.lr, weight_decay=0.00005, momentum=0.9)
    # optimizer = optim.SGD(params_model+params_model_ds, lr=args.lr, weight_decay=0.00005, momentum=0.9)
    # optimizer = optim.Adam(params_model+params_model_ds+params_model_bc, lr=args.lr, weight_decay=0.00005)
    center = torch.FloatTensor(feature_space).mean(dim=0) # 5000*512
    criterion = CompactnessLoss(center.to(device)) # 512
    criterion_ds = torch.nn.CrossEntropyLoss()
    criteiron_disentangle = torch.nn.CosineSimilarity(dim=1)
    criterion_bc = torch.nn.BCELoss()
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
        running_loss, running_loss_dict, running_domain_loss, running_disentangle_loss = run_epoch(model, model_ds, model_bc, train_loader, optimizer, criterion, criterion_ds, criteiron_disentangle, device, args.ewc, ewc_loss, args.domain_list)
        print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
        print(running_loss_dict)
        print('domain_loss:', running_domain_loss)
        print('disentangle_loss:', running_disentangle_loss)
        loss_list.append(running_loss_dict)
        # auc, feature_space = get_score(model, device, train_loader, test_loader, args.domain_list)
        auc_mix, _ = get_score_in_mix_domain(model, device, train_loader, args.domain_list, args)
        print('Epoch: {}, AUROC is: {}'.format(epoch + 1, auc_mix))
        
        # domain classificationの精度を計算
        if len(args.domain_list) > 1:
            acc_dict, all_acc = eval_domain_classification(model_ds, device, args.domain_list)
        print(acc_dict, all_acc)
        
        if auc_mix > best_auc:
            best_epoch = epoch
            best_auc = auc_mix
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
    
    return best_auc, all_acc
        


def run_epoch(model, model_ds, model_bc, train_loader, optimizer, criterion, criterion_ds, criterion_disentangle, device, ewc, ewc_loss, domain_list):
    running_loss = 0.0
    running_loss_dict = {}
    running_domain_loss = 0.0
    running_disentangle_loss = 0.0
    
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
        for j, domain in enumerate(domain_list):
            features_in_domain = features[num_per_class*j:num_per_class*(j+1)]
            loss_dict[domain] += criterion(features_in_domain)
            
            
        loss = sum(loss_dict.values())
        
        if len(domain_list) > 1:
            features_ds, logits_ds = model_ds(images)
            loss_ds = criterion_ds(logits_ds, labels)
            running_domain_loss += loss_ds.item()*args.alpha
            loss += loss_ds

            loss_disentangle = criterion_disentangle(features, features_ds).mean()*args.beta
            running_disentangle_loss += loss_disentangle.item()
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

def get_score_in_mix_domain(model, device, train_loader, domain_list, args):
    id_loaders, ood_loaders = utils.get_domain_testloaders(args.data_root, domain_list, args.severity, args.label)
    
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
    
    id_feature_space = []
    for id_loader in id_loaders:
        with torch.no_grad():
            for imgs, _ in tqdm(id_loader, desc='ID set feature extracting'):
                imgs = imgs.to(device)
                features, logit = model(imgs)
                id_feature_space.append(features)
    id_feature_space = torch.cat(id_feature_space, dim=0).contiguous().cpu().numpy()
    id_labels = np.zeros(id_feature_space.shape[0])
        
    ood_feature_space = []
    for ood_loader in ood_loaders:
        with torch.no_grad():
            for imgs, _ in tqdm(ood_loader, desc='OOD set feature extracting'):
                imgs = imgs.to(device)
                features, logit = model(imgs)
                ood_feature_space.append(features)
    ood_feature_space = torch.cat(ood_feature_space, dim=0).contiguous().cpu().numpy()
    ood_labels = np.ones(ood_feature_space.shape[0])
    
    test_feature_space = np.concatenate([id_feature_space, ood_feature_space])
    test_labels = np.concatenate([id_labels, ood_labels])

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
    dataloaders = utils.get_domain_loaders(domain_list, args.label, args)
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
    args.output_dir = osp.join(args.output, args.day)
    args.output_dir = osp.join(args.output_dir, f"{args.now.hour}-{args.now.minute}-{args.now.second}")
    print(f"Output Directory : {args.output_dir}")
    sys.stdout = Logger(fpath=args.output_dir)
    save_json_path = os.path.join(args.output_dir, "config_args.json")
        # argsのコピーを作成
    args_dict = vars(args).copy()
    # SummaryWriterオブジェクトを除外
    if "writer" in args_dict:
        del args_dict["writer"]
    if "now" in args_dict:
        del args_dict["now"]
    # JSONとしてファイルにダンプ
    with open(save_json_path, "w") as f:
        json.dump(args_dict, f, indent=2)
        
    tensorboard_dir = osp.join(args.output_dir, 'tensorboard_logs')
    if not osp.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    writer = SummaryWriter(tensorboard_dir)
    args.writer = writer
    
    print('Dataset: {}, Normal Label: {}, LR: {}'.format(args.dataset, args.label, args.lr))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train_loader, test_loader = utils.get_loaders(dataset=args.dataset, label_class=args.label, batch_size=args.batch_size, args=args)
    
    print(device)
    model = utils.get_resnet_model(resnet_type=args.resnet_type)
    model = model.to(device)
    
    model_ds = utils.get_resnet_model(resnet_type=args.resnet_type, pretrained=False)
    model_ds.fc = torch.nn.Linear(model_ds.fc.in_features, len(args.domain_list))
    model_ds = model_ds.to(device)
    
    model_bc = utils.get_resnet_model(resnet_type=args.resnet_type, pretrained=False)
    model_bc.fc = torch.nn.Linear(model_bc.fc.in_features, len(args.domain_list))
    model_bc = model_bc.to(device)

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
    best_auc, all_acc = train_model(model, model_ds, model_bc, train_loader, test_loader, device, args, ewc_loss)
    
    return best_auc, all_acc
    
    
def objective(trial):
    # 3. ハイパーパラメータの範囲を指定します
    epochs = trial.suggest_int("epochs", 1, 30)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    alpha = trial.suggest_float("alpha", 1e-2, 1e2, log=True)
    beta = trial.suggest_float("beta", 1e-2, 1e2, log=True)
    
    # 上記のハイパーパラメータを使用してモデルをトレーニングします
    args.epochs = epochs
    args.lr = lr
    args.batch_size = batch_size
    args.alpha = alpha
    args.beta = beta
    
    # モデルをトレーニングし、評価します
    auc, acc = main(args)
    
    return auc  # 最大化したい値を返します


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
    parser.add_argument('--alpha', type=float, default=1e-1)
    parser.add_argument('--beta', type=float, default=1e-1)
    parser.add_argument('--epochs_fourier', type=int, default=5)
    parser.add_argument('--data_root', type=str, default='~/data')
    parser.add_argument('--domain', type=str, default='clean')
    parser.add_argument('--optuna', action='store_true')
    parser.add_argument('--optuna_dir', type=str, default='optuna_results')
    parser.add_argument('--n_trials', type=int, default=10)
    
    args = parser.parse_args()
    
    args.output = args.output_dir
    now = datetime.datetime.now()
    args.now = now
    args.day = f"{now.year}-{now.month}-{now.day}"
    
    fix_seed(args.seed)
    
    if args.optuna:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.n_trials)
        
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
                
        best_params = study.best_trial.params
        
        # 全ての試行の結果とそのパラメータを保存するリストを作成
        all_trials = []
        for trial in study.trials:
            trial_data = {
                'value': trial.value,
                'params': trial.params
            }
            all_trials.append(trial_data)
        
        now = datetime.datetime.now()
        output_path = osp.join(args.optuna_dir, f"{now.year}-{now.month}-{now.day}", f"{now.hour}-{now.minute}-{now.second}")
        if not osp.exists(output_path):
            os.makedirs(output_path)
        
        # ベストなパラメータを保存
        with open(osp.join(output_path, 'best_hyperparameters.json'), 'w') as f:
            json.dump(best_params, f, indent=4)
        
        # 全ての試行の結果を保存
        with open(osp.join(output_path, 'all_trials_results.json'), 'w') as f:
            json.dump(all_trials, f, indent=4)

    else:
        main(args)