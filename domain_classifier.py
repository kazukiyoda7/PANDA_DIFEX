import torch
import utils
import torch.optim as optim
import argparse
from torchvision.utils import save_image
import optuna

def main(args):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = utils.get_loaders(dataset=args.dataset, label_class=args.label, batch_size=args.batch_size, args=args)

    model = utils.get_resnet_model(args.resnet_type, pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(args.domain_list))
    model = model.to(device)
    
    train(model, train_loader, test_loader, device, args)
    
    return model, device
    
    
def train(model, train_loader, test_loader, device, args):
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.00005, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(args.epochs):
        loss_all = 0
        
        for i, img_dict in enumerate(train_loader):
                
            img_list, domain_label_list = [], []

            for domain in args.domain_list:
                img_list.append(img_dict[domain][0])
                domain_label_list.append(img_dict[domain][1])
            
            imgs = torch.cat(img_list, dim=0).to(device)
            labels = torch.cat(domain_label_list, dim=0).to(device)
            
            optimizer.zero_grad()
            
            fefeatures, logits = model(imgs)
            
            loss = criterion(logits, labels)
            loss.backward()
            
            optimizer.step()
            
            loss_all += loss.item()
            
        acc_dict, acc_all = eval_domain_classification(model, device, args.domain_list)
        print(f'epoch {epoch}, loss {loss_all}, acc {acc_all}')



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

def objective(trial):
    # ハイパーパラメータの範囲を指定
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 64, log=True)
    epochs = trial.suggest_int("epochs", 15, 50)
    resnet_type = trial.suggest_categorical("resnet_type", [18])
    
    args = parser.parse_args()
    
    # argparseの代わりに、上記のハイパーパラメータをargsに設定
    # ここでは一部だけ変更しています。他の部分は固定の値やデフォルト値を使うことになります。
    args.lr = lr
    args.batch_size = batch_size
    args.epochs = epochs
    args.resnet_type = resnet_type

    # main関数を呼び出して学習
    model, device = main(args)
    
    # 精度を取得（ここでは簡単のため、すべてのエポックの平均精度を返すようにしていますが、最後のエポックの精度だけを返しても良い）
    acc_dict, acc_all = eval_domain_classification(model, device, args.domain_list)
    return acc_all

        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--diag_path', default='~/data/fisher_diagonal.pth', help='fim diagonal path')
    parser.add_argument('--ewc', action='store_true', help='Train with EWC')
    parser.add_argument('--epochs', default=15, type=int, metavar='epochs', help='number of epochs')
    parser.add_argument('--label', default=0, type=int, help='The normal class')
    parser.add_argument('--lr', type=float, default=1e-2, help='The initial learning rate.')
    parser.add_argument('--resnet_type', default=18, type=int, help='which resnet to use')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--batch_size_test', default=100, type=int)
    parser.add_argument('--output_dir', type=str, default=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--severity', type=int, default=1)
    parser.add_argument('--interval', type=int, default=5)
    parser.add_argument('--data_root', type=str, default='/home/yoda/data/')
    parser.add_argument('--domain', type=str, default='clean-gaussian_noise')
    
    args = parser.parse_args()
    
    utils.fix_seed(args.seed)
    
    # main(args)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)

    best_params = study.best_params

    with open("best_params.txt", "w") as f:
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")
            
    with open("all_trials.txt", "w") as f:
        for trial in study.trials:
            f.write(f"Trial {trial.number}\n")
            for param, value in trial.params.items():
                f.write(f"{param}: {value}\n")
            f.write(f"Value: {trial.value}\n\n")

