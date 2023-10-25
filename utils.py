import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import faiss
import ResNet
import random
from torch.utils.data import Dataset
from imagecorruptions import corrupt
from PIL import Image
import os
from corruption.cifar10c import CIFAR10C

mvtype = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper']

corruption_tuple = ("gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
                    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
                    "brightness", "contrast", "elastic_transform", "pixelate",
                    "jpeg_compression", "speckle_noise", "gaussian_blur", "spatter",
                    "saturate")

corruption_list = list(corruption_tuple)
corruption_list.remove("spatter")
corruption_tuple = tuple(corruption_list)

transform_list = [transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

transform_color = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform_gray = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.Grayscale(num_output_channels=3),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])


def get_resnet_model(resnet_type=152, pretrained=True):
    """
    A function that returns the required pre-trained resnet model
    :param resnet_number: the resnet type
    :return: the pre-trained model
    """
    if resnet_type == 18:
        return ResNet.resnet18(pretrained=pretrained, progress=True)
    elif resnet_type == 50:
        return ResNet.wide_resnet50_2(pretrained=pretrained, progress=True)
    elif resnet_type == 101:
        return ResNet.resnet101(pretrained=pretrained, progress=True)
    else:  #152
        return ResNet.resnet152(pretrained=pretrained, progress=True)


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return

def freeze_parameters(model, train_fc=False):
    for p in model.conv1.parameters():
        p.requires_grad = False
    for p in model.bn1.parameters():
        p.requires_grad = False
    for p in model.layer1.parameters():
        p.requires_grad = False
    for p in model.layer2.parameters():
        p.requires_grad = False
    if not train_fc:
        for p in model.fc.parameters():
            p.requires_grad = False

def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)

def get_outliers_loader(batch_size):
    dataset = torchvision.datasets.ImageFolder(root='./data/tiny', transform=transform_color)
    outlier_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return outlier_loader

def get_loaders(dataset, label_class, batch_size, args):
    
    # 訓練に使用するドメイン名の取得と表示
    if args.domain is None:
        domain_list = []
    elif args.domain == "all":
        domain_list = list(corruption_tuple+('clean',))
    else:
        domain_list = args.domain.split('-')
    args.domain_list = domain_list
    domain_num = len(domain_list)
    print(domain_list)
    assert all(noise in corruption_tuple+('clean',) for noise in domain_list), "corruption name is incorrect"
    
    if dataset in ['cifar10']:
        if dataset == "cifar10":
            ds = torchvision.datasets.CIFAR10
            transform = transform_color
            coarse = {}
            trainset = ds(root=args.data_root, train=True, download=True, transform=transform, **coarse)
            testset = ds(root=args.data_root, train=False, download=True, transform=transform, **coarse)
    
        idx = np.array(trainset.targets) == label_class
        testset.targets = [int(t != label_class) for t in testset.targets]
        trainset.data = trainset.data[idx]
        trainset.targets = [trainset.targets[i] for i, flag in enumerate(idx, 0) if flag]
        
        data_num = trainset.data.shape[0]
        all_idx = list(range(data_num))
        idx_dict = {}
        
        # ドメインごとのインデックスを生成しつつ，データを取得
        for domain in domain_list:
            idx_array = random.sample(all_idx, data_num//(domain_num))
            idx_array = np.sort(idx_array)
            all_idx = [x for x in all_idx if x not in idx_array]
            idx_dict[domain] = np.array(idx_array)
            
        trainset = CustomDataset(idx_dict, trainset, args.domain_list, transform_list, severity=1)
        
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size//domain_num, shuffle=True, num_workers=2, drop_last=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)
        return train_loader, test_loader
    else:
        print('Unsupported Dataset')
        exit()

def clip_gradient(optimizer, grad_clip):
    assert grad_clip>0, 'gradient clip value must be greater than 1'
    for group in optimizer.param_groups:
        for param in group['params']:
            # gradient
            if param.grad is None:
                continue
            param.grad.data.clamp_(-grad_clip, grad_clip)

def fix_seed(seed):    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CustomDataset(Dataset):
    def __init__(self, idx_dict, trainset, domain_list, transform_list=None, severity=1):
        self.data = trainset.data
        self.targets = trainset.targets
        self.idx_dict = idx_dict
        self.severity = severity
        self.transform = transforms.Compose(transform_list)
        self.domain_num = len(list(idx_dict.keys()))
        self.domain_list = domain_list

    def __len__(self):
        return len(self.data)//self.domain_num

    def __getitem__(self, idx):
        imgs = {}
        for key, value in self.idx_dict.items():
            img = self.data[value[idx]]
            if not key == "clean":    
                img = np.array(img)
                img = corrupt(img, corruption_name=key, severity=self.severity)
            img = Image.fromarray(img)
            img = self.transform(img)
            domain_label = self.domain_list.index(key)
            imgs[key] = img, domain_label
        return imgs
    

class DomainDataset(Dataset):
    def __init__(self, id_class, domain_list, domain, transform_list=None, severity=1, data_root='~/data'):
        self.domain = domain
        self.domain_list = domain_list
        if transform_list is not None:
            self.transform = transforms.Compose(transform_list)
        else:
            self.transform = None
        self.severity = severity
        
        if domain == 'clean':
            dataset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=False)
            data = dataset.data
            label = dataset.targets
        else:
            path = os.path.join(data_root, 'CIFAR-10-C', domain+'.npy')
            data = np.load(path)
            data = data[10000*(severity-1):10000*(severity-1)+10000]
            label_path = os.path.join(data_root, 'CIFAR-10-C', 'labels.npy')
            label = np.load(label_path)
            label = label[10000*(severity-1):10000*(severity-1)+10000]
        idx = np.array(label) == id_class
        data = data[idx]
        self.data = data
            
        
    def __len__(self):
        return len(self.data)
            
    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img)
        domain_label = self.domain_list.index(self.domain)
        if self.transform is not None:
            img = self.transform(img)
        return img, domain_label
    
def get_domain_loaders(domain_list, id_class, args):
    dataloaders = {}
    for domain in domain_list:
        dataset = DomainDataset(id_class, domain_list, domain, transform_list=transform_list, severity=args.severity, data_root=args.data_root)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size_test, shuffle=True, num_workers=2, drop_last=False)
        dataloaders[domain] = dataloader
    return dataloaders

def get_each_domain_testloader(root, corruption_name, severity, label, id=True):
    transform = transform_color
    if corruption_name == "clean":
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=False, transform=transform)
    else:
        testset = CIFAR10C(root=root, corruption_name=corruption_name, severity=severity, transform=transform)
    if id:
        testset = SplitInDataset(testset, label)
    else:
        testset = SplitOutDataset(testset, label)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=4)
    return testloader

def get_domain_testloaders(root, domain_list, severity, label):
    id_loaders = []
    ood_loaders = []
    for domain in domain_list:
        id_loader = get_each_domain_testloader(root, domain, severity, label, id=True)
        id_loaders.append(id_loader)
        ood_loader = get_each_domain_testloader(root, domain, severity, label, id=False)
        ood_loaders.append(ood_loader)
    return id_loaders, ood_loaders

class SplitInDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, id_class):
        self.data = dataset.data[np.array(dataset.targets) ==  id_class]
        self.targets = np.array([0]*self.data.shape[0])
        self.transform = dataset.transform
        self.target_transform = dataset.target_transform
        self.id_class = id_class

    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]
        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)
        if self.target_transform is not None: 
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        # 2つのデータセットのサイズが同じであることを前提として、サイズを返す
        return len(self.targets)
    
    
class SplitOutDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, id_class):
        self.data = dataset.data[np.array(dataset.targets) !=  id_class]
        self.targets = np.array([1]*self.data.shape[0])
        self.transform = dataset.transform
        self.target_transform = dataset.target_transform
        self.id_class = id_class

    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]
        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)
        if self.target_transform is not None: 
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        # 2つのデータセットのサイズが同じであることを前提として、サイズを返す
        return len(self.targets)