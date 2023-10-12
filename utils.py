import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import faiss
import ResNet
import random
from corruption import corrupt_image_cifar10
from torch.utils.data import Dataset
from imagecorruptions import corrupt
from PIL import Image

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

class NoiseAug(object):
    def __init__(self, aug=None, sev=1):
        self.aug = aug
        self.sev = sev
    def __call__(self, x):
        x = corrupt_image_cifar10(x, severity=self.sev, corruption_name=self.aug)
        return x

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
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

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