import torch
import torch.nn as nn

class CompactnessLoss(nn.Module):
    def __init__(self, center):
        super(CompactnessLoss, self).__init__()
        self.center = center

    def forward(self, inputs):
        m = inputs.size(1)
        variances = (inputs - self.center).norm(dim=1).pow(2) / m
        return variances.mean()


class EWCLoss(nn.Module):
    def __init__(self, frozen_model, fisher, lambda_ewc=1e4):
        super(EWCLoss, self).__init__()
        self.frozen_model = frozen_model
        self.fisher = fisher
        self.lambda_ewc = lambda_ewc

    def forward(self, cur_model):
        loss_reg = 0
        for (name, param), (_, param_old) in zip(cur_model.named_parameters(), self.frozen_model.named_parameters()):
            if 'fc' in name:
                continue
            loss_reg += torch.sum(self.fisher[name]*(param_old-param).pow(2))/2
        return self.lambda_ewc * loss_reg

def coral(x, y):
    
    if len(x) <= 1 or len(y) <= 1:
        return None
    
    mean_x = x.mean(0, keepdim=True)
    mean_y = y.mean(0, keepdim=True)
    cent_x = x - mean_x
    cent_y = y - mean_y

    cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
    cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

    mean_diff = (mean_x - mean_y).pow(2).mean()
    cova_diff = (cova_x - cova_y).pow(2).mean()

    return mean_diff + cova_diff