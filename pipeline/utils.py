import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from sklearn.ensemble import AdaBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from tqdm import tqdm 

homedir = os.path.dirname(os.path.abspath(__file__))
WEIGHT_FILE = "valid_weight.json"
STEPLAW_FILES = {
    "RedPajama": f"../data/RedPajama/steplaws.json",
}
SIZELAW_FILES = {
    "RedPajama": f"../data/RedPajama/sizelaws.json",
}
MIXLAW_FILES = {
    "RedPajama": f"../data/RedPajama/mixlaws.pt",
}

# DOMAINS_2_SUBDOMAINS = {
#     "academic": ["ArXiv", "PubMed_Abstracts", "PhilPapers", "NIH_ExPorter", "FreeLaw", "USPTO_Backgrounds", "PubMed_Central"],
#     "prose": ["PG19", "Books3", "BookCorpus2"],
#     "dialogue": ["Ubuntu_IRC", "OpenSubtitles", "EuroParl", "Enron_Emails", "HackerNews", "YoutubeSubtitles"],
#     "symbolic": ["DM_Mathematics", "Github"],
#     "internet": ["Pile-CC", "OpenWebText2", "StackExchange", "Wikipedia_en"],
# }
DOMAINS_2_SUBDOMAINS = {
    "subset_0": ["Github", "PubMed_Abstracts", "OpenWebText2", "EuroParl"],
    "subset_1": ["Pile-CC", "DM_Mathematics", "Wikipedia_en", "ArXiv", "USPTO_Backgrounds"],
    "subset_2": ["HackerNews", "PG19", "PubMed_Central",  "Ubuntu_IRC"],
    "subset_3": ["PhilPapers", "FreeLaw", "OpenSubtitles", "NIH_ExPorter", "BookCorpus2"],
    "subset_4": ["StackExchange", "Books3", "YoutubeSubtitles", "Enron_Emails"]
}
SUBDOMAINS_2_DOMAINS = {
    subdomain: domain
    for domain, subdomains in DOMAINS_2_SUBDOMAINS.items() for subdomain in subdomains
}
with open(WEIGHT_FILE, "r") as f:
    valid_weight = json.load(f)
for domain, subdomains in DOMAINS_2_SUBDOMAINS.items():
    valid_weight[domain] = sum(valid_weight[subdomain] for subdomain in subdomains)

MODEL_SIZES = {
    "70M": 18915328,
    "160M": 85056000,
    "305M": 201541632,
    "410M": 302311424,
    "1B": 805736448,
    "2.8B": 2517652480
}
# MODEL_SIZES = {
#     "70M": 12*6*512*512,
#     "160M": 12*12*768*768,
#     "305M": 12*16*1024*1024,
#     "410M": 12*24*1024*1024,
#     "1B": 12*16*2048*2048,
#     # "2.8B": 2517652480
# }
MODEL_FLOPS = {
    "70M": 6 * (6 * 512 * 512 + 4096 * 512 ),
    "160M": 12 * (6 * 768 * 768 + 4096 * 768 ), 
    "305M": 16 * (6 * 1024 * 1024 + 4096 * 1024 ), 
    "410M": 24 * (6 * 1024 * 1024 + 4096 * 1024 ), 
    "1B": 16 * (6 * 2048 * 2048 + 4096 * 2048 ),
    "2.8B": 32 * (6 * 2560 * 2560 + 4096 * 2560 ), 
}
PALLETES = [
    "#5865f2", "#57f287", "#eb459e", "#ed4245", "#520099", "#2c2f33",
    "#4AA2D9", "#D9CE32", "#F2A950", "#F24B4B", "#fd5c63"
]
BSZ = 4096 * 256

class ScalingLawWrap:
    def __init__(self, func):
        self.func = func
    
    def __call__(self, x, param):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
        if not isinstance(param, torch.Tensor):
            param = torch.tensor(param, dtype=torch.float)
        return self.func(x, param)

# def wrapped_power_law(func, x, param):
#     if not isinstance(x, torch.Tensor):
#         x = torch.tensor(x)
#     if not isinstance(param, torch.Tensor):
#         param = torch.tensor(param)
#     return func(x, param)

# def scaling_law_wrap(func):
#     return partial(wrapped_power_law, func)

def power_law_(x, param):
    log_c, log_k, alpha = param
    return torch.exp(log_c) + torch.exp(alpha * (log_k - torch.log(x)))
power_law = ScalingLawWrap(power_law_)

def power_law_share_alpha_(x, param):
    num_curves = (len(param) - 1) // 2
    log_c, log_k, alpha = param[:num_curves], param[num_curves:-1], param[-1]
    log_reducible_loss = alpha * (log_k[None, :] - x.log())
    return torch.exp(log_c) + torch.exp(log_reducible_loss)
power_law_share_alpha = ScalingLawWrap(power_law_share_alpha_)

def mixture_law_(x, param):
    log_c, log_k, t = param[0], param[1], param[2:]
    return torch.exp(log_c) + torch.exp(log_k + torch.matmul(x, t))
# def mixture_law_2(x, param):
#     c, k, t = param[0], param[1], param[2:]
#     return c + k * torch.exp(torch.matmul(x, t))
mixture_law = ScalingLawWrap(mixture_law_)

def mixture_law_one_domain(k, x, param):
    log_c, log_k, t = param[0], param[1], param[2:]
    return torch.exp(log_c) + torch.exp(log_k + x[:, k] * t[k])
# def mixture_law_2(x, param):
#     c, k, t = param[0], param[1], param[2:]
#     return c + k * torch.exp(torch.matmul(x, t))
mixture_law = ScalingLawWrap(mixture_law_)


def mixture_law_2_(x, param):
    result = 1
    c_0, param = param[0], param[1:]
    for i in range(len(x[0])):
        log_c, log_k, t = param[i*3:(i+1)*3]
        result *= torch.exp(log_c) + torch.exp(log_k + t * x[:, i])
    return result + c_0
mixture_law_2 = ScalingLawWrap(mixture_law_2_)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MixtureMLP(nn.Module):
    def __init__(self, dim=10, num_mixture=5, activation="exp", bias=None) -> None:
        super().__init__()
        self.in_linear = nn.Linear(num_mixture, dim)
        # for i in range(self.in_linear.weight.data.shape[0]):
        #     self.in_linear.weight.data[i, i%num_mixture] = -10 * np.abs(self.in_linear.weight.data[i, i%num_mixture]) 
        self.out_linear = nn.Linear(dim, 1)
        self.act = activation
        if bias is not None:
            self.out_linear.bias = nn.Parameter(torch.tensor([bias], dtype=torch.float))
        # self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float)
        h = self.in_linear(x)
        if self.act == "exp":
            h = torch.exp(h)
        elif self.act == "relu":
            h = torch.nn.functional.relu(h)
        elif self.act == "gelu":
            h = torch.nn.functional.gelu(h, approximate="tanh")
        elif self.act == "silu":
            h = torch.nn.functional.silu(h)
        elif self.act == "tanh":
            h = torch.nn.functional.tanh(h)
        elif self.act == "sigmoid":
            h = torch.nn.functional.sigmoid(h)
        elif self.act == "softplus":
            h = torch.nn.functional.softplus(h)
        # return self.out_linear(torch.nn.functional.gelu(self.in_linear(x)))
        w = F.softmax(self.out_linear.weight, dim=-1)
        # import ipdb;ipdb.set_trace()
        return torch.matmul(h, w.T) + self.out_linear.bias

class MLPEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, dim, num_mixture, activation):
        # Initialize parameters
        self.dim, self.num_mixture, self.activation = dim, num_mixture, activation
        self.mlp = MixtureMLP(dim, num_mixture, activation)
        self.use_cuda = torch.cuda.is_available()

    def fit(self, X, y):
        best_valid_loss, valid_split = 1e10, int(X.shape[0] / 5)
        X, y = torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float)
        self.mlp, X, y = self.mlp.cuda(), X.cuda(), y.cuda()
        
        optimizer = torch.optim.AdamW(self.mlp.parameters(), lr=0.1, weight_decay=1e-3)
        for _ in tqdm(range(5000)):
            self.mlp.train()
            # loss = optimizer.step(closure)
            loss = torch.nn.functional.huber_loss(self.mlp(torch.tensor(X[:-valid_split]).float()).squeeze(), torch.tensor(y[:-valid_split]).float(), delta=0.03)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                self.mlp.eval()
                valid_loss = torch.nn.functional.l1_loss(self.mlp(torch.tensor(X).float()).squeeze(), torch.tensor(y).float())
                # valid_loss = torch.nn.functional.mse_loss(mlp(torch.tensor(x[-5:]).float()).squeeze(), torch.tensor(y[-5:]).float())
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_param = self.mlp.state_dict()
        self.mlp.load_state_dict(best_param)

    @torch.no_grad()
    def predict(self, X):
        self.mlp.eval()
        if self.use_cuda:
            self.mlp = self.mlp.cuda()
            x = torch.tensor(X, dtype=torch.float, device='cuda')
            result = self.mlp(x).squeeze().cpu().numpy()
            self.mlp = self.mlp.cpu()
        else:
            x = torch.tensor(X, dtype=torch.float)
            result = self.mlp(x).squeeze().numpy()
        return result