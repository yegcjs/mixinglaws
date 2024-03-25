import os
import fire
import json
import torch
from collections import defaultdict
from utils import *

def load_loss(path, filter=True):
    losses = defaultdict(lambda: defaultdict(dict))   # domain -> ratio
    with open(path, "r") as f:
        lines = set(f.readlines())
    for line in lines:   
        model, data, loss = line.split()
        model = model.strip('/')
        step = int(model.split('/')[-1])
        if filter and int(model.split('/')[-2].split('-')[0].split('_')[-1]) > 900:
            continue
        ratio = "-".join(model.split('/')[-2].split('-')[1:])
        subdomain = data.split('/')[-1].split('.')[-2]
        losses[ratio][subdomain][step] = float(loss)
    # merge
    for domain, subdomains in DOMAINS_2_SUBDOMAINS.items():
        domain_norm = sum(valid_weight[subdomain] for subdomain in subdomains)
        for ratio in losses:
            for step in losses[ratio][subdomains[0]]:
                try:
                    losses[ratio][domain][step] = sum(losses[ratio][subdomain][step] * valid_weight[subdomain] / domain_norm for subdomain in subdomains)
                except:
                    continue
    return losses

class GetLoss:
    def __init__(self, train_data):
        homedir = os.path.dirname(os.path.abspath(__file__))
        self.train_data = train_data
        self.OBSERVED_LOSSES = {
            "70M": load_loss(f"../data/{train_data}/70M.txt"), # filter=False),
            "160M": load_loss(f"../data/{train_data}/160M.txt"), # filter=False),
            "305M": load_loss(f"../data/{train_data}/305M.txt"),
            "410M": load_loss(f"../data/{train_data}/410M.txt")
        }   # sz, ratio, domain, step
        self.steplaws = defaultdict(lambda:defaultdict(lambda:defaultdict))
        if os.path.exists(STEPLAW_FILES[train_data]):
            with open(STEPLAW_FILES[train_data], "r") as f:
                self.steplaws = json.load(f) # sz, ratio, domain
        self.sizelaws = defaultdict(lambda:defaultdict(lambda:defaultdict)) 
        if os.path.exists(SIZELAW_FILES[train_data]):
            with open(SIZELAW_FILES[train_data], "r") as f:
                self.sizelaws = json.load(f) # size, step, ratio, domain
        self.mixlaws = defaultdict(lambda:defaultdict(lambda:defaultdict))
        if os.path.exists(MIXLAW_FILES[train_data]):
            self.mixlaws = torch.load(MIXLAW_FILES[train_data])

    def load(self, size, filter=True):
        self.OBSERVED_LOSSES[size] = load_loss(f"{homedir}/data/{self.train_data}/{size}.txt", filter=filter)

    def get_loss(self, size, ratio, domain, step):
        try:
            loss = self.OBSERVED_LOSSES[size][ratio][domain][step]
        except:
            if size in self.steplaws:
                # print("from steplaw")
                token = BSZ * step
                step_param = self.steplaws[size][ratio][domain]
                logc, logd0, alpha = step_param["log_c"], step_param["log_d0"], step_param["alpha"]
                loss = power_law(token, torch.tensor([logc, logd0, alpha]))
            elif ratio in self.sizelaws:
                # print("from sizelaw")
                size_param = self.sizelaws[ratio][domain][str(step)]            
                logc, logn0, alpha = size_param["log_c"], size_param["log_n0"], size_param["alpha"]
                loss = power_law(MODEL_SIZES[size], torch.tensor([logc, logn0, alpha]))
            elif (size in self.mixlaws) and (domain in self.mixlaws[size]):
                # print("from mixlaw")
                param = self.mixlaws[size][domain][step]
                x = np.array([list(map(float, ratio.split('-')))])
                if param["type"] == "linear":
                    loss = mixture_law_2(x, param["params"]).item() # mixture_law(x, param["params"])
                elif param["type"] == "mlp":
                    mlp = MixtureMLP(); mlp.load_state_dict(param["params"])
                    loss = mlp(torch.tensor(x)).item()
                elif param["type"] == "mlp_adaboost":
                    regr = param["params"]
                    loss = regr.predict(x).item() 
                else:
                    raise NotImplementedError            
        return loss

def main(size, step, ratios="ratios/default"):
    with open(ratios, "r") as f:
        ratios = [line.strip() for line in f]
    
    for ratio in ratios:
        for domain in DOMAINS_2_SUBDOMAINS:
            print(size, ratio, domain, step, get_loss(size, ratio, domain, step))


if __name__ == '__main__':
    fire.Fire(main)