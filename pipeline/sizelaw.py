import os
import fire
import json
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

from get_loss import GetLoss

from law import ScalingLaw
from utils import  *

import warnings
warnings.filterwarnings('ignore')

def param_generator_share_alpha(log_c):
    for log_N0 in np.linspace(15, 25, 20):
        for alpha in np.linspace(0.3, 0.8, 10):
            for _ in range(50):
                yield [_log_c + (1+np.random.rand()) for _log_c in log_c] + [log_N0 + np.random.rand() - 0.5 for _ in log_c] + [alpha]
      
def main(savefig, fit_sizes, target_size, train_data, step, ratios=None, tie_alpha="all", seed=42, variable="size"):
    global MODEL_SIZES
    MODEL_SIZES = MODEL_SIZES if variable != "flops" else MODEL_FLOPS
    # if variable == "flops":
    #     MODEL_SIZES = MODEL_FLOPS
    set_seed(seed)
    fit_sizes = fit_sizes.split(',')
    # OBSERVED_LOSSES = GetLoss(train_data).OBSERVED_LOSSES
    SIZELAW_FILE = SIZELAW_FILES[train_data]
    STEPLAW_FILE = STEPLAW_FILES[train_data]
    with open(STEPLAW_FILE, "r") as f:
        steplaws = json.load(f)
    token = step * BSZ
    if ratios is not None:
        with open(ratios, "r") as f:
            ratios = [line.strip() for line in f]
    else:
        ratios = steplaws[fit_sizes[0]]

        
    sizelaws = defaultdict(lambda: defaultdict(lambda:defaultdict(dict)))
    if os.path.exists(SIZELAW_FILE):
        with open(SIZELAW_FILE, "r") as f:
            sizelaws.update(json.load(f))
    
    fig, axes = plt.subplots(len(ratios), 2, figsize=(15, 60))
    
    if tie_alpha == "all":
        i = -1
        indices = {(ratio, domain): (i:=i+1)
            for ratio in ratios for domain in DOMAINS_2_SUBDOMAINS
        }
        x = np.array([[MODEL_SIZES[sz] for sz in fit_sizes] for ratio in ratios for domain in DOMAINS_2_SUBDOMAINS]).T
        y = np.array([
            [
                power_law(token, (steplaws[sz][ratio][domain]["log_c"],steplaws[sz][ratio][domain]["log_d0"], steplaws[sz][ratio][domain]["alpha"])) for sz in fit_sizes
            ]
            for ratio in ratios for domain in DOMAINS_2_SUBDOMAINS
        ]).T
        law = ScalingLaw(power_law_share_alpha)
        param = law.fit(x, y, param_generator_share_alpha(np.log(np.min(y, axis=0))), 100, eps=0, valid_split=1, delta=8e-2)
        fit_x = torch.stack([
            torch.linspace(np.min(x), MODEL_SIZES[target_size], 1000)
            for ratio in ratios for domain in DOMAINS_2_SUBDOMAINS
        ], dim=1)
        prediction = power_law_share_alpha(torch.tensor(fit_x), torch.tensor(param))
        
        for ax_row, ratio in zip(axes, ratios):
            for j, domain in enumerate(DOMAINS_2_SUBDOMAINS):
                idx = indices[(ratio, domain)]
                plot_x = np.array([MODEL_SIZES[sz] for sz in fit_sizes]) #  + [MODEL_SIZES[target_size]])
                plot_y = np.array([
                    power_law(token, (steplaws[sz][ratio][domain]["log_c"],steplaws[sz][ratio][domain]["log_d0"], steplaws[sz][ratio][domain]["alpha"])) 
                    for sz in fit_sizes # + [target_size]
                ])
                ax_row[0].scatter(plot_x, plot_y - np.exp(param[idx]), c=PALLETES[j])
                ax_row[0].plot(fit_x[:, idx], prediction[:, idx] - np.exp(param[idx]), c=PALLETES[j])
                ax_row[1].scatter(plot_x, plot_y, c=PALLETES[j])
                ax_row[1].plot(fit_x[:, idx], prediction[:, idx], c=PALLETES[j])
                sizelaws[ratio][domain][step] = {"log_c": param[idx], "log_n0": param[idx+(len(param)-1)//2], "alpha": param[-1]}
            ax_row[0].set_xscale("log"); ax_row[0].set_yscale("log")
            ax_row[1].set_xscale("log"); ax_row[1].set_yscale("log")
    elif tie_alpha == "valid":
        for ax_row, ratio in zip(axes, ratios):
            for domain in DOMAINS_2_SUBDOMAINS:
                for sz in fit_sizes:
                    try:
                        x = steplaws[sz][ratio][domain]
                    except:
                        import ipdb;ipdb.set_trace()
                         
            x = np.array([[MODEL_SIZES[sz] for sz in fit_sizes] for domain in DOMAINS_2_SUBDOMAINS]).T
            y = np.array([
                [
                    power_law(token, (steplaws[sz][ratio][domain]["log_c"],steplaws[sz][ratio][domain]["log_d0"], steplaws[sz][ratio][domain]["alpha"])) for sz in fit_sizes
                ]
                for domain in DOMAINS_2_SUBDOMAINS
            ]).T
            law = ScalingLaw(power_law_share_alpha)
            param = law.fit(x, y, param_generator_share_alpha(np.log(np.min(y, axis=0))), 10, eps=0, valid_split=0, delta=8e-2)
            fit_x = torch.stack([
                torch.linspace(np.min(x), MODEL_SIZES[target_size], 1000)
                for domain in DOMAINS_2_SUBDOMAINS
            ], dim=1)
            prediction = power_law_share_alpha(torch.tensor(fit_x), torch.tensor(param))
            for j, domain in enumerate(DOMAINS_2_SUBDOMAINS):
                idx = j
                plot_x = np.array([MODEL_SIZES[sz] for sz in fit_sizes]) #  + [MODEL_SIZES[target_size]])
                plot_y = np.array([
                    power_law(token, (steplaws[sz][ratio][domain]["log_c"],steplaws[sz][ratio][domain]["log_d0"], steplaws[sz][ratio][domain]["alpha"])) 
                    for sz in fit_sizes # + [target_size]
                ])
                ax_row[0].scatter(plot_x, plot_y - np.exp(param[idx]), c=PALLETES[j])
                ax_row[0].plot(fit_x[:, idx], prediction[:, idx] - np.exp(param[idx]), c=PALLETES[j])
                ax_row[1].scatter(plot_x, plot_y, c=PALLETES[j])
                ax_row[1].plot(fit_x[:, idx], prediction[:, idx], c=PALLETES[j])
                sizelaws[ratio][domain][step] = {"log_c": param[idx], "log_n0": param[idx+(len(param)-1)//2], "alpha": param[-1]}
            ax_row[0].set_xscale("log"); ax_row[0].set_yscale("log")
            ax_row[1].set_xscale("log"); ax_row[1].set_yscale("log") 
    else:
        raise NotImplementedError
    
    fig.savefig(savefig)
    with open(SIZELAW_FILE, "w") as f:
        json.dump(sizelaws, f, indent=4)
        
        
        
if __name__ == '__main__':
    fire.Fire(main)