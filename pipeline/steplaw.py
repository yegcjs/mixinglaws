import os
import fire
import json
import torch
from collections import defaultdict
import matplotlib.pyplot as plt

from law import ScalingLaw
from utils import *
from get_loss import GetLoss

import numpy as np


import warnings
warnings.filterwarnings('ignore')

### SL related
# XXX: for valid
def param_generator_share_alpha(log_c):
    for log_D0 in np.linspace(15, 23, 11):
        for alpha in np.linspace(0.65, 0.9, 10):
            for _ in range(50):
                yield [_log_c + (1+np.random.rand())  for _log_c in log_c] + [log_D0 + np.random.rand() - 0.5 for _ in log_c] + [alpha]


def check(ratios, steps, losses):
    pass_flag = True
    for domain in DOMAINS_2_SUBDOMAINS:
        for ratio in ratios:
            for step in steps:
                try:
                    t = losses[ratio][domain][step]
                except:
                    pass_flag = False
                    print("NOT FOUND:", ratio, domain, step)
    if not pass_flag:
        raise ReferenceError

def main(savefig, train_data, ratios=None, model_size="70M", fit_step_range=[10000, 30000], tie_alpha="all", seed=42):
    """
    params
        savefig: path to save the prediction fig
        model_size: the model size of the law to fit
        fit_step_range: mininum and maximum step used to fit the law
        tie_alpha: enum of ["all", "valid", "none"] tie alpha of different scaling law curves. 
        seed: random seed
    """
    OBSERVED_LOSSES = GetLoss(train_data).OBSERVED_LOSSES
    STEPLAW_FILE = STEPLAW_FILES[train_data]
    set_seed(seed)
    steplaws = defaultdict(lambda:defaultdict(lambda:defaultdict(dict)))
    if os.path.exists(STEPLAW_FILE):
        with open(STEPLAW_FILE, "r") as f:
            steplaws.update(json.load(f))
    if ratios is not None:
        with open(ratios, "r") as f:
            ratios = [line.strip() for line in f]
    else:
        ratios = list(OBSERVED_LOSSES[model_size])
    
    if "0.125-0.5-0.0-0.0625-0.1875-0.0-0.125" in ratios:   # FIXME: remove this item in data
        ratios.remove("0.125-0.5-0.0-0.0625-0.1875-0.0-0.125") 
    step_start, step_end = fit_step_range
    fit_steps = sorted([ step
        for step in OBSERVED_LOSSES[model_size][ratios[-1]][list(DOMAINS_2_SUBDOMAINS.keys())[0]]
        if step_start <= int(step) <= step_end
    ])
    check(ratios, fit_steps, OBSERVED_LOSSES[model_size])
    
    fig, axes = plt.subplots(len(ratios), 2, figsize=(15, len(ratios)*5))
    
    # start fitting
    if tie_alpha == "all":
        i = -1
        indices = {(ratio, domain): (i:=i+1)
            for ratio in ratios for domain in DOMAINS_2_SUBDOMAINS
        }
        tokens = BSZ * np.array([fit_steps for ratio in ratios for domain in DOMAINS_2_SUBDOMAINS]).T
        loss = np.array([
            [OBSERVED_LOSSES[model_size][ratio][domain][step] for step in fit_steps]
            for ratio in ratios for domain in DOMAINS_2_SUBDOMAINS
        ]).T
        law = ScalingLaw(power_law_share_alpha)
        param = law.fit(
            tokens, loss, param_generator_share_alpha(np.min(np.log(loss), axis=0)),
            300, eps=0, workers=-1, delta=2e-2, valid_split=3
        )
        fit_x = torch.stack([
            torch.linspace(10000 * BSZ, 1000000 * BSZ, 1000) 
            for ratio in ratios for domain in DOMAINS_2_SUBDOMAINS
        ], dim=1)
        prediction = power_law_share_alpha(torch.tensor(fit_x), torch.tensor(param))

        for ax_row, ratio in zip(axes, ratios):
            for j, domain in enumerate(DOMAINS_2_SUBDOMAINS):
                idx = indices[(ratio, domain)]
                plot_x = [step * BSZ for step in OBSERVED_LOSSES[model_size][ratio][domain] if step >= step_start]
                plot_y = [loss for step, loss in OBSERVED_LOSSES[model_size][ratio][domain].items() if step >= step_start]
                ax_row[0].scatter(plot_x, plot_y - np.exp(param[idx]), c=PALLETES[j], label=domain)
                ax_row[0].plot(fit_x[:, idx], prediction[:, idx] - np.exp(param[idx]), c=PALLETES[j])
                ax_row[1].scatter(plot_x, plot_y, c=PALLETES[j])
                ax_row[1].plot(fit_x[:, idx], prediction[:, idx], c=PALLETES[j])
                try:
                    steplaws[model_size][ratio][domain] = {"log_c": param[idx], "log_d0": param[idx+(len(param)-1)//2], "alpha": param[-1]}
                except:
                    if model_size not in steplaws:
                        steplaws[model_size] = defaultdict(lambda:defaultdict(dict))
                    elif ratio not in steplaws[model_size]:
                        steplaws[model_size][ratio] = defaultdict(dict)
                    elif domain not in steplaws[model_size][domain]:
                        steplaws[model_size][ratio][domain] = {}
                    steplaws[model_size][ratio][domain] = {"log_c": param[idx], "log_d0": param[idx+(len(param)-1)//2], "alpha": param[-1]}
 
            ax_row[0].set_xscale("log"); ax_row[0].set_yscale("log");ax_row[0].legend();ax_row[0].set_title(ratio)
            ax_row[1].set_xscale("log"); ax_row[1].set_yscale("log")
        
    elif tie_alpha == "valid":
        for ax_row, ratio in zip(axes, ratios):
            tokens = BSZ * np.array([fit_steps for domain in DOMAINS_2_SUBDOMAINS]).T
            loss = np.array([
                [OBSERVED_LOSSES[model_size][ratio][domain][step] for step in fit_steps]
                for domain in DOMAINS_2_SUBDOMAINS
            ]).T
            law = ScalingLaw(power_law_share_alpha)
            param = law.fit(
                tokens, loss, param_generator_share_alpha(np.min(np.log(loss), axis=0)),
                20, eps=0, workers=-1, delta=5e-2, valid_split=5
            )
            fit_x = torch.stack([
                torch.linspace(10000 * BSZ, 1000000 * BSZ, 1000) 
                for domain in DOMAINS_2_SUBDOMAINS
            ], dim=1)
            prediction = power_law_share_alpha(torch.tensor(fit_x), torch.tensor(param))
            
            for idx, domain in enumerate(DOMAINS_2_SUBDOMAINS):
                plot_x = [step * BSZ for step in OBSERVED_LOSSES[model_size][ratio][domain] if step >= step_start]
                plot_y = [loss for step, loss in OBSERVED_LOSSES[model_size][ratio][domain].items() if step >= step_start]
                ax_row[0].scatter(plot_x, plot_y - np.exp(param[idx]), c=PALLETES[idx], label=domain)
                ax_row[0].plot(fit_x[:, idx], prediction[:, idx] - np.exp(param[idx]), c=PALLETES[idx])
                ax_row[1].scatter(plot_x, plot_y, c=PALLETES[idx])
                ax_row[1].plot(fit_x[:, idx], prediction[:, idx], c=PALLETES[idx])
                try:
                    steplaws[model_size][ratio][domain] = {"log_c": param[idx], "log_d0": param[idx+(len(param)-1)//2], "alpha": param[-1]}
                except:
                    if model_size not in steplaws:
                        steplaws[model_size] = defaultdict(lambda:defaultdict(dict))
                    elif ratio not in steplaws[model_size]:
                        steplaws[model_size][ratio] = defaultdict(dict)
                    elif domain not in steplaws[model_size][domain]:
                        steplaws[model_size][ratio][domain] = {}
                    steplaws[model_size][ratio][domain] = {"log_c": param[idx], "log_d0": param[idx+(len(param)-1)//2], "alpha": param[-1]}
 
            ax_row[0].set_xscale("log"); ax_row[0].set_yscale("log");ax_row[0].legend();ax_row[0].set_title(ratio)
            ax_row[1].set_xscale("log"); ax_row[1].set_yscale("log")
    elif tie_alpha == "none":
        pass
    else:
        raise NotImplementedError
    
    fig.savefig(savefig)
    with open(STEPLAW_FILE, "w") as f:
        json.dump(steplaws, f, indent=4)

if __name__ == '__main__':
    fire.Fire(main)