import os
import fire
import json
import torch
from tqdm import tqdm
from collections import defaultdict
from utils import *
# from get_loss import steplaws, sizelaws, mixlaws
import matplotlib.pyplot as plt

GRID_SZ = 1 / 16 
max_size = [1, 1, 0.5857, 0.2588, 0.2254, 0.1779, 0.1727]

def predict_losses(size, step, ratios):
    ckpt = torch.load("../data/RedPajama/mixlaws.pt")
    law = ckpt[size]["full"][step]["params"]
    prediction = law.predict(np.array(ratios))
    return prediction.tolist()
    # total_losses, domain_losses = 0, {domain: [] for domain in DOMAINS_2_SUBDOMAINS}
    # for domain in DOMAINS_2_SUBDOMAINS:
    #     loss = batch_get_loss(size, ratios, domain, step)
    #     total_losses += loss * valid_weight[domain]
    #     domain_losses[domain] = loss.tolist()
    # total_losses = total_losses.tolist()
    # return total_losses, domain_losses


def dfs(depth, rs):
    results = []
    if depth == len(max_size) - 1:
        r = 1 - sum(rs)
        if 0 <= r <= max_size[depth]:
            results.append(rs + [r])
    else:
        for s in range(0, 1+int(max_size[depth] / GRID_SZ)):
            r = GRID_SZ * int(max_size[depth] / GRID_SZ) * (2**(-s))
            if r < GRID_SZ:
                r = 0.0
            results += dfs(depth+1, rs+[r])
    return results

def find_optimal_ratio(size, step, savefig, write_losses=None):
    GRID = 256
    optimal_ratio, min_loss, min_loss_domain = [], 1000, None
    t = int(GRID * (1 - 0.103806741))
    # ratios = [
    #     (r1/GRID, r2/GRID, (GRID - r1 - r2 - r4 - r5)/GRID, r4/GRID, r5/GRID)
    #     for r1 in tqdm(range(int(0.56 * GRID))) 
    #     for r2 in range(int(min(1-r1/GRID, 0.247208695) * GRID))
    #     for r4 in range(int(min(1-r1/GRID-r2/GRID, 0.272774883) * GRID))
    #     for r5 in range(int(min(1-r1/GRID-r2/GRID-r4/GRID, 0.769600522) * GRID))
    #     if (r1 + r2 + r4 + r5 <= GRID) and (r1 + r2 + r4 + r5 >= t)
    # ]
    ratios = dfs(0, [])
    ratios = set(["-".join(map(str, ratio)) for ratio in ratios])
    ratios = [list(map(float, ratio.split('-'))) for ratio in ratios]
    ratio_chunks = [ratios[i:i+8192] for i in range(0, len(ratios), 8192)]
    all_losses = []
    for ratio_ch in tqdm(ratio_chunks):
        loss_ch = predict_losses(size, step, ratio_ch)
        min_loss_idx = np.argmin(loss_ch)
        if loss_ch[min_loss_idx] < min_loss:
            min_loss = loss_ch[min_loss_idx]  
            optimal_ratio = ratio_ch[min_loss_idx] 
            # min_loss_domain = {domain: domain_losses[domain][min_loss_idx] for domain in domain_losses}
        all_losses += loss_ch
    if write_losses is not None:
        with open(write_losses, "w") as f:
            for ratio, loss in zip(ratios, all_losses):
                f.write(f"{ratio}\t{loss}\n")
    plt.hist(all_losses, bins=int((max(all_losses)-min(all_losses))/0.01))
    plt.legend()
    plt.savefig(savefig)
    print("optimal ratio:", optimal_ratio, "\nmin loss:", min_loss, "\n", min_loss_domain)

if __name__ == '__main__':
    fire.Fire(find_optimal_ratio)