import os
import fire
import json
from tqdm import tqdm
from collections import defaultdict
import multiprocessing as mp
from utils import *
from get_loss import GetLoss
from law import ScalingLaw

import matplotlib.pylab as plt
import warnings
warnings.filterwarnings('ignore')


def fit_regressor(args):
    dim, activation, x, y, state = args
    # random.seed(seed+seed2)
    # np.random.seed(seed+seed2)
    # batch_ratios = np.random.choice(ratios, size, replace=False)
    # x = np.array([list(map(float, ratio.split('-'))) for ratio in batch_ratios])
    # y = np.array([
    #     sum(losses[ratio][domain][STEP] * valid_weight[domain] for domain in DOMAINS_2_SUBDOMAINS) 
    #     for ratio in batch_ratios
    # ], dtype=np.float32)
    regr = AdaBoostRegressor(MLPEstimator(dim, x.shape[-1], activation), random_state=state, n_estimators=30)
    regr.fit(x, y)
    return None, regr, args

def main(size, step, train_data, ratios, target="full", seed=42):
    get_loss = GetLoss(train_data)
    set_seed(seed)
    with open(ratios, "r") as f:
        ratios = [line.strip() for line in f]
    np.random.shuffle(ratios)
    num_mixture = len(ratios[0].split('-')) 

    mixlaws = defaultdict(lambda:defaultdict(lambda: defaultdict(dict))) # size, step, domain
    if os.path.exists(MIXLAW_FILES[train_data]):
        mixlaws.update(torch.load(MIXLAW_FILES[train_data]))
        
    x, y = [], []
    for ratio in ratios:
        if target == "full":
            y.append(sum(
                valid_weight[domain] * get_loss.get_loss(size, ratio, domain, step)
                for domain in DOMAINS_2_SUBDOMAINS
            ))
        else:
           y.append(get_loss.get_loss(size, ratio, target, step))
        x.append(list(map(float, ratio.split('-'))))
    
    args_list = []
    for i in range(16):
        args_list.append(
            [30, "exp", np.array(x), np.array(y), seed+i],
        )
    min_mae, best_model = 100, None
    with mp.Pool(16) as p:
    # for args in tqdm(args_list):
    #     fit_regressor(args_list[0])
    #     _, regr, _ = fit_regressor(args)
        for _, regr, args in tqdm(p.imap(fit_regressor, args_list)):
            prediction = regr.predict(np.array(x))
            mae = np.mean(np.abs(prediction - y))
            print(mae)
            if mae < min_mae:
                min_mae = mae
                best_model = regr
        if target not in mixlaws[size]:
            mixlaws[size][target] = defaultdict(dict)
        mixlaws[size][target][step] = {"type": "mlp_adaboost", "params": best_model}
    
    torch.save({size: {domain: dict(content) for domain, content in mixlaws[size].items()}for size in mixlaws}, MIXLAW_FILES[train_data])
    

if __name__ == '__main__':
    mp.get_start_method("fork")
    fire.Fire(main)