# Data Mixing Laws: Optimizing Data Mixture by Predicting Language Modeling Performance

Code and data for "[Data Mixing Laws: Optimizing Data Mixture by Predicting Language Modeling Performance (ICLR2025)](https://arxiv.org/abs/2403.16952)"

## Data Mixing Laws

We include the codes to reproduce experiments and figures to discover data mixing laws in 
* `mix_2_domains.ipynb`: two training domains, single validation domain
* `mix_3_domains.ipynb`: multiple training domains, single validation domain
* `mix_5_domains.ipynb`: multiple training domains, multiple validation domains

## Prediction Pipeline

Our full prediction pipeline can be reproduced with
```bash
cd pipeline
bash run.sh
```

## Citation
```
@inproceedings{ye2025dml,
  title={Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance},
  author={Ye, Jiasheng and Liu, Peiju and Sun, Tianxiang and Zhan, Jun and Zhou, Yunhua and Qiu, Xipeng},
  booktitle={The Thirteenth International Conference on Learning Representations}
  year={2025}
}
```
