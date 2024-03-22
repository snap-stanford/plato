# PLATO: High dimensional, tabular deep learning with an auxiliary knowledge graph

## Overview
<p align="center">
<img src="img/plato_github_figure.png" width="1100" align="center">
</p>

**PLATO is a method that enables deep learning on a tabular dataset with orders-of-magnitude more features than samples by using an auxiliary knowledge graph.** **(a)** In PLATO, every input feature in a tabular dataset corresponds to a node in an auxiliary knowledge graph with information about the domain. **(b)** In the first layer of a MLP, every input feature corresponds to a vector of weights. **(c,d)** PLATO is based on the inductive bias that, if two input features correspond to similar nodes in the auxiliary KG, they should have similar weight vectors in the first layer of the MLP. PLATO captures the inductive bias by inferring the weight vector for each input feature from its corresponding node in the auxiliary KG. Ultimately, input features with similar embeddings produce similar weight vectors, regularizing the MLP and capturing the inductive bias.

For more information about PLATO, please refer to our [paper](https://openreview.net/pdf?id=GGylthmehy), [5-minute video](https://neurips.cc/virtual/2022/poster/72216), [slides](https://neurips.cc/virtual/2022/poster/72216), or [poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202023/72216.png?t=1702450694.1190798)!

## Preparing your environment
Please run the commands below:
```bash
conda create -n plato python=3.8
bash install.sh
```

## Downloading data
Download data from this anonymous link `https://drive.google.com/file/d/1kMGagEdSE5nCrJDTRI-_yen9RwLXT5Ea/view?usp=sharing`, and put it under `plato/data`.

## Running PLATO
Make sure to replace `$save_path` with the directory where you would like to save your results.
```
cd plato/baseline
python single_source_pipeline.py --filename $save_path/save.pt \
  --device 0 --epochs 30 --runs 3 --model GGMLP \
  --l1_weight 0 --l2_weight 0 --batch_size 32 --tensorboard_dir $save_path \
  --simple --seed 0 --drug_div 1 --gene_div 1000 \
  --mlp_m_layer_list 32,32,1 --enlarge 1 --lr 0.001 --mp --beta 0.1 --load_dir ../data/kg --cache_dir ../data --dataset BRCA
```

## Contact
Please contact Camilo Ruiz (caruiz@cs.stanford.edu) and Hongyu Ren (hyren@cs.stanford.edu) with any questions.

## Citation
```
@inproceedings{ruiz2023high,
  title={High dimensional, tabular deep learning with an auxiliary knowledge graph},
  author={Ruiz, Camilo and Ren, Hongyu and Huang, Kexin and Leskovec, Jure},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```