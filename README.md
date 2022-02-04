# ML-Reproducibility-Challenge-2021
R252  ML Reproducibility Challenge

![image-20220204091008862](https://tva1.sinaimg.cn/large/008i3skNgy1gz1kq9shmhj30xq0dwgn1.jpg)



## Setup environment

Create an environment with conda:
```sh
conda create -n gem python=3.8.8
conda activate gem
```
Install PyTorch with CUDA 10.2:
```sh
conda install pytorch cudatoolkit=10.2 -c pytorch
```
Or install PyTorch without CUDA:
```sh
conda install pytorch cpuonly -c pytorch
```
Install other required packages:
```sh
conda install opencv scikit-learn networkx pandas matplotlib seaborn
pip install tensorboardx
```

## Prepare classification models
See [preparation.md](preparation.md) or extract from zip file:
```sh
unzip ckpt.zip
unzip gnnexp_explanation.zip
```

## Distillation
```sh
python generate_ground_truth.py --dataset=syn1 --top_k=6
python generate_ground_truth.py --dataset=syn4 --top_k=6
python generate_ground_truth_graph_classification.py --dataset=Mutagenicity --output=mutag --graph-mode --top_k=20
python generate_ground_truth_graph_classification.py --dataset=NCI1 --output=nci1_dc --graph-mode --top_k=20 --disconnected
```
or you can directly extract from zip file
```
unzip distillation.zip
```

## Train Gem
```sh
python explainer_gae.py --dataset=syn1 --distillation=syn1_top6 --output=syn1_top6
python explainer_gae.py --dataset=syn4 --distillation=syn4_top6 --output=syn4_top6
python explainer_gae_graph.py --distillation=mutag_top20 --output=mutag_top20 --dataset=Mutagenicity --gpu -b 128 --weighted --gae3 --loss=mse --early_stop --graph_labeling --train_on_positive_label --epochs=300 --lr=0.01
python explainer_gae_graph.py --distillation=nci1_dc_top20 --output=nci1_dc_top20 --dataset=NCI1 --gpu -b 128 --weighted --gae3 --loss=mse --early_stop --graph_labeling --train_on_positive_label --epochs=300 --lr=0.01
```

## Evaluate Gem
```sh
python test_explained_adj.py --dataset=syn1 --distillation=syn1_top6 --exp_out=syn1_top6 --top_k=6
python test_explained_adj.py --dataset=syn4 --distillation=syn4_top6 --exp_out=syn4_top6 --top_k=6
python test_explained_adj_graph.py --graph-mode --dataset=Mutagenicity --exp_out=mutag_top20 --distillation=mutag_top20 --top_k=15 --test_out=mutag_top20_top15
python test_explained_adj_graph.py --graph-mode --dataset=NCI1 --exp_out=nci1_dc_top20 --distillation=nci1_dc_top20 --top_k=15 --test_out=nci1_dc_top20_top15
```

## Visualization
Run `*.ipynb` files in Jupyter Notebook or Jupyter Lab.
