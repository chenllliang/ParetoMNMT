# On the Pareto Front of Multilingual Neural Machine Translation

In this repo, we provide the source code for you to reproduce the collapse of Pareto front phenomena and the visualization result as in our paper.

## Environment
```
conda create -n ParetoMNMT python=3.8.15
conda activate ParetoMNMT
bash setup.sh
```

## Reproduce the results

We provide the training scripts for reproducing the 2d and 3d trade-off front in our paper.

The training log and checkpoint will be saved at `./logs` and `./checkpoints` directories. 

```
cd scripts
bash frdezh_trade_off.sh # 3d-trade-off front
bash frzh_trade_off.sh # 2d-trade-off front
# you can split the training to different GPU to speed up
```


- you can also use the `scripts/inference.sh` to compute the BLEU score of each models 

```
cd scripts
bash inference.sh <checkpoint_dir> # you can change the inferenced directions in the script
```


## Visualization

We provide a jupyter notebook `./scripts/3d-vis.ipynb` to visulize the 3d Pareto front after training all models.

The results:

<div align=center>
<b>3d trade-off front of fr10M-de4.6M-zh260k</b>
<br>

<img width="300" src="./imgs/3d-trade-off.png"/>

</div>




