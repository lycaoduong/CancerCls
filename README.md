# CancerCls

## Model Support
- [x] Resnet
- [x] inceptionNet

## For Dataset
Open configs/dataset/cancer.yaml and modify root dir, train, val and test *.csv file before training

## For Train
Run python script train.py with variable parser arguments:
```
python train.py --model resnet101 --epochs 50
```
## For Evalation
Run python script eval.py with variable parser arguments:
```
python eval.py --model resnet101 --ckpt best.pt
```
