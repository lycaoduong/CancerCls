# CancerCls

## Model Support
- [x] Resnet
- [x] InceptionNetV4

## For Dataset
Open configs/dataset/cancer.yaml and modify root dir, train, val and test *.csv file before training

## For Train
Run python script train.py with variable parser arguments:<br />
Resnet101
```
python train.py --model resnet101 --epochs 50
```
InceptionNetV4
```
python train.py --model inception_v4 --epochs 50
```
## For Evalation
Run python script eval.py with variable parser arguments:<br />
Resnet101
```
python eval.py --model resnet101 --ckpt best.pt
```
InceptionNetV4
```
python eval.py --model inception_v4 --ckpt best.pt
```
