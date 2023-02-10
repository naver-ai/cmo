# The Majority Can Help the Minority: Context-rich Minority Oversampling for Long-tailed Classification (CVPR, 2022)
by **Seulki Park<sup>1</sup>, Youngkyu Hong<sup>2</sup>, Byeongho Heo<sup>2</sup>, Sangdoo Yun<sup>2</sup>, Jin Young Choi<sup>1</sup>**

<sup>1</sup> Seoul National University, <sup>2</sup> NAVER AI Lab

This is the official implementation of [Context-rich Minority Oversampling for Long-tailed Classification](https://arxiv.org/abs/2112.00412) in PyTorch.

[Paper](https://arxiv.org/abs/2112.00412) | [Bibtex](#Citation) | [Video](https://youtu.be/ngvazICgOxo) | [Slides](https://www.slideshare.net/SeulkiPark10/cvpr-22-contextrich-minority-oversampling-for-longtailed-classification)

## Requirements

All codes are written by Python 3.7 with
- PyTorch (>= 1.6)
- torchvision (>= 0.7)
- NumPy

## Training 

We provide several training examples:

### CIFAR-100-LT
- CE-DRW + CMO

```bash
python cifar_train.py --dataset cifar100 --loss_type CE --train_rule DRW --epochs 200 --data_aug CMO

```
- BS + CMO
```bash
python cifar_train.py --dataset cifar100 --loss_type BS --epochs 200 --data_aug CMO

```
- BS + CMO (400 epochs, AutoAug)

```bash
python cifar_train.py --dataset cifar100 --loss_type BS --epochs 400 --data_aug CMO --use_randaug

```

### ImageNet-LT
root: location of Imagenet dataset. (Assume ImageNet data is located at data/ILSVRC/)

At least 4 GPUs are used in the experiments.
- BS + CMO 

```bash
python imagenet_train.py -a resnet50 --root data/ILSVRC/ --dataset Imagenet-LT --loss_type BS \
--data_aug CMO --epochs 100 --num_classes 1000 --workers 12 --print_freq 100

```
- BS + CMO (400 epochs, RandAug)

```bash
python imagenet_train.py -a resnet50 --root data/ILSVRC/ --dataset Imagenet-LT --loss_type BS \
--data_aug CMO --epochs 400 --num_classes 1000 --workers 12 --print_freq 100  --wd 5e-4 --lr 0.02 \
--cos --use_randaug
```


### iNaturalist2018
root: location of iNaturalist2018 dataset. (Assume data is located at data/iNat2018/)

At least 4 GPUs are used in the experiments.
- BS + CMO 

```bash
python inat_train.py -a resnet50 --root data/iNat2018/ --dataset iNat18 --loss_type BS --data_aug CMO \
--epochs 100 --num_classes 8142 --workers 12 --print_freq 100 -b 256 
```
- BS + CMO (400 epochs, RandAug)

```bash
python inat_train.py -a resnet50 --root data/iNat2018/ --dataset iNat18 --loss_type BS --data_aug CMO \
--epochs 400 --num_classes 8142 --workers 12 --print_freq 100 --wd 1e-4 --lr 0.02 --cos --use_randaug
```

## Results and Pretrained models
### Test
```bash
python test.py -a resnet50 --root data/iNat2018/ --dataset iNat18 --loss_type CE --train_rule DRW  \
--resume ckpt.best.pth.tar 
```

### ImageNet-LT

 | Method | Model | Top-1 Acc(%) | link | 
 | :---:  | :---: | :---: | :---: | 
 | BS + CMO  | ResNet-50  | 52.3 | [download](https://drive.google.com/file/d/1RIHcrFwzZccqvOs8GgSX5CUFUkXlVvWp/view?usp=sharing) | 
 | BS + CMO (400 epochs)  | ResNet-50 | 58.0 | [download](https://drive.google.com/file/d/1lcG6JBAxgw4bl6fCyQIhrEgnqhrk6Jvr/view?usp=sharing) | 
 
### iNaturalist2018

 | Method | Model | Top-1 Acc(%) | link | 
 | :---:  | :---: | :---: | :---: | 
 | CE-DRW + CMO  | ResNet-50  | 70.9 | [download](https://drive.google.com/file/d/1D-uNavMMM0E1bTw6noFgPXUBOjinb2uM/view?usp=sharing) | 
 | BS + CMO (400 epochs)  | ResNet-50 | 74.0 | [download](https://drive.google.com/file/d/1D5DgNdvW7mX6Ra82MuEYP-9YfzBmKAMj/view?usp=sharing) | 

## License
This project is distributed under [MIT license](LICENSE), except util/moco_loader.py which is adopted from https://github.com/facebookresearch/moco.

```
Copyright (c) 2022-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## Citation

If you find our paper and repo useful, please cite our paper.

```
@inproceedings{park2021cmo,
  title={The Majority Can Help The Minority: Context-rich Minority Oversampling for Long-tailed Classification},
  author={Park, Seulki and Hong, Youngkyu and Heo, Byeongho and Yun, Sangdoo and Choi, Jin Young},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  year={2022}
}
```
