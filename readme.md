# Empirical Priors for Bayesian Neural Networks via Weight Pruning

## Pytorch Implementation

This repository contains the PyTorch implementation of the paper "Empirical Priors for Bayesian Neural Networks via Weight Pruning". The code contrains implementations for main results in the paper, including classification (ResNet, DenseNet, MobileNet, and ViT) for CIFAR-10, CIFAR-100, and TinyImageNet. Ablation studies on different tasks such as regression, depth estimation, and object detection will be released soon.

## Installation

```bash
conda create -n spin python=3.10 -y
cd bayesian-torch && pip install .
cd ..
pip install -r requirements.txt
```

## Usage

-  **Pretrain a model**: Train a deterministic neural network on dataset of interest. 

```
python train.py --type dnn --model [resnet20, densenet30, vit-tiny-layernorm-micro, resnet18, mobilenet] --dataset [cifar10, cifar100, tinyimagenet] --ood [cifar100, tinyimagenet] --optimizer sgd --bs 128 --lr 0.1 --epochs 90 --weight_decay 1e-4 --momentum 0.9
```

- **Prune the model**: Prune the pretrained model using magnitude-based pruning.

```
python train.py --type dnn --model [resnet20, densenet30, vit-tiny-layernorm-micro, resnet18, mobilenet] --dataset [cifar10, cifar100, tinyimagenet] --ood [cifar100, tinyimagenet] --optimizer sgd --bs 128 --lr 0.001 --epochs 90 --weight_decay 1e-4 --momentum 0.9 --prune --weight "weight_path"
```

- **Train a Bayesian Neural Network**: Train a Bayesian neural network with conventional priors [e.g., isotropic Gaussian, Laplace, Student-t, Spike-and-Slab]

```
python train.py --type uni --model [resnet20, densenet30, vit-tiny-layernorm-micro, resnet18, mobilenet] --dataset [cifar10, cifar100, tinyimagenet] --ood [cifar100, tinyimagenet] --optimizer sgd --bs 128 --lr 0.1 --epochs 90 --weight_decay 0.0 --momentum 0.9 --prior [normal, laplace, student-t, spike-and-slab] 
```

- **Train a Bayesian Neural Network with SPIN**: Train a Bayesian neural network with empirical priors derived from the pruned model.

```
python train_with_good_prior.py --type uni --model [resnet20, densenet30, vit-tiny-layernorm-micro, resnet18, mobilenet] --dataset [cifar10, cifar100, tinyimagenet] --ood [cifar100, tinyimagenet] --optimizer sgd --bs 128 --lr 0.1 --epochs 90 --weight_decay 0.0 --momentum 0.9 --prior normal --weight "path of sparse model --std 0.001"
# Here, std is the hyperparameter that controls the variance of the empirical prior.
```

- **Evaluate a trained model**: Evaluate a trained model on in-distribution and out-of-distribution datasets.

```
python test.py --type [dnn, uni] --model [resnet20, densenet30, vit-tiny-layernorm-micro, resnet18, mobilenet] --dataset [cifar10, cifar100, tinyimagenet] --ood [cifar100, tinyimagenet] --weight "path of trained model" --prior [normal, laplace, student-t, spike-and-slab] 
```


