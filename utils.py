import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from termcolor import colored
import logging
from tqdm import tqdm
import copy
import numpy as np
import torch.nn.functional as F
from bayesian_torch.models.dnn_to_bnn import get_kl_loss
import os 
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

# Distirbuted Data Parallel
from torch.nn.parallel import DistributedDataParallel as DDP
import os 
import torch.distributed as dist
from torch.utils.data import DistributedSampler

# Load necessary models
from bayesian_torch.models.bayesian.resnet_variational import resnet20 as resnet20_bayesian
from bayesian_torch.models.deterministic.resnet import resnet20 as resnet20_deterministic
from bayesian_torch.models.deterministic.resnet_large import resnet18 as resnet18_deterministic
from bayesian_torch.models.bayesian.resnet_variational_large import resnet18 as resnet18_bayesian
from bayesian_torch.models.bayesian.densenet_variational import densenet_bc_30_uni
from bayesian_torch.models.deterministic.densenet import densenet_bc_30
#from bayesian_torch.models.deterministic.densenet import densenet_bc_121
from bayesian_torch.models.deterministic.mobilenet import MobileNet
from bayesian_torch.models.bayesian.mobilenet_uni import MobileNet_uni
from bayesian_torch.models.deterministic.vit_tiny_dnn import ViT_Tiny_dnn, vit_tiny_dnn
from bayesian_torch.models.bayesian.vit_tiny_uni import ViT_Tiny_uni, vit_tiny_uni

from bayesian_torch.models.deterministic.mlp import MLP
from bayesian_torch.models.bayesian.mlp_variational import MLP_uni
import argparse

def prune_model(model, sparsity, logger):
    """
    모델 전체의 Conv2d 및 Linear 레이어에 대해 global unstructured pruning을 적용합니다.
    Args:
        model (torch.nn.Module): 가지치기할 모델.
        sparsity (float): 가지치기 비율 (0.0 ~ 1.0). 전체 weight 중 프루닝할 비율.
    """
    # 프루닝할 (module, parameter) 쌍을 모읍니다.
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
        # if isinstance(module, (nn.Conv2d)):
            parameters_to_prune.append((module, 'weight'))
    
    # global unstructured pruning을 적용합니다.
    prune.global_unstructured(
        parameters_to_prune,
        # pruning_method=prune.L1Unstructured,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )
    
    # 각 모듈별 가지치기 결과 출력
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            total_params = module.weight.numel()
            # 프루닝 후 각 모듈에는 'weight_mask' 버퍼가 생성됩니다.
            remaining_params = module.weight_mask.sum().item() if hasattr(module, 'weight_mask') else total_params
            pruned_percentage = 1 - (remaining_params / total_params)
            print(f"{name}: {int(remaining_params)}/{int(total_params)} parameters remaining ({pruned_percentage:.2%} pruned)")
        else:
            pass
            
    # Calculate Total Sparsity in the model
    total_params = sum(module.weight.numel() for module in model.modules() if isinstance(module, (nn.Conv2d, nn.Linear)))
    remaining_params = sum(module.weight_mask.sum().item() for module in model.modules() if hasattr(module, 'weight_mask'))
    total_sparsity = 1 - (remaining_params / total_params)
    logger.info(colored(f"Total sparsity: {total_sparsity:.2%}", 'yellow'))
    

def prune_dnn_ffn_only(model, sparsity, logger):
    """
    DNN ViT 모델의 Transformer Block 내 FFN(MLP) 레이어에 대해서만
    global unstructured pruning을 적용합니다.
    
    Args:
        model (torch.nn.Module): 가지치기할 DNN ViT 모델.
        sparsity (float): 가지치기 비율 (0.0 ~ 1.0).
        logger: 로깅을 위한 객체.
    """
    # 1. FFN Linear 레이어만 선택하여 프루닝할 파라미터 목록 생성
    parameters_to_prune = []
    for name, module in model.named_modules():
        # 표준 Linear 레이어이고, 이름에 '.mlp.'가 포함된 경우
        if isinstance(module, nn.Linear) and '.mlp.' in name:
            parameters_to_prune.append((module, 'weight'))

    if not parameters_to_prune:
        logger.warning("No FFN layers to prune were found.")
        return

    # 2. 선택된 FFN 레이어들에 대해 Global Unstructured Pruning 적용
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )

    # 3. 각 모듈별 및 전체 가지치기 결과 로깅
    total_params = 0
    remaining_params = 0
    
    logger.info("--- Pruning Results for DNN ViT ---")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and '.mlp.' in name:
            # 'weight_mask'가 있는지 확인하여 프루닝 적용 여부 판단
            if hasattr(module, 'weight_mask'):
                module_total = module.weight_orig.numel()
                module_remaining = module.weight_mask.sum().item()
                pruned_percentage = 1 - (module_remaining / module_total)
                
                total_params += module_total
                remaining_params += module_remaining
                
                logger.info(f"{name}: {int(module_remaining)}/{int(module_total)} params remaining ({pruned_percentage:.2%} pruned)")

    if total_params > 0:
        total_sparsity = 1 - (remaining_params / total_params)
        logger.info(colored(f"Total FFN Sparsity: {total_sparsity:.2%}", "yellow"))
    else:
        logger.info("No parameters were pruned.")

def check_params(model):
    """
    모델의 파라미터 수를 출력합니다.
    Args:
        model (torch.nn.Module): 파라미터 수를 확인할 모델.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_model(args, logger):
    
    logging.info(colored(f"Creating {args.type} {args.model} model, dataset {args.data}", 'yellow'))

    if args.data in ['cifar10', 'svhn', 'mnist', 'fashionmnist']:
        num_classes = 10
    elif args.data == 'cifar100':
        num_classes = 100
    elif args.data == 'tinyimagenet':
        num_classes = 200
        img_size = 64
    else:
        raise ValueError(f"Unknown dataset: {args.data}")
    if args.type == 'dnn':
        
        if args.model == 'resnet20':
            model = resnet20_deterministic(num_classes=num_classes)
        elif args.model == 'resnet18':
            model = resnet18_deterministic(num_classes=num_classes)
            
        elif args.model == 'densenet30':
            model = densenet_bc_30(num_classes=num_classes)
        elif args.model == 'densenet121':
            model = densenet_bc_121(num_classes=num_classes)
            
        elif args.model == 'mobilenet':
            model = MobileNet(num_classes=num_classes)
            
        elif args.model == 'vit-tiny-layernorm-nano':
            if args.data == 'tinyimagenet':
                model = vit_tiny_dnn(img_size=img_size, num_classes=num_classes, model='nano')
            else:
                model = vit_tiny_dnn(num_classes=num_classes, img_size = 32, model='nano')
        
        elif args.model == 'vit-tiny-layernorm-micro':
            if args.data == 'tinyimagenet':
                model = vit_tiny_dnn(img_size=img_size, num_classes=num_classes, model='micro')
            else:
                model = vit_tiny_dnn(num_classes=num_classes, img_size = 32, model='micro')
                
        elif args.model == 'vit-tiny-layernorm-original':
            if args.data == 'tinyimagenet':
                model = ViT_Tiny_dnn(img_size=img_size, num_classes=num_classes, model='original')
            else:
                model = ViT_Tiny_dnn(num_classes=num_classes, img_size = 32, model='original')                

        elif args.model == 'vit-tiny-layernorm-pico':
            if args.data == 'tinyimagenet':
                model = vit_tiny_dnn(img_size=img_size, num_classes=num_classes, model='pico')
            else:
                model = vit_tiny_dnn(num_classes=num_classes, img_size = 32, model='pico')
                
        elif args.model == 'mlp':
            
            model = MLP(input_dim=28*28, hidden_dims=[200, 100], output_dim=num_classes)
            
        else:
            raise ValueError(f"Unknown model of type: {args.model} of type {args.type}")
        
    elif args.type == 'uni':
        
        if args.model == 'resnet20':
            model = resnet20_bayesian(num_classes = num_classes, prior_type=args.prior_type, args = args)
            
        elif args.model == 'resnet18':
            model = resnet18_bayesian(num_classes=num_classes, prior_type=args.prior_type, args = args)
        elif args.model == 'densenet30':
            model = densenet_bc_30_uni(num_classes=num_classes, prior_type=args.prior_type, args = args)

        elif args.model == 'mobilenet':
            model = MobileNet_uni(num_classes=num_classes, prior_type=args.prior_type)
            
        elif args.model == 'vit-tiny-layernorm-nano':
            if args.data == 'tinyimagenet':
                model = vit_tiny_uni(num_classes=num_classes, model='nano', img_size = img_size, prior_type=args.prior_type, args = args)
            else:
                model = vit_tiny_uni(num_classes=num_classes, model='nano', img_size = 32, prior_type=args.prior_type, args = args)
        elif args.model == 'vit-tiny-layernorm-micro':
            if args.data == 'tinyimagenet':
                model = vit_tiny_uni(num_classes=num_classes, model='micro', img_size = img_size, prior_type=args.prior_type, args = args)
            else:
                model = vit_tiny_uni(num_classes=num_classes, model='micro', img_size = 32, prior_type=args.prior_type, args = args)
                
        elif args.model == 'vit-tiny-layernorm-original':
            if args.data == 'tinyimagenet':
                model = ViT_Tiny_uni(num_classes=num_classes, model='original', img_size = img_size, prior_type=args.prior_type)
            else:
                model = ViT_Tiny_uni(num_classes=num_classes, model='original', img_size = 32, prior_type=args.prior_type, args = args)
                
        elif args.model == 'vit-tiny-layernorm-pico':
            if args.data == 'tinyimagenet':
                model = vit_tiny_uni(num_classes=num_classes, model='pico', img_size = img_size, prior_type=args.prior_type, args = args)
            else:
                model = vit_tiny_uni(num_classes=num_classes, model='pico', img_size = 32, prior_type=args.prior_type, args = args)
                
        elif args.model == 'mlp':
            
            model = MLP_uni(input_dim=28*28, hidden_dims=[200, 100], output_dim=num_classes, prior_type=args.prior_type, args = args)
        else:
            raise ValueError(f"Unknown model of type: {args.model} of type {args.type}")
        
    else:
        raise ValueError(f"Unknown model type: {args.type}")
    
    total_params, trainable_params = check_params(model)
    logging.info(colored(f"Total parameters: {total_params:,}, Trainable parameters: {trainable_params:,}", 'yellow'))
    
    return model

def get_dataset(args, logger):
    
    # assert args.data in ['mnist', 'fashionmnist', 'cifar10', 'cifar100', 'tinyimagenet', 'svhn'], f"Dataset: {args.data} not supported"
    
    if args.data == 'mnist':
        
        logger.info(colored(f"MNIST dataset is loaded", 'green'))

        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST(root='./data/', train=True, transform=transform_train, download=True)
        test_dataset = datasets.MNIST(root='./data/', train=False, transform=transform_test)
        
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
        
    elif args.data == 'fashionmnist':
        logger.info(colored(f"Fashion-MNIST dataset is loaded", 'green'))

        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        
        train_dataset = datasets.FashionMNIST(root='./data/', train=True, transform=transform_train, download=True)
        test_dataset = datasets.FashionMNIST(root='./data/', train=False, transform=transform_test)
        
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
        
    elif args.data == 'cifar10':
        
        logger.info(colored(f"CIFAR-10 dataset is loaded", 'green'))
        img_size = 32
        
        if 'cifar10' in args.ood: # Assume Model is trained for Tiny ImageNet (64x64)
            img_size = 64
        
            
        transform_train = transforms.Compose([
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        train_dataset = datasets.CIFAR10(root='./data/', train=True, transform=transform_train, download=True)
        test_dataset = datasets.CIFAR10(root='./data/', train=False, transform=transform_test)
        
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
    
    elif args.data == 'cifar100':
        
        logger.info(colored(f"CIFAR-100 dataset is loaded", 'green'))
        img_size = 32
        
        if 'cifar100' in args.ood:# Assume Model is trained for Tiny ImageNet (64x64)
            img_size = 224
            
        transform_train = transforms.Compose([
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        train_dataset = datasets.CIFAR100(root='./data/', train=True, transform=transform_train, download=True)
        test_dataset = datasets.CIFAR100(root='./data/', train=False, transform=transform_test)
        
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
        
    elif args.data == 'tinyimagenet':

        img_size = 224
        
        if 'tinyimagenet' in args.ood:
            img_size = 32
            
        logger.info(colored(f"TinyImageNet dataset is loaded", 'green'))
        
        if args.model in ['vit-tiny-layernorm-original']:
            img_size = 64
            from vits_for_small_scale_datasets.utils.autoaug import ImageNetPolicy
            from vits_for_small_scale_datasets.utils.random_erasing import RandomErasing
            from vits_for_small_scale_datasets.utils.sampler import RASampler
            re = 0.25
            re_sh = 0.4
            re_r1 = 0.3
            ra = 3
            augmentations = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, padding=4),
                ImageNetPolicy(),
                transforms.ToTensor(),
                transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
                RandomErasing(probability = re, sh = re_sh, r1 = re_r1, mean = [0.4802, 0.4481, 0.3975])
            ]
            
            transform_train = transforms.Compose(augmentations)
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
            ])
            train_dataset = ImageFolder(root='data/tiny-imagenet-200/train/', transform = transform_train)
            test_dataset = ImageFolder(root='data/tiny-imagenet-200/val/', transform = transform_test)
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_sampler = RASampler(len(train_dataset), args.bs, 1, ra, shuffle=True, drop_last=True)
            )
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
            
        else:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
                

            ])
            
            transform_test = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),

            ])
            
            train_dataset = ImageFolder(root='data/tiny-imagenet-200/train/', transform = transform_train)
            test_dataset = ImageFolder(root='data/tiny-imagenet-200/val/', transform = transform_test)
            
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
            
        
    elif args.data == 'svhn':
        
        logger.info(colored(f"SVHN dataset is loaded", 'green'))
        img_size = 32
        
        if 'svhn' in args.ood: # Assume Model is trained for Tiny ImageNet (64x64)
            img_size = 64
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        
        train_dataset = datasets.SVHN(root='./data/', split='train', transform=transform_train, download=True)
        test_dataset = datasets.SVHN(root='./data/', split='test', transform=transform_test, download=True) 
        
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
                                                  
    elif args.data == 'fashionmnist':
        
        logger.info(colored(f"Fashion-MNIST dataset is loaded", 'green'))

        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        
        train_dataset = datasets.FashionMNIST(root='./data/', train=True, transform=transform_train, download=True)
        test_dataset = datasets.FashionMNIST(root='./data/', train=False, transform=transform_test)
        
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
        
    else:
        raise ValueError('Dataset not found')
    
    logger.info(colored(f"Train Transforms:"))
    logger.info(colored(f"{transform_train}", 'green'))
    
    logger.info(colored(f"Test Transforms:"))
    logger.info(colored(f"{transform_test}", 'green'))
        
    if torch.cuda.device_count() > 1 and args.multi_gpu:
        
        # DDP 초기화
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl', init_method='env://')
        
        # Set device
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        
        # DistributedSampler 설정
        args.train_sampler = DistributedSampler(train_dataset)
        args.test_sampler = DistributedSampler(test_dataset, shuffle=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, sampler=args.train_sampler, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, sampler=args.test_sampler, num_workers=4, pin_memory=True)
        print(colored(f"Data is wrapped by DistributedSampler", 'red'))

    return train_loader, test_loader
    
class EarlyStopping:
    
    """
    Validation Loss가 개선되지 않으면 일정 patience 만큼 기다렸다가 학습을 조기 종료합니다.
    """
    def __init__(self, patience=5, min_delta=0.0):
        """
        Args:
            patience (int): 성능이 개선되지 않는 Epoch가 patience를 초과하면 학습 중단
            min_delta (float): Loss가 이전 최저값 대비 어느 정도(=min_delta) 이하로 내려가야 '개선'으로 판단
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model_state = None  # 최적 모델 가중치 저장

    def __call__(self, val_loss, model):
        """
        val_loss (float): 현재 Epoch에서 측정한 Validation Loss
        model (nn.Module): 학습 중인 모델 객체
        """
        if val_loss < self.best_loss - self.min_delta:
            # 성능이 개선된 경우
            self.best_loss = val_loss
            self.counter = 0
            # 모델의 가중치(파라미터) 복사해 저장
            self.best_model_state = copy.deepcopy(model.state_dict())
        else:
            # 성능 개선 없음
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_BNN(epoch, model, train_loader, test_loader, optimizer, writer, args, mc_runs, bs, device, logger):

    model.to(device)
    best_loss = torch.inf
    best_nll = torch.inf
    best_acc = 0
    
    early_stopping = EarlyStopping(patience=100, min_delta=0.0)
    
    for e in range(epoch):
        if args.train_sampler:
            args.train_sampler.set_epoch(e)            
            
        model.train()
        nll_total = []
        kl_total = []
        correct = 0
        total = 0
        
        pbar = tqdm(enumerate(train_loader))
        N = len(train_loader.dataset)
        
        if args.scale == 'N':
            scaling = N
        else:
            scaling = bs
            
        for batch_idx, (data, target) in pbar:
    
            data, target = data.to(device), target.to(device)
            
            if args.model in ['vit-tiny-layernorm-original']:
                from vits_for_small_scale_datasets.utils.mix import cutmix_data, mixup_data, mixup_criterion
                from vits_for_small_scale_datasets.utils.losses import LabelSmoothingCrossEntropy
                criterion = LabelSmoothingCrossEntropy()
                # print("Using CutMix or MixUp")
                np.int = int
                args.alpha = 1.0
                args.beta = 1.0
                r = np.random.rand(1)
                if r < 0.5:
                    switching_prob = np.random.rand(1)
                    
                    # Cutmix
                    if switching_prob < 0.5:
                        slicing_idx, y_a, y_b, lam, sliced = cutmix_data(data, target, args)
                        data[:, :, slicing_idx[0]:slicing_idx[2], slicing_idx[1]:slicing_idx[3]] = sliced
                        output, kl = model(data)
                        
                        nll =  mixup_criterion(criterion, output, y_a, y_b, lam)
                        
                    # Mixup
                    else:
                        data, y_a, y_b, lam = mixup_data(data, target, args)
                        output, kl = model(data)
                        nll = mixup_criterion(F.cross_entropy, output, y_a, y_b, lam)
                else:
                    output, kl = model(data)
                    nll = criterion(output, target)
                
                _, predicted = torch.max(output.data, 1)
                
                scaled_kl = kl / scaling
                loss = nll * (1/args.t) + scaled_kl#N # args.t: Cold posterior temperature
                # loss = nll
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                outputs =[]
                kls = []
                
                for _ in range(1): # For training, mc_runs is set to 1
                    output, kl = model(data)
                    outputs.append(output)
                    kls.append(kl)
                
                output = torch.mean(torch.stack(outputs), dim=0)
                kl_loss = torch.mean(torch.stack(kls), dim=0).mean()
                
                _, predicted = torch.max(output.data, 1)
                
                nll = F.cross_entropy(output, target)
                scaled_kl = kl_loss / scaling
                loss = nll * (1/args.t) + scaled_kl#N # args.t: Cold posterior temperature
                # loss = nll
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            nll_total.append(nll.detach().cpu())
            kl_total.append(scaled_kl.detach().cpu())
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
            acc = correct / total            
            
            pbar.set_description(colored(f"[Train] Epoch: {e+1}/{epoch}, Acc: {acc:.5f}, NLL: {np.mean(nll_total):.5f} KL: {np.mean(kl_total):,}, KL scaling: {scaling}", 'blue'))
            
        args.scheduler.step()
        
        acc_test, nll, kl = test_BNN(model = model, test_loader = test_loader, bs = bs, mc_runs = 30, device = device, args = args)
        logger.info(f"[Test] Acc: {acc_test:.5f}, NLL: {nll:.5f}, KL: {kl:,}, KL scaling: {scaling}")
        
        # args.scheduler.step()
        # print(colored(f"Learning rate: {optimizer.param_groups[0]['lr']}", 'red'))
        # Tensorboard
        writer.add_scalar('Train/Accuracy', acc, e)
        writer.add_scalar('Train/loss/NLL', np.mean(nll_total), e)
        writer.add_scalar('Train/loss/KL', np.mean(kl_total), e)
        writer.add_scalar('Train/loss/ELBO', -np.mean(nll_total) - np.mean(kl_total), e)
        writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], e)
        writer.add_scalar('Test/accuracy', acc_test, e)
        writer.add_scalar('Test/loss/NLL', nll, e)
        writer.add_scalar('Test/loss/KL', kl, e)
        writer.add_scalar('Test/loss/ELBO', - nll - kl, e)
        
        # Evaluate the best model by the total loss (test)
        if best_loss > nll + kl:
            best_loss = nll + kl
            
            torch.save(model.state_dict(), os.path.join(writer.log_dir, 'best_model.pth'))    
            logger.info(f"Best model saved at epoch {e}")
            
        if best_nll > nll:
            best_nll = nll
            torch.save(model.state_dict(), os.path.join(writer.log_dir, 'best_nll_model.pth'))
            logger.info(f"Best NLL model saved at epoch {e}")
            
            best_model_weight = copy.deepcopy(model.state_dict())   
            args.weight = os.path.join(writer.log_dir, 'best_nll_model.pth')

        if best_acc < acc_test:
            best_acc = acc_test
            torch.save(model.state_dict(), os.path.join(writer.log_dir, 'best_acc_model.pth'))
            logger.info(f"Best ACC model saved at epoch {e}")
            
        early_stopping(val_loss=nll, model=model)
        
        if early_stopping.early_stop:
            logger.info(f"Early stopping at epoch {e+1}")
            return False
        
    torch.save(model.state_dict(), os.path.join(writer.log_dir, 'last_model.pth'))
    logger.info(f"Last model saved")
    
    return best_model_weight

def test_BNN(model, test_loader, bs, device, args, moped=False, mc_runs = 30):
    
    assert mc_runs == 30, "MC runs should be set to 30 for testing BNN"
    
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    nll_total = []
    kl_total = []
    
    N = len(test_loader.dataset)
    if args.scale == 'N':
        scaling = N
    else:
        scaling = bs
        
    with torch.no_grad():
        
        for data, target in tqdm(test_loader, desc=f'Testing [MC_runs={mc_runs}]'):
            data, target = data.to(device), target.to(device)
            
            outputs = []
            kls = []
            for _ in range(mc_runs):
                output, kl = model(data)
                outputs.append(output)
                kls.append(kl)
                kls.append(kl)
                    
            output = torch.mean(torch.stack(outputs), dim=0).to(device)
            kl = torch.mean(torch.stack(kls), dim=0).mean().to(device)

            _, predicted = torch.max(output.data, 1)
            
            nll = F.cross_entropy(output, target) 
            scaled_kl = kl / scaling
            nll_total.append(nll.item())
            kl_total.append(scaled_kl.item())
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
    return correct / total, np.mean(nll_total), np.mean(kl_total)

def train_DNN(epoch, model, train_loader, test_loader, optimizer, device, writer, args, logger):
    
    model.to(device)    
    model.train()
    nlls = []
    correct = 0
    total = 0
    
    best_loss = torch.inf
    best_acc = 0
    best_model_found = False
    
    early_stopping = EarlyStopping(patience=100, min_delta=0.0)
    
    for e in range(epoch):
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=0)
        model.train()
        # for batch_idx, (data, target) in pbar:
        for batch_idx, batch_data in pbar:
            
            data, target = batch_data[0], batch_data[1]
            data, target = data.to(device).squeeze(1), target.to(device)
            
            if args.model in ['vit-tiny-layernorm-original']:
                from vits_for_small_scale_datasets.utils.mix import cutmix_data, mixup_data, mixup_criterion
                from vits_for_small_scale_datasets.utils.losses import LabelSmoothingCrossEntropy
                criterion = LabelSmoothingCrossEntropy()
                # print("Using CutMix or MixUp")
                np.int = int
                args.alpha = 1.0
                args.beta = 1.0
                r = np.random.rand(1)
                if r < 0.5:
                    switching_prob = np.random.rand(1)
                    
                    # Cutmix
                    if switching_prob < 0.5:
                        slicing_idx, y_a, y_b, lam, sliced = cutmix_data(data, target, args)
                        data[:, :, slicing_idx[0]:slicing_idx[2], slicing_idx[1]:slicing_idx[3]] = sliced
                        output = model(data)
                        
                        loss =  mixup_criterion(criterion, output, y_a, y_b, lam)
                        
                        
                    # Mixup
                    else:
                        data, y_a, y_b, lam = mixup_data(data, target, args)
                        output = model(data)
                        loss = mixup_criterion(F.cross_entropy, output, y_a, y_b, lam)
                else:
                    output = model(data)
                    loss = criterion(output, target)
                    
                optimizer.zero_grad()
                loss.backward()
                _, predicted = torch.max(output.data, 1)
                optimizer.step()
                # args.scheduler.step()
                
            else:
                optimizer.zero_grad()
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
            
            nlls.append(loss.item())
            correct += (predicted == target).sum().item()
            total += target.size(0)
            acc_train = correct / total
            pbar.set_description(colored(f"[Train] Epoch: {e+1}/{epoch}, Acc: {acc_train:.3f}, NLL: {np.mean(nlls):.3f}, LR: {optimizer.param_groups[0]['lr']:.10f}", 'blue'))

            # writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], batch_idx + e * len(train_loader))
            
        args.scheduler.step()
        acc_test, nll_test = test_DNN(model, test_loader, device, args)
        logger.info(f"[Test] Acc: {acc_test:.3f}, NLL: {nll_test:.3f}")
        

        if args.prune:
            writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], e + 1 + args.total_epoch)
            writer.add_scalar('Train/accuracy', acc_train, e + 1 + args.total_epoch)
            writer.add_scalar('Train/loss/NLL', np.mean(nlls), e + 1 + args.total_epoch)
            writer.add_scalar('Test/accuracy', acc_test, e + 1 + args.total_epoch)
            writer.add_scalar('Test/loss/NLL', np.mean(nll_test), e + 1 + args.total_epoch)

        else:
            writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], e)
            writer.add_scalar('Train/accuracy', acc_train, e)
            writer.add_scalar('Train/loss/NLL', np.mean(nlls), e)
            writer.add_scalar('Test/accuracy', acc_test, e)
            writer.add_scalar('Test/loss/NLL', np.mean(nll_test), e)
        
        
        
        if best_loss > nll_test:
            best_loss = nll_test
            torch.save(model.state_dict(), os.path.join(writer.log_dir, 'best_model.pth'))
            logger.info(f"Best model saved at epoch {e+1}")

            best_model_weight = copy.deepcopy(model.state_dict())
            args.weight = os.path.join(writer.log_dir, 'best_model.pth')

        if args.prune:

            logger.info(f"Original best NLL: {args.best_nll:.4f}, Current NLL: {nll_test:.4f}")
            logger.info(f"Original best ACC: {args.best_acc:.4f}, Current ACC: {acc_test:.4f}")
            
            if best_acc <= acc_test:
                best_acc = acc_test
            
            if nll_test <= best_loss:
                best_loss = nll_test
            
            # if best_acc >= args.best_acc and acc_test >= best_acc: 
            if best_loss <= args.best_nll and nll_test <= best_loss:
                
                logger.info(f"Early stopping at epoch {e+1}")
                best_model_weight = model.state_dict()
                save_path = os.path.join(writer.log_dir, f'pruned_model_iter_{args.prune_iter}.pth')
                save_pruned_model(model, save_path)
                
                best_model_found = True
                return False
            elif e == epoch - 1 and not best_model_found:
                logger.info(f"Stop to fine-tune at {e+1} epoch since the NLL does not recovered")
                return True
            
        early_stopping(val_loss=nll_test, model=model)
            
        if early_stopping.early_stop:
            logger.info(f"Early stopping at epoch {e+1}")
            best_model_weight = early_stopping.best_model_state
            save_path = os.path.join(writer.log_dir, f'best_model.pth')
            torch.save(best_model_weight, save_path)
            return False

    torch.save(model.state_dict(), os.path.join(writer.log_dir, 'last_model.pth'))
    logger.info(f"Last model saved")
    
    return best_model_weight
    
def test_DNN(model, test_loader, device, args):

    model.cuda()
    model.eval()
    correct = 0
    total = 0
    nlls = []
    with torch.no_grad():
        
        # for data, target in test_loader:
        for batch_data in test_loader:
            
            data, target = batch_data[0], batch_data[1]
            data, target = data.to(device).squeeze(1), target.to(device)
                
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            loss = F.cross_entropy(output, target)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            nlls.append(loss.item())
            
    return correct / total, np.mean(nlls)


def save_pruned_model(model, save_path):
    """
    Save the pruned model with masks removed.
    Args:
        model (torch.nn.Module): The model to save.
        save_path (str): Path to save the model.
    """
    # Remove pruning masks before saving
    for name, module in model.named_modules():
        if hasattr(module, "weight") and hasattr(module, "weight_orig"):
            prune.remove(module, "weight")
    
    # Save the model's state_dict
    torch.save(model.state_dict(), save_path)


def get_conv_layers(model):
    from bayesian_torch.layers.variational_layers.conv_variational import Conv2dReparameterization
    conv_layers = []
    conv_types = (nn.Conv2d, Conv2dReparameterization)
    
    def find_conv_layers(module):
        # Check if the module is an instance of the convolutional layers we're interested in
        if isinstance(module, conv_types):
            conv_layers.append(module)
        # Recursively go through the children of the module
        for child in module.children():
            find_conv_layers(child)

    # Start the recursive search from the given model
    find_conv_layers(model)
    
    return conv_layers

def get_linear_layers(model):
    from bayesian_torch.layers.variational_layers.linear_variational import LinearReparameterization

    linear_layers = []
    linear_types = (nn.Linear, LinearReparameterization)
    def find_linear_layers(module):
        if isinstance(module, linear_types):
            linear_layers.append(module)
        for child in module.children():
            find_linear_layers(child)
    
    find_linear_layers(model)
    
    return linear_layers

if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser(description='Model Initialization Example')
    parser.add_argument('--type', type=str, default='uni', help='Model type (dnn or uni)')
    parser.add_argument('--data', type=str, default='cifar10', help='Dataset name')
    parser.add_argument('--model', type=str, default='resnet20', help='Model name')
    parser.add_argument('--prior_type', type=str, default='normal', help='Prior type (normal or laplace)')
    args = parser.parse_args()
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    model = get_model(args, logger)
    
    
    
    
