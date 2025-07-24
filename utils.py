import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from termcolor import colored
import logging
from bayesian_torch.models.bayesian.resnet_variational import resnet20 as resnet20_bayesian
from bayesian_torch.models.deterministic.resnet import resnet20 as resnet20_deterministic
from bayesian_torch.models.bayesian.densenet_variational import densenet_bc_30_uni
from bayesian_torch.models.deterministic.densenet import densenet_bc_30
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
        else:
            pass
            
    # Calculate Total Sparsity in the model
    total_params = sum(module.weight.numel() for module in model.modules() if isinstance(module, (nn.Conv2d, nn.Linear)))
    remaining_params = sum(module.weight_mask.sum().item() for module in model.modules() if hasattr(module, 'weight_mask'))
    total_sparsity = 1 - (remaining_params / total_params)
    logger.info(colored(f"Total sparsity: {total_sparsity:.2%}", 'yellow'))
    
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
    
    if args.type == 'dnn':
        
        if args.model == 'resnet20':
            model = resnet20_deterministic(num_classes=num_classes)
        elif args.model == 'densenet30':
            model = densenet_bc_30(num_classes=num_classes)
        elif args.model == 'vit-tiny-layernorm-nano':
            pass
        elif args.model == 'mlp':
            pass
        else:
            raise ValueError(f"Unknown model of type: {args.model} of type {args.type}")
        
    elif args.type == 'uni':
        
        if args.model == 'resnet20':
            model = resnet20_bayesian(num_classes = num_classes, prior_type=args.prior_type)
        elif args.model == 'densenet30':
            model = densenet_bc_30_uni(num_classes=num_classes, prior_type=args.prior_type)
        elif args.model == 'vit-tiny-layernorm-nano':
            pass
        elif args.model == 'mlp':
            pass
        else:
            raise ValueError(f"Unknown model of type: {args.model} of type {args.type}")
        
    else:
        raise ValueError(f"Unknown model type: {args.type}")
    
    total_params, trainable_params = check_params(model)
    logging.info(colored(f"Total parameters: {total_params:,}, Trainable parameters: {trainable_params:,}", 'yellow'))
    
    return model

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
    
    
    
    