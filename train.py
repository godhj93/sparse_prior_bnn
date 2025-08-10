import torch
import argparse
import logging
import datetime
from tqdm import tqdm
from utils import get_model, get_dataset, train_DNN, train_BNN, test_DNN, prune_model
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter
import os
from test import evaluate

def main(args):
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    model = get_model(args = args, logger = logger)
    train_loader, test_loader = get_dataset(args = args, logger = logger)
    
    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov = args.nesterov)
    
    prior_params = [param for name, param in model.named_parameters() if 'log_a_q' in name or 'log_b_q' in name]
    base_params = [param for name, param in model.named_parameters() if not ('log_a_q' in name or 'log_b_q' in name)]

    param_groups = [
        {'params': base_params},
        {'params': prior_params, 'lr': args.lr_prior} # 사전 분포 파라미터에만 다른 lr 적용
    ]
    
    if args.optimizer == 'sgd':
        optim = torch.optim.SGD(param_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    elif args.optimizer == 'adam':
        optim = torch.optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    args.scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[args.epochs//3, args.epochs//3*2], gamma=0.1)
    
    log_params = {
        'model': args.model,
        'type': args.type,
        'data': args.data,
        'moped': args.moped,
        'std': args.std,
        'scale': args.scale,
        'prior_type': args.prior_type,
        'date': date,

    }
    
    param_str = "_".join([f"{k}_{v}" for k, v in log_params.items()])
    log_path = f"runs/{log_params['data']}/{log_params['model']}/{log_params['date'][:8]}/{log_params['type']}/{param_str}"
    
    writer = SummaryWriter(log_dir=log_path, comment=param_str)
    
    file_handler = logging.FileHandler(log_path + '/log.txt')
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Save the training arguments
    with open(f"{log_path}/config.txt", "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
            
    if args.type == 'dnn':

        if args.prune:
            model.load_state_dict(torch.load(args.weight))
            args.best_acc, args.best_nll = test_DNN(model, test_loader, device, args)
            logger.info(colored(f"Test accuracy of DNN: {args.best_acc:.4f}, Test NLL: {args.best_nll:.4f}", 'green'))
            
            save_path = os.path.join(writer.log_dir, f'original_model.pth')
            torch.save(model.state_dict(), save_path)
            logger.info(colored(f"Original model is saved at {save_path}", 'green'))
            
            args.total_epoch = 0
            for i in range(1, 100):

                args.prune_iter = i

                # Pruning step
                prune_model(model, sparsity=i/100.0, logger=logger)
                optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov = args.nesterov)
                args.scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[args.epochs//3, args.epochs//3*2], gamma=0.1)
                
                # Training
                if train_DNN(epoch=args.epochs, 
                        model=model, 
                        train_loader=train_loader, 
                        test_loader=test_loader, 
                        optimizer=optim, 
                        writer=writer,
                        device=device,
                        args=args,
                        logger=logger): break
                
            best_model_weight = None
                
        else:
            best_model_weight = train_DNN(epoch=args.epochs, 
                    model=model, 
                    train_loader=train_loader, 
                    test_loader=test_loader, 
                    optimizer=optim, 
                    writer=writer,
                    device=device,
                    args=args,
                    logger=logger)
            
            

    else:
        best_model_weight = train_BNN(epoch=args.epochs, 
                  model=model, 
                  train_loader=train_loader, 
                  test_loader=test_loader, 
                  optimizer=optim, 
                  mc_runs=args.mc_runs, 
                  bs=args.bs, 
                  writer=writer,
                  device=device,
                  args=args,
                  logger=logger)

    # Let's test the model after training
    if best_model_weight is not None:
        evaluate(model, best_model_weight, device, args, logger)
    
    else:
        logger.info(colored("No best model weight found. Skipping evaluation.", 'red'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Bayesian Neural Network')
    parser.add_argument('--epochs', type=int, default=90, help='Number of epochs to train')
    parser.add_argument('--mc_runs', type=int, default=1, help='Number of Monte Carlo runs')
    parser.add_argument('--lr', type=float, default=1e-1, help='Learning rate')
    parser.add_argument('--lr_prior', type=float, default=1e-3, help='Learning rate for prior parameters')
    parser.add_argument('--bs', type=int, default=128, help='Batch size')
    parser.add_argument('--model', type=str, help='Model to train')
    parser.add_argument('--type', type=str, help='Type of model [dnn, uni]')
    parser.add_argument('--t', type=float, default=1.0, help='Cold Posterior temperature')
    parser.add_argument('--data', type=str, help='Dataset to use [cifar10, cifar100, svhn, tinyimagenet]')
    parser.add_argument('--train_sampler', type=bool, default=False, help='Do not use this argument')
    parser.add_argument('--prune', action='store_true', help='Use pruning')
    parser.add_argument('--optimizer', type=str, help='Optimizer to use [sgd, adam]')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--nesterov', action='store_true', help='Use Nesterov')
    parser.add_argument('--std', type = float, default = 1e-3, help='Set a std for a good prior')
    parser.add_argument('--scale', type=str, default='BS', help='KLD scale [N, BS]')
    parser.add_argument('--prior_type', type=str, help='Prior type [normal, laplace]')
    parser.add_argument('--multi-gpu', action='store_true', help='Use multi-GPU')
    parser.add_argument('--weight', type=str, help='DNN weight path for initialization')
    parser.add_argument('--moped', action='store_true', help='DO NOT USE')
    parser.add_argument('--ood', type=str, nargs='*', default=None, help='OOD datasets to evaluate')
    parser.add_argument('--clustering_method', type=str, default='umap', help='Clustering method for visualization [tsne, umap]'   )
    parser.add_argument('--perplexity', type=int, default=30)          # t‑SNE
    parser.add_argument('--n_neighbors', type=int, default=15)         # UMAP
    parser.add_argument('--min_dist', type=float, default=0.1)         # UMAP
    parser.add_argument('--eps', type=float, default=0.02, help='Epsilon for adversarial attack')

    args = parser.parse_args()
    print(colored(args, 'blue'))

    main(args)