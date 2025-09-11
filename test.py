from termcolor import colored
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import argparse
import logging
from utils import get_model, get_dataset, test_BNN, test_DNN
import json
import os
import ast

def compute_tr_h_bnn(model, loader, criterion, device, mc_runs=30, num_hvp_iter=10):
    """
    Hutchinson's method를 사용하여 BNN의 Tr(H_phi)를 계산합니다.
    H_phi는 손실 함수의 변분 파라미터(phi)에 대한 헤시안입니다.
    """
    model.eval()

    # 변분 파라미터(mu, rho)를 대상으로 함
    variational_params = [p for p in model.parameters() if p.requires_grad]
    
    if not variational_params:
        raise ValueError("모델에서 학습 가능한 변분 파라미터를 찾을 수 없습니다.")

    num_params = sum(p.numel() for p in variational_params)
    trace_estimates = []

    print(colored(f"Calculating Tr(H_phi) with Hutchinson's method ({num_hvp_iter} iterations)...", 'yellow'))
    
    for i in tqdm(range(num_hvp_iter), desc="Tr(H_phi) Iterations"):
        # 1. 확률 벡터 v 생성 (Rademacher 분포 사용)
        v = torch.randint_like(torch.empty(num_params), low=0, high=2, device=device) * 2 - 1
        v.requires_grad = False

        hvp_dot_v_total = 0.0
        n_batches = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            # BNN의 확률적 특성을 고려하기 위한 MC 샘플링
            batch_hvp = torch.zeros_like(v)
            for _ in range(mc_runs):
                model.zero_grad()
                
                # 2. 첫 번째 역전파: 그래디언트 g 계산
                output, _ = model(images)
                loss = criterion(output, labels)
                
                # create_graph=True는 두 번째 역전파를 위해 필요
                grads = torch.autograd.grad(loss, variational_params, create_graph=True)
                g = torch.cat([p.view(-1) for p in grads])

                # 3. 두 번째 역전파: 헤시안-벡터 곱 (Hv) 계산
                grad_v_dot = torch.dot(g, v)
                model.zero_grad()
                hvp = torch.autograd.grad(grad_v_dot, variational_params, retain_graph=False)
                hvp = torch.cat([p.contiguous().view(-1) for p in hvp])
                
                batch_hvp += hvp
            
            # MC 샘플에 대한 평균 Hv 계산
            avg_batch_hvp = batch_hvp / mc_runs
            
            # v^T * H * v 계산
            hvp_dot_v_total += torch.dot(avg_batch_hvp, v)
            n_batches += 1
            
        # 데이터셋 전체에 대한 평균 v^T * H * v
        avg_hvp_dot_v = hvp_dot_v_total / n_batches
        trace_estimates.append(avg_hvp_dot_v)

    # 여러 확률 벡터 v에 대한 평균으로 최종 Tr(H) 추정
    tr_h_phi = torch.mean(torch.stack(trace_estimates))
    
    return tr_h_phi.item()

def compute_tr_c_bnn(model, loader, device, mc_runs):
    """
    BNN의 예측 분포를 고려하여 Tr(C_theta)를 계산합니다. (메모리 효율적인 버전)
    """
    model.eval()

    # variational_params = [p for name, p in model.named_parameters() if 'rho' in name or 'mu' in name and p.requires_grad]
    variational_params = [p for name, p in model.named_parameters() if 'mu' in name and p.requires_grad]
    
    if not variational_params:
        raise ValueError("No variational parameters (mu, rho) found in the model.")

    def get_grad_vec():
        grads = []
        for p in variational_params:
            if p.grad is not None:
                grads.append(p.grad.view(-1))
        return torch.cat(grads)

    sum_grad_theta = None
    sum_of_squared_norms = 0.0
    n_batches = len(loader)

    print(colored("Calculating Tr(C_theta) with memory-efficient gradient accumulation...", 'yellow'))
    
    for images, labels in tqdm(loader, desc="Tr(C_theta)"):
        images, labels = images.to(device), labels.to(device)
        
        # --- 그래디언트 누적 방식 변경 ---
        model.zero_grad() # 배치마다 그래디언트 초기화
        for i in range(mc_runs):
            output, _ = model(images)
            # 각 MC 샘플의 손실을 mc_runs로 나눈 뒤 backward() 호출
            loss = F.cross_entropy(output, labels) / mc_runs
            loss.backward() # 매번 작은 그래프에 대해 역전파 수행
            
        # mc_runs번의 backward() 후, .grad 속성에는 그래디언트의 합이 누적되어 있음
        batch_grad_theta = get_grad_vec()
        
        # 1. 그래디언트 벡터 누적
        if sum_grad_theta is None:
            sum_grad_theta = torch.zeros_like(batch_grad_theta)
        sum_grad_theta += batch_grad_theta
        
        # 2. 그래디언트 제곱놈 누적
        sum_of_squared_norms += torch.sum(batch_grad_theta**2)

    # 루프 종료 후 평균 계산
    mean_grad_theta = sum_grad_theta / n_batches
    mean_of_squared_norms = sum_of_squared_norms / n_batches
    
    tr_c_theta = mean_of_squared_norms - torch.sum(mean_grad_theta**2)
    
    return tr_c_theta.item()

def test_ood_detection_dnn(model, in_loader, out_loader, n_bins=15, args=None):
    
    model.eval().cuda()
    
    results = {
        'msp': {'scores': [], 'labels': []},
        'entropy': {'scores': [], 'labels': []}
    }
    
    all_confidences = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        # ──────────────────────────────────────────────
        # In-Distribution 처리
        # ──────────────────────────────────────────────
        for images, y in tqdm(in_loader, desc=f'In-distribution: {args.in_data}'):
            images = images.cuda()
            output = model(images)  # 단일 forward pass
            probs = F.softmax(output, dim=1)  # softmax 확률 계산
            
            # 최대 softmax 확률 (MSP)과 예측값 계산
            confidences, predictions = torch.max(probs, dim=1)
            all_confidences.extend(confidences.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
            # MSP 점수 저장 (ID는 label 0)
            results['msp']['scores'].extend(confidences.cpu().numpy())
            results['msp']['labels'].extend([0] * images.size(0))
            
            # 예측 엔트로피 계산: -sum(p * log(p))
            predictive_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            results['entropy']['scores'].extend(predictive_entropy.cpu().numpy())
            results['entropy']['labels'].extend([0] * images.size(0))
            
        # ──────────────────────────────────────────────
        # Out-of-Distribution 처리
        # ──────────────────────────────────────────────
        for images, _ in tqdm(out_loader, desc=f'Out-of-distribution: {args.data}'):
            images = images.cuda()
            output = model(images)
            probs = F.softmax(output, dim=1)
            
            confidences, _ = torch.max(probs, dim=1)
            
            # MSP 점수 저장 (OOD는 label 1)
            results['msp']['scores'].extend(confidences.cpu().numpy())
            results['msp']['labels'].extend([1] * images.size(0))
            
            # 예측 엔트로피 저장 (OOD는 label 1)
            predictive_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            results['entropy']['scores'].extend(predictive_entropy.cpu().numpy())
            results['entropy']['labels'].extend([1] * images.size(0))
    
    # ──────────────────────────────────────────────────────
    # In-Distribution 데이터에 대한 ECE 및 calibration plot 계산
    # ──────────────────────────────────────────────────────
    ece, avg_confidence_per_bin, accuracy_per_bin = compute_ece_and_plot_confidence_vs_accuracy_batches(
        torch.tensor(all_confidences),
        torch.tensor(all_preds),
        torch.tensor(all_labels),
        n_bins=n_bins,
    )
    print(f"ECE: {ece:.4f}")
    
    # ──────────────────────────────────────────────────────
    # OOD 검출을 위한 AUROC 계산 (MSP와 Entropy)
    # ──────────────────────────────────────────────────────

    # MSP Calculation
    print("\nCalculating AUROC for OOD detection...")
    scores = np.array(results['msp']['scores'])
    labels = np.array(results['msp']['labels'])
    auroc_msp = roc_auc_score(labels, -scores)  # MSP는 ID일수록 큰 값이므로 음수 반전
    print(f"AUROC (MSP): {auroc_msp:.4f}")

    # Entropy Calculation
    scores = np.array(results['entropy']['scores'])
    labels = np.array(results['entropy']['labels'])
    auroc_entropy = roc_auc_score(labels, scores)
    print(f"AUROC (Entropy): {auroc_entropy:.4f}")


    # for method in ['msp', 'entropy']:
    #     scores = np.array(results[method]['scores'])
    #     labels = np.array(results[method]['labels'])
        
    #     # MSP의 경우, in-distribution일수록 큰 값이므로 OOD 검출을 위해 음수 반전
    #     if method == 'msp':
    #         scores = -scores
        
    #     fpr, tpr, thresholds = roc_curve(labels, scores)
    #     auroc = roc_auc_score(labels, scores)
    #     print(f"AUROC ({method.upper()}): {auroc:.4f}")
    
    return {'ece': ece, 
            'auroc_msp': auroc_msp, 
            'auroc_entropy': auroc_entropy,
    }

def test_ood_detection_bnn(model, in_loader, out_loader, mc_runs=30, n_bins=15, args=None):
    
    assert mc_runs == 30, "mc_runs must be 30 for OOD evaluation"
    
    model.eval().cuda()
    
    results = {
        'msp': {'scores': [], 'labels': []},
        'entropy': {'scores': [], 'labels': []},
        'mi': {'scores': [], 'labels': []}
    }
    
    all_confidences = []
    all_preds = []
    all_labels = []
    
    all_mean_probs_in = []
    all_labels_in = []
    
    with torch.no_grad():
        # ──────────────────────────────────────────────
        # In-Distribution 처리
        # ──────────────────────────────────────────────
        for images, y in tqdm(in_loader, desc=f'In-distribution: {args.in_data}'):
            images = images.cuda()
            mc_outputs = []
            for _ in range(mc_runs):
                output, _ = model(images)
                mc_outputs.append(F.softmax(output, dim=1))
            
            mc_outputs = torch.stack(mc_outputs, dim=0)  # [MC, Batch, Classes]
            mean_probs = torch.mean(mc_outputs, dim=0)   # [Batch, Classes]
            
            all_mean_probs_in.append(mean_probs.cpu().numpy())
            all_labels_in.append(y.cpu().numpy())
            
            confidences, predictions = torch.max(mean_probs, dim=1)
            
            all_confidences.extend(confidences.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        
            msp_scores, _ = torch.max(mean_probs, dim=1)

            results['msp']['scores'].extend(msp_scores.cpu().numpy())
            results['msp']['labels'].extend([0] * images.size(0))  
            
            mi_scores, predictive_entropy = compute_mutual_information(mc_outputs)
            results['mi']['scores'].extend(mi_scores.cpu().numpy())
            results['mi']['labels'].extend([0] * images.size(0))
            
            results['entropy']['scores'].extend(predictive_entropy.cpu().numpy())
            results['entropy']['labels'].extend([0] * images.size(0))
        
        # ──────────────────────────────────────────────
        # Out-of-Distribution 처리
        # ──────────────────────────────────────────────
        for images, _ in tqdm(out_loader, desc=f'Out-of-distribution: {args.data}'):
            images = images.cuda()
            mc_outputs = []
            for _ in range(mc_runs):
                output, _ = model(images)
                mc_outputs.append(F.softmax(output, dim=1))
            
            mc_outputs = torch.stack(mc_outputs, dim=0)
            mean_probs = torch.mean(mc_outputs, dim=0)
            
            msp_scores, _ = torch.max(mean_probs, dim=1)
            results['msp']['scores'].extend(msp_scores.cpu().numpy())
            results['msp']['labels'].extend([1] * images.size(0))  # out-dist -> 라벨 1
            
            mi_scores, predictive_entropy = compute_mutual_information(mc_outputs)
            results['mi']['scores'].extend(mi_scores.cpu().numpy())
            results['mi']['labels'].extend([1] * images.size(0))
            
            results['entropy']['scores'].extend(predictive_entropy.cpu().numpy())
            results['entropy']['labels'].extend([1] * images.size(0))
    
    # ──────────────────────────────────────────────────────
    # In-Distribution 데이터 ECE & Calibration plot
    # ──────────────────────────────────────────────────────
    ece, avg_confidence_per_bin, accuracy_per_bin = compute_ece_and_plot_confidence_vs_accuracy_batches(
                torch.tensor(all_confidences),
                torch.tensor(all_preds),
                torch.tensor(all_labels),
            )
    print(f"ECE: {ece:.4f}")
    
    summary_results = {}
    summary_results['ece'] = ece

    # ──────────────────────────────────────────────────────
    # OOD 검출용 AUROC (MSP/Entropy/MI) + ECE subplot
    # ──────────────────────────────────────────────────────
    
    # MSP Calculation
    print("\nCalculating AUROC for OOD detection...")
    scores = np.array(results['msp']['scores'])
    labels = np.array(results['msp']['labels'])
    auroc_msp = roc_auc_score(labels, -scores)  # MSP는 ID일수록 큰 값이므로 음수 반전
    print(f"AUROC (MSP): {auroc_msp:.4f}")

    # Entropy Calculation
    scores = np.array(results['entropy']['scores'])
    labels = np.array(results['entropy']['labels'])
    auroc_entropy = roc_auc_score(labels, scores)
    print(f"AUROC (Entropy): {auroc_entropy:.4f}")

    # Mutual Information Calculation
    scores = np.array(results['mi']['scores'])
    labels = np.array(results['mi']['labels'])
    auroc_mi = roc_auc_score(labels, scores)
    print(f"AUROC (Mutual Information): {auroc_mi:.4f}")


    # for i, method in enumerate(['msp', 'entropy', 'mi']):
    #     scores = np.array(results[method]['scores'])
    #     labels = np.array(results[method]['labels'])
        
    #     # MSP는 ID일수록 값이 크므로, OOD 점수로 쓰기 위해 음수 반전
    #     if method == 'msp':
    #         scores = -scores
        
    #     fpr, tpr, thresholds = roc_curve(labels, scores)
    #     auroc = roc_auc_score(labels, scores)
    #     summary_results[f'auroc_{method}'] = auroc
    #     print(f"AUROC ({method.upper()}): {auroc:.4f}")
    
    return {'ece': ece, 
            'auroc_msp': auroc_msp, 
            'auroc_entropy': auroc_entropy,
            'auroc_mi': auroc_mi,}
    

def compute_ece_and_plot_confidence_vs_accuracy_batches(confidences_batches, preds_batches, labels_batches, n_bins=15):
    
    # tensor라면 numpy 배열로 변환
    if torch.is_tensor(confidences_batches):
        all_confidences = confidences_batches.detach().cpu().numpy()
    else:
        all_confidences = np.array(confidences_batches)
    
    if torch.is_tensor(preds_batches):
        all_preds = preds_batches.detach().cpu().numpy()
    else:
        all_preds = np.array(preds_batches)
    
    if torch.is_tensor(labels_batches):
        all_labels = labels_batches.detach().cpu().numpy()
    else:
        all_labels = np.array(labels_batches)
    
    # Confidence bins 설정
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    N = len(all_confidences)  # 전체 샘플 수

    accuracy_per_bin = []
    avg_confidence_per_bin = []
    bin_fractions = []  # 각 bin별 (|Bin|/N)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        bin_mask = (all_confidences > bin_lower) & (all_confidences <= bin_upper)
        bin_size = np.sum(bin_mask)
        
        if (bin_size > 0):
            acc = np.mean(all_preds[bin_mask] == all_labels[bin_mask])
            conf = np.mean(all_confidences[bin_mask])
            
            # ECE 누적 계산
            fraction = bin_size / N
            ece += np.abs(conf - acc) * fraction
            
            accuracy_per_bin.append(acc)
            avg_confidence_per_bin.append(conf)
            bin_fractions.append(fraction)
        else:
            # bin에 샘플이 없는 경우를 어떻게 처리할지는 옵션
            accuracy_per_bin.append(0.0)
            avg_confidence_per_bin.append((bin_lower + bin_upper)/2.0)
            bin_fractions.append(0.0)

    # 이후 ece, accuracy_per_bin, avg_confidence_per_bin 등을 반환
    return ece, avg_confidence_per_bin, accuracy_per_bin

def compute_mutual_information(mc_probabilities):
    """
    상호 정보량과 예측 엔트로피를 계산합니다.

    Parameters:
        mc_probabilities (torch.Tensor): [MC Samples, Batch Size, Num Classes] 형태의 예측 확률
    Returns:
        mutual_information (torch.Tensor): [Batch Size] 형태의 상호 정보량 값
        predictive_entropy (torch.Tensor): [Batch Size] 형태의 예측 엔트로피 값
    """
    # 평균 예측 확률 계산 (Mean of MC probabilities)
    mean_probabilities = torch.mean(mc_probabilities, dim=0)  # [Batch Size, Num Classes]
    
    # 예측 엔트로피 계산 (H[y | x])
    predictive_entropy = -torch.sum(mean_probabilities * torch.log(mean_probabilities + 1e-8), dim=1)  # [Batch Size]

    # 샘플별 엔트로피 계산 및 평균 (E[H[y | x, θ]])
    sample_entropies = -torch.sum(mc_probabilities * torch.log(mc_probabilities + 1e-8), dim=2)  # [MC Samples, Batch Size]
    expected_entropy = torch.mean(sample_entropies, dim=0)  # [Batch Size]

    # 상호 정보량 계산 (I[y, θ | x])
    mutual_information = predictive_entropy - expected_entropy  # [Batch Size]

    return mutual_information, predictive_entropy

def evaluate(model, best_model_weight, device, args, logger):

    assert args.mc_runs == 30, "mc_runs must be 30 for evaluation"
    
    model.load_state_dict(best_model_weight)
    model.eval()
    logger.info(colored(f"Pretrained model weights loaded", 'blue'))

    import json
    from datetime import datetime

    experiment_results = {
        'info': {},
        'id_performance': {},
        'ood_performance': {},
        'clustering_performance': {},
    }

    # Add test date
    experiment_results['info']['test_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for key, value in vars(args).items():
        # Skip MultiStepLR and optimizer
        if key in ['scheduler', 'optimizer']:
            continue
        experiment_results['info'][key] = value

    logger.info(colored(f"In dataset evaluation starting for {args.data}...", 'blue'))
    # ──────────────────────────────────────────────
    # ID Evaluation
    # ──────────────────────────────────────────────
    _, test_loader = get_dataset(args=args, logger=logger)
    args.in_data = args.data    
    if args.type == 'dnn':
        acc, nll = test_DNN(model, test_loader, device, args)
        print(colored(f"Acc: {acc:.4f}, NLL: {nll:.4f}", 'blue'))
        experiment_results['id_performance'] = {'accuracy': acc, 'nll': nll}
    
    elif args.type == 'uni':
        acc, nll, kld = test_BNN(model=model, test_loader=test_loader, bs=args.bs, device=device, mc_runs=args.mc_runs, args=args)
        print(f"Dataset: {args.data}")
        print(colored(f"Acc: {acc:.4f}, NLL: {nll:.4f}, KLD: {kld:.4f}", 'blue'))
        experiment_results['id_performance'] = {'accuracy': acc, 'nll': nll, 'kld': kld}

    else:
        raise NotImplementedError("Not implemented yet")
    
    if args.data in ['cifar10' , 'svhn' , 'mnist' , 'fashionmnist']:
        logger.info(colored("Clustering performance evaluation starting...", 'blue'))
        # TSNE
        # 데이터 로드
        from torchvision import datasets, transforms
        from typing import List, Sequence
        from tsne import extract_features, run_embedding, cluster_metrics, global_rank_corr
        from torch.utils.data import DataLoader, Subset
        
        args.samples_per_class = 30

        IMG_STATS = {
            "TinyImageNet": ((0.4802, 0.4481, 0.3975),
                            (0.2302, 0.2265, 0.2262)),
            "cifar10": ((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)),

            "svhn": ((0.4377, 0.4438, 0.4728),
                     (0.1980, 0.2010, 0.1970)),
            
            "mnist": ((0.1307,), (0.3081,)),
            "fashionmnist": ((0.2860,), (0.3530,)),
        }


        CLASSIFICATION_DATASETS = {
            "cifar10": lambda root: datasets.CIFAR10(
                root=root,
                train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(*IMG_STATS["cifar10"]),
                ]),
                download=True,
            ),
            "svhn": lambda root: datasets.SVHN(
                root=root,
                split='test',
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(*IMG_STATS["svhn"]),
                ]),
                download=True,
            ),
            "mnist": lambda root: datasets.MNIST(
                root=root,
                train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(*IMG_STATS["mnist"]),
                ]),
                download=True,
            ),
            "fashionmnist": lambda root: datasets.FashionMNIST(
                root=root,
                train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(*IMG_STATS["fashionmnist"]),
                ]),
                download=True,
            )

            
            # # ★ 여기 root 경로 수정:  .../val  (images 아님!)
            # "TinyImageNet": lambda root: ImageFolder(
            #     root=os.path.join(root, "tiny-imagenet-200", "val"),
            #     transform=transforms.Compose([
            #         transforms.Resize((64, 64)),
            #         transforms.ToTensor(),
            #         transforms.Normalize(*IMG_STATS["TinyImageNet"]),
            #     ]),
            # ),
        }
        
        if args.data == 'svhn':
            dataset = datasets.SVHN(root='./data/', split='test', transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(*IMG_STATS["svhn"]),
                ]), download=True) 
            num_classes = 10
        elif args.data == 'mnist':
            dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(*IMG_STATS["mnist"]),
                ]), download=True)
            num_classes = 10
        elif args.data == 'fashionmnist':
            dataset = datasets.FashionMNIST(root='./data/', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(*IMG_STATS["fashionmnist"]),
                ]), download=True)
            num_classes = 10    
        else:
            dataset = CLASSIFICATION_DATASETS[args.data]("./data")
            num_classes = len(dataset.classes)

        counter = {i: 0 for i in range(num_classes)}
        sel_idx: List[int] = []
        for idx, (_, lbl) in tqdm(enumerate(dataset), total=len(dataset), desc="Sampling"):
            if counter[lbl] < args.samples_per_class:
                sel_idx.append(idx)
                counter[lbl] += 1
            if all(v >= args.samples_per_class for v in counter.values()):
                break

        loader = DataLoader(Subset(dataset, sel_idx), batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        # feats, lbls = extract_features(model, loader, args.mc_runs, device)
        feats, lbls = extract_features(model, loader, 5, device)

        #* [N, D] features and [N] labels

        Y, kl = run_embedding(
                feats.numpy(),
                method=args.clustering_method,
                perplexity=args.perplexity,
                n_neighbors=args.n_neighbors,
                min_dist=args.min_dist,
            )
        
        # Save feats, lbls, Y, kl 
        
        torch.save({
            'feats': feats,
            'lbls': lbls,
            'Y': Y,
            'kl': kl,
        }, args.weight.replace('.pth', '_clustering.pth'))
        sil, db, pr, gv = cluster_metrics(Y, lbls.numpy())
        rho = global_rank_corr(feats.numpy(), Y)

        print("\n=== Metrics ===")
        print(f"Silhouette={sil:.3f}  DaviesBouldin={db:.3f}  PR={pr:.2f}  GV={gv:.3e}  Spearmanρ={rho:.3f}\n")

        experiment_results['clustering_performance'] = {
            'silhouette': sil,
            'davies_bouldin': db,
            'pr': pr,
            'global_variance': gv,
            'spearman_rho': rho,
        }
    
    else:
        logger.warning("Clustering performance evaluation is only implemented for CIFAR-10 dataset.")

    # ──────────────────────────────────────────────
    # OOD Evaluation
    # ──────────────────────────────────────────────
    if args.ood is not None:
        logger.info(colored("Out-of-Distribution evaluation starting...", 'blue'))
        for ood in args.ood:
            args.data = ood
            print(f"Out of Distribution Dataset: {args.data}")
            _, out_data_loader = get_dataset(args, logger=logger)

            if args.type == 'dnn':
                experiment_results['ood_performance'][ood] = test_ood_detection_dnn(model, test_loader, out_data_loader, args=args)

            elif args.type == 'uni':
                experiment_results['ood_performance'][ood] = test_ood_detection_bnn(model, test_loader, out_data_loader, mc_runs=args.mc_runs, args=args)
            else:
                raise NotImplementedError("Not implemented yet")
    else:
        print(colored("No OOD datasets specified. Skipping OOD evaluation.", 'red'))

    logger.info(colored("Adversarial attack evaluation starting...", 'blue'))
    
    # Adversarial Attack Evaluation
    if args.type == 'uni':
        from adversarialattack import fgsm_clf, pgd_clf, eval_clf
        # FGSM
        adv_data, adv_labels = [], []
        for x,y in test_loader:
            x_adv = fgsm_clf(model, x, y, args.eps, device=device)
            adv_data.append(x_adv.cpu())
            adv_labels.append(y.cpu())
        
        x_adv = torch.cat(adv_data, dim=0)
        y_adv = torch.cat(adv_labels, dim=0)
        adversarial_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_adv, y_adv), batch_size=128, shuffle=False)
        
        fgsm_nll, fgsm_acc = eval_clf(model, adversarial_loader,
                            mc_runs=args.mc_runs, device=device)
        
        # PGD
        adv_data, adv_labels = [], []
        for x,y in test_loader:
            x_adv = pgd_clf(model, x, y, args.eps, device=device)
            adv_data.append(x_adv.cpu())
            adv_labels.append(y.cpu())
        x_adv = torch.cat(adv_data, dim=0)
        y_adv = torch.cat(adv_labels, dim=0)
        adversarial_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_adv, y_adv), batch_size=128, shuffle=False)
        
        pgd_nll, pgd_acc = eval_clf(model, adversarial_loader,
                            mc_runs=args.mc_runs, device=device)
        
        experiment_results['adversarial_performance'] = {
            'fgsm': {'nll': fgsm_nll, 'accuracy': fgsm_acc},
            'pgd': {'nll': pgd_nll, 'accuracy': pgd_acc}
        }
    else:
        print(colored("Adversarial attack evaluation is only implemented for BNN models.", 'red'))
    
    # Summarize results
    print("\n=== Summary of Results ===")
    print("In-Distribution Performance:")
    for metric, value in experiment_results['id_performance'].items():
        print(f"  {metric}: {value:.4f}")
    if experiment_results['ood_performance']:
        print("\nOut-of-Distribution Performance:")
        for ood, metrics in experiment_results['ood_performance'].items():
            print(f"  OOD Dataset: {ood}")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")
    if experiment_results.get('clustering_performance'):
        print("\nClustering Performance:")
        for metric, value in experiment_results['clustering_performance'].items():
            print(f"  {metric}: {value:.4f}")
    if experiment_results.get('adversarial_performance'):
        print("\nAdversarial Performance:")
        for attack, metrics in experiment_results['adversarial_performance'].items():
            print(f"  Attack: {attack}")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")
                
    # ──────────────────────────────────────────────
    # 최종 결과 파일 저장
    # ──────────────────────────────────────────────
    
    if args.save_results:
        save_path = args.weight.replace('.pth', '_results2.json')
        with open(save_path, 'w') as f:
            json.dump(experiment_results, f, indent=4)
        print(colored(f"\nAll experiment results saved to: {save_path}", 'magenta'))

def main(args):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    model = get_model(args = args, logger = logger)
    best_model_weight = torch.load(args.weight, map_location=device)

    evaluate(model, best_model_weight, device, args, logger)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Test a Pretrained Model')
    parser.add_argument('--type', type=str, help='[dnn, uni, multi]')
    parser.add_argument('--model', type=str, help='Model to train [resnet20, densenet30, vgg7]')
    parser.add_argument('--bs', type=int, default=128, help='Batch size')
    parser.add_argument('--data', type=str, default='cifar10', help='Dataset to use [cifar10]')
    parser.add_argument('--mc_runs', type=int, default=30, help='Monte Carlo runs')
    parser.add_argument('--weight', type=str, help='Path to load weights')
    parser.add_argument('--ood', nargs='+', help='Out-of-distribution datasets')
    parser.add_argument('--scale', type=str, default='BS', help='KLD scale')
    parser.add_argument('--prior_type', type=str, help='Prior type [normal, laplace]')
    parser.add_argument('--multi-gpu', action='store_true', help='Use multi-GPU')
    parser.add_argument('--clustering_method', type=str, default='umap', help='Clustering method for visualization [tsne, umap]'   )
    parser.add_argument('--perplexity', type=int, default=30)          # t‑SNE
    parser.add_argument('--n_neighbors', type=int, default=15)         # UMAP
    parser.add_argument('--min_dist', type=float, default=0.1)         # UMAP
    parser.add_argument('--eps', type=float, default=0.02, help='Epsilon for adversarial attack')
    parser.add_argument('--save_results', action='store_true', default=True, help='Save results to a json file')
    parser.add_argument('--sparsity', type=float, default=0.0, help='sparsity level for sparse prior')
    parser.add_argument('--std', type=float, default=1.0, help='std for normal prior')
    args = parser.parse_args()

    # ast.literal_eval을 사용하여 문자열 형식의 리스트를 실제 리스트로 변환
    if isinstance(args.ood, list) and len(args.ood) == 1:
        try:
            args.ood = ast.literal_eval(args.ood[0])
        except (ValueError, SyntaxError):
            # 만약 변환에 실패하면, 원래 값(문자열 리스트)을 그대로 사용
            pass

    # 로깅 설정
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    print(colored(args, 'green'))
    
    # Load json config if exists
    import os, json, ast
    if os.path.exists(args.weight.replace('.pth', '_results.json')):
        with open(args.weight.replace('.pth', '_results.json'), 'r') as f:
            saved_args = json.load(f)
        for key, value in saved_args['info'].items():
            if hasattr(args, key):
                # Set MC runs as 30 for evaluation
                
                if key == 'mc_runs':
                    value = 30
                    
                elif key == 'weight':
                    value = args.weight
                
                setattr(args, key, value)
                
            # if the argument does not exist in current args, add it
            else:
                setattr(args, key, value)
        print(colored("Loaded args from config file:", 'yellow'))
        print(colored(saved_args['info'], 'red'))
    else:
        # Load "config.txt" (exists in same folder as weights)
        # Get directory path from args.weight
        dir_path = os.path.dirname(args.weight)
        config_path = os.path.join(dir_path, 'config.txt')
        if os.path.exists(config_path):
            # Config contains lines like key:value
            with open(config_path, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        if key in ['weight', 'data', 'ood']:
                            continue
                        elif key == 'pruned_dnn_acc':
                            args.pruned_dnn_acc = float(value)
                        elif key == 'pruned_dnn_loss':
                            args.pruned_dnn_loss = float(value)
                        elif key == 'pruned_dnn_sparsity':
                            args.pruned_dnn_sparsity = float(value)
                        if hasattr(args, key):
                            # Convert to appropriate type
                            current_val = getattr(args, key)
                            current_type = type(current_val)

                            if current_val is None and key == 'ood': # Special case for ood which is a list
                                try:
                                    value = ast.literal_eval(value)
                                except (ValueError, SyntaxError):
                                    pass # Keep as string if it fails
                            elif current_type == bool:
                                value = value.lower() in ['true', '1', 'yes']
                            elif current_val is not None:
                                value = current_type(value)
                            else: # For other args with default None, try to infer
                                try:
                                    value = ast.literal_eval(value)
                                except (ValueError, SyntaxError):
                                    pass # Keep as string
                            
                            # Set MC runs as 30 for evaluation
                            if key == 'mc_runs':
                                value = 30
                            
                            setattr(args, key, value)
                        else:
                            # If the argument does not exist in current args, add it
                            try:
                                value = ast.literal_eval(value)
                            except (ValueError, SyntaxError):
                                pass # Keep as string if it fails

    print(args)
    main(args)