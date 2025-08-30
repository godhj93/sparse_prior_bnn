import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
# from utils.models.densenet_uni import densenet_bc_30_uni
from tqdm import tqdm
import torch

"""
Dataset
"""
classification_datasets = {
    "CIFAR10": lambda root: datasets.CIFAR10(
        root=root, train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ]),
        download=True
    ),
    "MNIST": lambda root: datasets.MNIST(
        root=root, train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]),
        download=True
    ),
    "FashionMNIST": lambda root: datasets.FashionMNIST(
        root=root, train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]),
        download=True
    ),
}

"""
Inference
"""
@torch.no_grad()
def eval_clf(model, loader, mc_runs, device):
    model.eval()
    nll_sum = 0.0
    correct = 0
    total = 0
    for x, y in tqdm(loader, desc="Eval"):
        x, y = x.to(device), y.to(device)
        logits_mc = []
        for _ in range(mc_runs):
            logits, kl = model(x)
            logits_mc.append(logits)
        probs = F.softmax(torch.stack(logits_mc), dim=-1).mean(0) 
        nll_sum += F.nll_loss(probs.log(), y, reduction='sum').item()
        preds = probs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    nll = nll_sum / total
    acc = correct / total
    return nll, acc

def fgsm_clf(model, x, y, eps=0.02, device='cuda'):
    x_adv = x.detach().clone().to(device).requires_grad_(True) 
    loss = F.cross_entropy(model(x_adv)[0], y.to(device)) 
    loss.backward() 
    x_adv = x_adv + eps * x_adv.grad.sign()
    return x_adv.clamp(0, 1).detach()
'''
def pgd_clf_hj(model, x, y, eps=0.02, alpha=0.004, steps=20, device='cuda'):
    x = x.to(device)
    y = y.to(device)
    x_adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0,1).to(device)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        
        logits_mc = [model(x_adv)[0] for _ in range(30)] # mc_runs is 30
        avg_logits = torch.stack(logits_mc).mean(0)
        # loss = F.cross_entropy(model(x_adv)[0], y.to(device))
        loss = F.cross_entropy(avg_logits, y)
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
            x_adv = x_adv.clamp(0, 1)
    return x_adv.detach()
'''

import torch
import torch.nn.functional as F

def pgd_clf_hj(model, x, y, eps=0.02, alpha=0.004, steps=20, mc_runs=1, device='cuda'):
    model.eval()  # 평가 모드로 설정
    model.to(device)  # 모델을 지정된 장치로 이동
    x = x.to(device)
    y = y.to(device)
    x_adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1).to(device)

    for step in range(steps):
        x_adv.requires_grad_(True)
        
        if x_adv.grad is not None:
            x_adv.grad.zero_()

        # MC 루프 시작
        for i in range(mc_runs):
            # [중요] 루프 내에서 생성되는 모든 텐서가 루프 종료 시 사라지도록 관리
            logits = model(x_adv)[0]
            loss = F.cross_entropy(logits, y) / mc_runs
            loss.backward()

            # [추가] 계산 그래프를 사용한 변수들을 명시적으로 삭제하여 메모리 누수 가능성을 차단합니다.
            del logits, loss
        
        # [추가] 하나의 PGD 스텝에 대한 모든 그래디언트 누적이 끝난 후,
        # 파이토치의 GPU 캐시 메모리를 정리합니다.
        # 이 코드는 성능 저하를 일으킬 수 있으나, 메모리 누수 확인에 매우 효과적입니다.
        # torch.cuda.empty_cache()

        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
            x_adv = x_adv.clamp(0, 1)
            
    return x_adv.detach()

def pgd_clf(model, x, y, eps=0.02, alpha=0.004, steps=20, device='cuda'):
    x = x.to(device)
    y = y.to(device)
    x_adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0,1).to(device)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        
        loss = F.cross_entropy(model(x_adv)[0], y.to(device))
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
            x_adv = x_adv.clamp(0, 1)
    return x_adv.detach()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', required=True,
                        help='checkpoint *.pth to load')
    parser.add_argument('--data', required=True,
                        choices=list(classification_datasets.keys()))
    parser.add_argument('--mc_runs', type=int, default=30,
                        help='MC 샘플링 횟수')
    parser.add_argument('--attack', choices=['none','fgsm','pgd'],
                        default='none', help='이상치 공격 종류')
    parser.add_argument('--eps', type=float, default=0.02,
                        help='공격 강도 (L∞)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_set = classification_datasets[args.data]('./data')
    test_loader = DataLoader(test_set, batch_size=128,
                             shuffle=False, num_workers=4)

    model = densenet_bc_30_uni().to(device)
    ckpt = torch.load(args.weight, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    if args.attack != 'none':
        adv_data, adv_labels = [], []
        for x, y in test_loader:
            if args.attack == 'fgsm':
                x_adv = fgsm_clf(model, x, y, eps=args.eps, device=device) 
            else:
                x_adv = pgd_clf(model, x, y, eps=args.eps,
                                alpha=args.eps, steps=10,
                                device=device)
            adv_data.append(x_adv.cpu()); adv_labels.append(y)
        x_adv = torch.cat(adv_data, dim=0) 
        y_adv = torch.cat(adv_labels, dim=0) 
        test_loader = DataLoader(
            torch.utils.data.TensorDataset(x_adv, y_adv),
            batch_size=128, shuffle=False
        )
        print(f"[!] Using adversarial data ({args.attack}, ε={args.eps})")

    nll, acc = eval_clf(model, test_loader,
                        mc_runs=args.mc_runs,
                        device=device)
    
    print(f"Dataset : {args.data}")
    print(f"MC Runs : {args.mc_runs}")
    print(f"NLL   : {nll:.4f}")
    print(f"Acc   : {acc*100:.2f}%")

if __name__ == "__main__":
    main()
