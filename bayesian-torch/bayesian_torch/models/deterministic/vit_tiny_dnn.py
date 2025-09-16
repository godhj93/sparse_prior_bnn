import torch
import torch.nn as nn
from typing import Type
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer, Attention as TimmAttention
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss

__all__ = ['ViT_Tiny_dnn', 'vit_tiny_dnn']

class ViT_Tiny_dnn(nn.Module):
    def __init__(self, img_size = 32, num_classes=100, model = 'nano'):
        super().__init__()
        
        # load ViT backbone
        if model == 'original':
            if img_size == 32:
                patch_size = 4
            elif img_size == 64:
                patch_size = 8
            else:
                raise ValueError("img_size must be 32 or 64 for the 'original' model.")
            self.base_model = VisionTransformer(img_size=img_size, patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=2)
            
        elif model == 'nano':
            self.base_model = VisionTransformer(
                img_size=img_size,
                patch_size=8,       # 큰 패치로 시퀀스 길이를 16으로 단축
                embed_dim=96,       # 임베딩 차원을 낮게 설정
                depth=4,            # 얕은 깊이
                num_heads=3,        # 헤드 수 감소
                mlp_ratio=2.0,      # MLP 확장 비율 축소 (기본값 4.0)
                num_classes=num_classes
            )
        elif model == 'micro':
            self.base_model = VisionTransformer(
                img_size=img_size,
                patch_size=4,       # 작은 패치로 더 많은 특징 학습 (시퀀스 길이 64)
                embed_dim=192,      # 표준적인 경량 모델의 임베딩 차원
                depth=6,            # 중간 수준의 깊이
                num_heads=3,        # 임베딩 차원에 맞춰 헤드 수 조절
                mlp_ratio=2.0,      # MLP 확장 비율 축소
                num_classes=num_classes
            )
        elif model == 'pico':
            self.base_model = VisionTransformer(
        img_size=img_size,
        patch_size=4,
        embed_dim=256,      # 표현력을 위해 임베딩 차원 확장
        depth=7,            # 모델 깊이 추가
        num_heads=4,        # 확장된 임베딩 차원에 맞춰 헤드 수 증가
        mlp_ratio=3.0,      # 성능을 위해 MLP 비율을 소폭 상향
        num_classes=num_classes
    )
        # replace head
        num_ftrs = self.base_model.head.in_features
        self.base_model.head = nn.Linear(num_ftrs, num_classes)
       
        
    def forward(self, x):
        out = self.base_model(x)
        return out

def vit_tiny_dnn(img_size, num_classes=100, model='nano'):
    """
    Create a ViT Tiny model with DNN to BNN conversion.
    
    Args:
        num_classes (int): Number of output classes.
        model (str): Model variant ('original', 'nano', 'micro', 'pico').
    
    Returns:
        ViT_Tiny_dnn: Configured ViT Tiny model.
    """
    model = ViT_Tiny_dnn(img_size, num_classes=num_classes, model=model)
    
    return model