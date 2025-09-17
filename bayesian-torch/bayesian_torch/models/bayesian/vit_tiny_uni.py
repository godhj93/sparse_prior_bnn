import torch
import torch.nn as nn
from typing import Type
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
from bayesian_torch.layers.variational_layers.linear_variational import LinearReparameterization

__all__ = ['ViT_Tiny_uni', 'vit_tiny_uni']

class ViT_Tiny_uni(nn.Module):
    def __init__(self, num_classes=100, model = 'nano', img_size = 32, prior_type=None):
        super().__init__()
        
        assert prior_type in ['normal', 'laplace', 'student-t', 'spike-and-slab'], "prior_type must be either 'normal', 'laplace', 'student-t', or 'spike-and-slab'"
        const_bnn_prior_parameters = {
            'prior_mu': 0.0,
            'prior_sigma': 1.0,
            'prior_type': prior_type,
            'posterior_mu_init': 0.0,
            'posterior_rho_init': -3.0,
            'type': 'Reparameterization',
            'moped_enable': False,
            'moped_delta': 0.5,
            
        }

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

        num_ftrs = self.base_model.head.in_features
        self.base_model.head = nn.Linear(num_ftrs, num_classes)

        # Apply BNN conversion first
        dnn_to_bnn(self.base_model, const_bnn_prior_parameters)

        # Restore original Conv2d in patch embedding
        old_proj = self.base_model.patch_embed.proj
        conv2d_params = {
            "in_channels": old_proj.in_channels,
            "out_channels": old_proj.out_channels,
            "kernel_size": old_proj.kernel_size,
            "stride": old_proj.stride,
            "padding": old_proj.padding,
            "dilation": old_proj.dilation,
            "groups": old_proj.groups,
            "bias": old_proj.bias is not None,
            # "padding_mode": old_proj.padding_mode
        }
        self.base_model.patch_embed.proj = nn.Conv2d(**conv2d_params)
        
    def forward(self, x):
        out = self.base_model(x)
        kl = get_kl_loss(self.base_model)
        return out, kl

def vit_tiny_uni(num_classes=100, model='nano', img_size=32, prior_type=None):
    """
    Create a ViT Tiny model with Bayesian conversion.
    
    Args:
        num_classes (int): Number of output classes.
        model (str): Model variant, e.g., 'nano', 'micro', etc.
    
    Returns:
        ViT_Tiny_uni: A ViT Tiny model with Bayesian layers.
    """
    return ViT_Tiny_uni(num_classes=num_classes, model=model, img_size=img_size, prior_type=prior_type)