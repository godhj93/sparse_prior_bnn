# Copyright (C) 2024 Intel Labs
#
# BSD-3-Clause License
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
# OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# ===============================================================================================
import torch
import torch.nn as nn
import torch.distributions as distributions
from itertools import repeat
import collections

standard_normal = distributions.Normal(0, 1)    

def get_kernel_size(x, n):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

import torch
import torch.nn as nn
from torch.distributions import Normal, HalfCauchy, LogNormal
import math

# 기존 코드의 standard_normal.cdf를 대체하기 위한 간단한 에러 함수(erf) 기반 cdf 구현
def standard_normal_cdf(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

class BaseVariationalLayer_(nn.Module):
    def __init__(self, spike_and_slab_pi=0.5, horseshoe_tau=0.1):
        """
        Spike-and-Slab과 Horseshoe prior를 위한 파라미터 초기화 추가
        
        Parameters:
            * spike_and_slab_pi (float): Spike-and-Slab prior의 포함 확률(pi) 하이퍼파라미터
            * horseshoe_tau (float): Horseshoe prior의 전역 수축 파라미터(tau) 초기값
        """
        super().__init__()
        self._dnn_to_bnn_flag = False
        
        
        # # --- Spike-and-Slab을 위한 파라미터 ---
        # # 사후 분포의 포함 확률(alpha)을 학습 가능한 파라미터로 선언합니다.
        # # 로짓(logit) 공간에서 최적화하는 것이 수치적으로 안정적입니다.
        # # 이 파라미터는 가중치(mu_q, sigma_q)와 동일한 shape를 가져야 하므로,
        # # 실제 Linear 또는 Conv 레이어의 __init__에서 가중치 shape에 맞게 초기화되어야 합니다.
        # # 예시로 None으로 두고, 실제 사용 시 초기화 필요함을 명시합니다.
        # self.log_alpha = nn.Parameter(torch.Tensor(1)) # 실제 레이어에서 shape 지정 필요
        
        # # 사전 분포의 하이퍼파라미터 pi
        # self.spike_and_slab_pi = spike_and_slab_pi

        # # --- Horseshoe를 위한 파라미터 ---
        # # 지역(lambda) 및 전역(tau) 수축 파라미터의 변분 사후 분포로 LogNormal을 사용합니다.
        # # LogNormal의 파라미터(loc, scale)를 학습 가능한 파라미터로 선언합니다.
        # self.l_loc = nn.Parameter(torch.Tensor(1)) # lambda의 loc (local)
        # self.l_scale = nn.Parameter(torch.Tensor(1)) # lambda의 scale (local)
        # self.t_loc = nn.Parameter(torch.tensor(math.log(horseshoe_tau))) # tau의 loc (global)
        # self.t_scale = nn.Parameter(torch.tensor(0.1)) # tau의 scale (global)

        # 실제 사용 시에는 nn.init 등으로 위 파라미터들을 적절히 초기화해야 합니다.
        # 예: nn.init.normal_(self.log_alpha, -3.0, 0.1)

    @property
    def dnn_to_bnn_flag(self):
        return self._dnn_to_bnn_flag

    @dnn_to_bnn_flag.setter
    def dnn_to_bnn_flag(self, value):
        self._dnn_to_bnn_flag = value

    def kl_div(self, mu_q, sigma_q, mu_p, sigma_p, prior_type):
        """
        다양한 사전 분포에 대한 KL Divergence를 계산합니다.
        Spike-and-Slab과 Horseshoe가 추가되었습니다.
        """
        
        if prior_type == 'normal':
            kl = torch.log(sigma_p) - torch.log(sigma_q) + \
                (sigma_q**2 + (mu_q - mu_p)**2) / (2 * (sigma_p**2)) - 0.5
            return kl.mean()

        elif prior_type == 'laplace':
            b_p = torch.tensor(1.0, device=mu_q.device)
            mu_p = torch.tensor(0.0, device=mu_q.device)
            mu_q_offset = mu_q - mu_p
            
            exp_abs_val = sigma_q * torch.sqrt(torch.tensor(2.0 / torch.pi)) * \
                        torch.exp(-mu_q_offset**2 / (2 * sigma_q**2)) + \
                        mu_q_offset * (1 - 2 * standard_normal_cdf(-mu_q_offset / sigma_q))
            
            kl = torch.log(2 * b_p) - 0.5 * torch.log(2 * torch.pi * sigma_q**2) - 0.5 + \
                (1 / b_p) * exp_abs_val
            return kl.mean()

        # elif prior_type == 'student-t':
        #     nu = torch.tensor(1.0, device=mu_q.device)
        #     epsilon = torch.randn_like(mu_q)
        #     w_sample = mu_q + sigma_q * epsilon
            
        #     log_q = Normal(mu_q, sigma_q).log_prob(w_sample)
            
        #     log_p_norm_const = torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2) - \
        #                        0.5 * torch.log(nu * torch.pi)
        #     log_p = log_p_norm_const - ((nu + 1) / 2) * torch.log(1 + (w_sample**2) / nu)
            
        #     kl = log_q - log_p
        #     return kl.mean()
        elif prior_type == 'student-t':
            if mu_q.requires_grad:
                L=10
            else:
                L=1
            # print("Using Monte Carlo with L=10 samples for Student-t KL Divergence")
            nu = torch.tensor(1.0, device=mu_q.device)
    
            # 1. L개의 샘플을 담을 수 있도록 텐서 차원 확장
            #    mu_q shape: [S] -> [1, S] -> [L, S]
            #    sigma_q shape: [S] -> [1, S] -> [L, S]
            mu_q_expanded = mu_q.unsqueeze(0).expand(L, *mu_q.shape)
            sigma_q_expanded = sigma_q.unsqueeze(0).expand(L, *sigma_q.shape)
            
            # 2. L개의 샘플을 한 번에 생성
            #    epsilon shape: [L, S]
            epsilon = torch.randn_like(mu_q_expanded)
            w_sample = mu_q_expanded + sigma_q_expanded * epsilon
            
            # 3. log q(w)를 L개 샘플에 대해 한 번에 계산
            #    결과 shape: [L, S]
            log_q = Normal(mu_q_expanded, sigma_q_expanded).log_prob(w_sample)

            # 4. log p(w)를 L개 샘플에 대해 한 번에 계산
            #    결과 shape: [L, S]
            log_p_norm_const = torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2) - \
                            0.5 * torch.log(nu * torch.pi)
            log_p = log_p_norm_const - ((nu + 1) / 2) * torch.log(1 + (w_sample**2) / nu)
            
            # 5. L개 샘플에 대한 KL Divergence를 한 번에 계산
            #    결과 shape: [L, S]
            kl_samples = log_q - log_p
            
            # 6. 먼저 L 차원(dim=0)에 대해 평균을 내고,
            #    그 다음 모든 가중치 차원에 대해 평균을 냄
            kl_avg_over_L = kl_samples.mean(dim=0) # shape: [S]
            return kl_avg_over_L.mean()          # shape: scala
            
        # --- Spike-and-Slab 구현 ---
        # elif prior_type == 'spike-and-slab':
        #     # 1. 사후 분포의 포함 확률 alpha_q 계산 (로짓 -> 확률)
        #     #    수치적 안정을 위해 작은 epsilon 값을 더하고 뺌
        #     eps = 1e-8
        #     alpha_q = torch.sigmoid(self.log_alpha).clamp(eps, 1.0 - eps)
            
        #     # 2. KL(Bernoulli(alpha_q) || Bernoulli(pi_p)) 계산
        #     #    이산적인 부분(Spike)에 대한 KL Divergence
        #     pi_p = self.spike_and_slab_pi
        #     kl_bernoulli = alpha_q * (torch.log(alpha_q / pi_p)) + \
        #                    (1 - alpha_q) * (torch.log((1 - alpha_q) / (1 - pi_p)))
                           
        #     # 3. KL(N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2)) 계산
        #     #    연속적인 부분(Slab)에 대한 KL Divergence
        #     kl_gaussian = torch.log(sigma_p) - torch.log(sigma_q) + \
        #                   (sigma_q**2 + (mu_q - mu_p)**2) / (2 * (sigma_p**2)) - 0.5
                          
        #     # 4. 최종 KL Divergence 결합
        #     #    KL = KL_Bernoulli + alpha_q * KL_Gaussian
        #     kl = kl_bernoulli + alpha_q * kl_gaussian
        #     return kl.mean()
      
       

        else:
            raise ValueError(f"Unknown prior_type: {prior_type}")
    
    def kl_div_spike_and_slab(self, mu_q, sigma_q, log_alpha, mu_p, sigma_p, pi_p):
        
        # 1. 사후 분포의 포함 확률 alpha_q 계산 (로짓 -> 확률)
        #    수치적 안정을 위해 작은 epsilon 값을 더하고 뺌
        eps = 1e-8
        alpha_q = torch.sigmoid(log_alpha).clamp(eps, 1.0 - eps)
        
        # 2. KL(Bernoulli(alpha_q) || Bernoulli(pi_p)) 계산
        #    이산적인 부분(Spike)에 대한 KL Divergence
        kl_bernoulli = alpha_q * (torch.log(alpha_q / pi_p)) + \
                        (1 - alpha_q) * (torch.log((1 - alpha_q) / (1 - pi_p)))
                        
        # 3. KL(N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2)) 계산
        #    연속적인 부분(Slab)에 대한 KL Divergence
        kl_gaussian = torch.log(sigma_p) - torch.log(sigma_q) + \
                        (sigma_q**2 + (mu_q - mu_p)**2) / (2 * (sigma_p**2)) - 0.5
                        
        # 4. 최종 KL Divergence 결합
        #    KL = KL_Bernoulli + alpha_q * KL_Gaussian
        kl = kl_bernoulli + alpha_q * kl_gaussian
        return kl.mean()
      
    def kl_div_horseshoe(self, mu_q, sigma_q, l_loc, l_scale, t_loc, t_scale):
        
        eps = 1e-8
        
        # --- 수정 지점 ---
        # LogNormal의 스케일 파라미터가 0이 되지 않도록 clamp를 추가합니다.
        q_lambda_scale = torch.exp(l_scale).clamp(min=eps)
        q_tau_scale = torch.exp(t_scale).clamp(min=eps)

        # 수정된 스케일 값으로 분포를 생성합니다.
        w_dist = Normal(mu_q, sigma_q)
        lambda_dist = LogNormal(l_loc, q_lambda_scale)
        tau_dist = LogNormal(t_loc, q_tau_scale)

        # 샘플링 후에도 clamp를 유지하는 것이 더 안전합니다.
        w_sample = w_dist.rsample()
        lambda_sample = lambda_dist.rsample().clamp(min=eps)
        tau_sample = tau_dist.rsample().clamp(min=eps)

        log_q = w_dist.log_prob(w_sample).sum() + \
                lambda_dist.log_prob(lambda_sample).sum() + \
                tau_dist.log_prob(tau_sample).sum()
        
        p_w = Normal(0, (lambda_sample * tau_sample).clamp(min=eps))
        p_lambda = HalfCauchy(1.0)
        p_tau = HalfCauchy(1.0)

        log_p = p_w.log_prob(w_sample).sum() + \
                p_lambda.log_prob(lambda_sample).sum() + \
                p_tau.log_prob(tau_sample).sum()

        kl = log_q - log_p
        
        return kl
# class BaseVariationalLayer_(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self._dnn_to_bnn_flag = False

#     @property
#     def dnn_to_bnn_flag(self):
#         return self._dnn_to_bnn_flag

#     @dnn_to_bnn_flag.setter
#     def dnn_to_bnn_flag(self, value):
#         self._dnn_to_bnn_flag = value

#     def kl_div(self, mu_q, sigma_q, mu_p, sigma_p, prior_type):
#         """
#         Calculates kl divergence between two gaussians (Q || P)

#         Parameters:
#              * mu_q: torch.Tensor -> mu parameter of distribution Q
#              * sigma_q: torch.Tensor -> sigma parameter of distribution Q
#              * mu_p: float -> mu parameter of distribution P
#              * sigma_p: float -> sigma parameter of distribution P

#         returns torch.Tensor of shape 0
#         """
        
#         if prior_type == 'normal':
#         # This implementation is correct.
#             kl = torch.log(sigma_p) - torch.log(
#             sigma_q) + (sigma_q**2 + (mu_q - mu_p)**2) / (2 * (sigma_p**2)) - 0.5
#             return kl.mean()

#         elif prior_type == 'laplace':
            
#             # --- 샘플링 대신 해석적 해를 사용한 새로운 구현 ---
            
#             # Laplace 분포의 스케일 파라미터 b는 sigma_p로 전달됩니다.
#             b_p = torch.tensor(1.0)
#             mu_p = torch.tensor(0.0)
#             # Laplace 분포의 위치(mu_p)를 고려하여 mu_q의 오프셋을 계산합니다.
#             mu_q_offset = mu_q - mu_p
            
#             # 해석적 공식에 따라 KL Divergence를 계산합니다.
#             # KL(q||p) = log(2b) - 0.5*log(2*pi*sigma^2) - 0.5 + (1/b) * E_q[|w - mu_p|]
            
#             # E_q[|w - mu_p|]는 접힌 정규 분포(Folded Normal)의 평균입니다.
#             exp_abs_val = sigma_q * torch.sqrt(torch.tensor(2.0 / torch.pi)) * \
#                         torch.exp(-mu_q_offset**2 / (2 * sigma_q**2)) + \
#                         mu_q_offset * (1 - 2 * standard_normal.cdf(-mu_q_offset / sigma_q))
            
#             # 모든 항을 결합합니다.
#             kl = torch.log(2 * b_p) - 0.5 * torch.log(2 * torch.pi * sigma_q**2) - 0.5 + \
#                 (1 / b_p) * exp_abs_val
                
#             return kl.mean()

#         elif prior_type == 'student-t':
#             # Student-t Prior (몬테카를로 근사)
            
#             # 자유도(nu)는 하이퍼파라미터입니다. 값이 작을수록 꼬리가 두껍습니다.
#             nu = torch.tensor(1.0, device=mu_q.device)
            
#             # 1. 재매개변수화 트릭을 이용한 샘플링 (S=1)
#             #    학습 중에는 배치 단위로 평균을 내므로 샘플은 1개로 충분합니다.
#             epsilon = torch.randn_like(mu_q)
#             w_sample = mu_q + sigma_q * epsilon
            
#             # 2. 샘플에 대한 log q(w|D) 계산 (가우시안 PDF)
#             log_q = -torch.log(sigma_q) - 0.5 * torch.log(torch.tensor(2.0 * torch.pi, device=mu_q.device)) - \
#                     ((w_sample - mu_q)**2) / (2 * sigma_q**2)
            
#             # 3. 샘플에 대한 log p(w) 계산 (Student-t PDF)
#             #    p(w)의 정규화 상수 계산 (로그 스케일)
#             log_p_norm_const = torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2) - \
#                                0.5 * torch.log(nu * torch.pi)
            
#             #    전체 log p(w) 계산
#             log_p = log_p_norm_const - ((nu + 1) / 2) * torch.log(1 + (w_sample**2) / nu)
            
#             # 4. KL Divergence 근사: E_q[log q(w) - log p(w)]
#             kl = log_q - log_p
            
#             return kl.mean()

#         else:
#             raise ValueError(f"Unknown prior_type: {prior_type}")

#     def kl_div_multivariate_gaussian(self, mu_q, sigma_q, mu_p, sigma_p, device='cuda'):
#         """
#         Calculates kl divergence between two multivariate gaussians (Q || P)

#         Parameters:
#              * mu_q: torch.Tensor -> mu parameter of distribution Q
#              * sigma_q: torch.Tensor -> sigma parameter of distribution Q
#              * mu_p: float -> mu parameter of distribution P
#              * sigma_p: float -> sigma parameter of distribution P

#         returns torch.Tensor of shape 0
#         """

#         kl = 0.5 * (torch.logdet(sigma_p).to(device) - torch.logdet(sigma_q).to(device) +
#                     torch.trace(torch.matmul(sigma_q.to(device), torch.inverse(sigma_p).to(device))) +
#                     torch.matmul(torch.matmul((mu_q.to(device) - mu_p.to(device)).unsqueeze(-1).permute(1, 0), torch.inverse(sigma_p.to(device))), (mu_q.cuda() - mu_p.cuda()).unsqueeze(-1)).squeeze() - mu_p.shape[0])
#         return kl.mean()
    
        