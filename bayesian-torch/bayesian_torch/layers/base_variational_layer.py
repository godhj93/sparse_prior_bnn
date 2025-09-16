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

def standard_normal_cdf(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

class BaseVariationalLayer_(nn.Module):
    def __init__(self, spike_and_slab_pi=0.5, horseshoe_tau=0.1):
        super().__init__()
        self._dnn_to_bnn_flag = False
        
    @property
    def dnn_to_bnn_flag(self):
        return self._dnn_to_bnn_flag

    @dnn_to_bnn_flag.setter
    def dnn_to_bnn_flag(self, value):
        self._dnn_to_bnn_flag = value

    def kl_div(self, mu_q, sigma_q, mu_p, sigma_p, prior_type):
        
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

        elif prior_type == 'student-t':
            if mu_q.requires_grad:
                L=10
            else:
                L=1
            nu = torch.tensor(1.0, device=mu_q.device)
    
            mu_q_expanded = mu_q.unsqueeze(0).expand(L, *mu_q.shape)
            sigma_q_expanded = sigma_q.unsqueeze(0).expand(L, *sigma_q.shape)
            
            epsilon = torch.randn_like(mu_q_expanded)
            w_sample = mu_q_expanded + sigma_q_expanded * epsilon
            
            log_q = Normal(mu_q_expanded, sigma_q_expanded).log_prob(w_sample)

            log_p_norm_const = torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2) - \
                            0.5 * torch.log(nu * torch.pi)
            log_p = log_p_norm_const - ((nu + 1) / 2) * torch.log(1 + (w_sample**2) / nu)
            
            kl_samples = log_q - log_p
            
            kl_avg_over_L = kl_samples.mean(dim=0) # shape: [S]
            return kl_avg_over_L.mean()          # shape: scala
            
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
