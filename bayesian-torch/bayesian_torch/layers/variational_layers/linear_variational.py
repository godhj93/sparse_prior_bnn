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
#
# Linear Reparameterization Layers with reparameterization estimator to perform
# variational inference in Bayesian neural networks. Reparameterization layers
# enables Monte Carlo approximation of the distribution over 'kernel' and 'bias'.
#
# Kullback-Leibler divergence between the surrogate posterior and prior is computed
# and returned along with the tensors of outputs after linear opertaion, which is
# required to compute Evidence Lower Bound (ELBO).
#
# @authors: Ranganath Krishnan
# ======================================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
from ..base_variational_layer import BaseVariationalLayer_
import math
from torch.quantization.observer import HistogramObserver, PerChannelMinMaxObserver, MinMaxObserver
from torch.quantization.qconfig import QConfig


class LinearReparameterization(BaseVariationalLayer_):
    def __init__(self,
                 in_features,
                 out_features,
                 prior_mean=0,
                 prior_variance=1,
                 prior_type=None,
                 posterior_mu_init=0,
                 posterior_rho_init=-3.0,
                 spike_and_slab_pi=0.5, # Spike-and-Slab prior
                 bias=True):
        """
        Implements Linear layer with reparameterization trick.

        Inherits from bayesian_torch.layers.BaseVariationalLayer_

        Parameters:
            in_features: int -> size of each input sample,
            out_features: int -> size of each output sample,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            bias: bool -> if set to False, the layer will not learn an additive bias. Default: True,
        """
        super(LinearReparameterization, self).__init__(spike_and_slab_pi=spike_and_slab_pi)
        assert prior_type is not None, "prior_type must be specified for LinearReparameterization layer"
        assert prior_type in ['normal', 'laplace', 'student-t', 'spike-and-slab', 'horseshoe'], "prior_type must be one of ['normal', 'laplace', 'student-t', 'spike-and-slab', 'horseshoe']"

        self.in_features = in_features
        self.out_features = out_features
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init,  # mean of weight
        self.prior_type = prior_type # Store prior type
        # variance of weight --> sigma = log (1 + exp(rho))
        self.posterior_rho_init = posterior_rho_init,
        self.bias = bias

        self.mu_weight = Parameter(torch.Tensor(out_features, in_features))
        self.rho_weight = Parameter(torch.Tensor(out_features, in_features))
        if prior_type == 'spike-and-slab':
            self.spike_and_slab_pi = spike_and_slab_pi
            self.log_alpha_weight = Parameter(torch.Tensor(out_features, in_features)) # Spike-and-Slab prior
        elif prior_type == 'horseshoe':
            self.l_loc_weight = Parameter(torch.Tensor(out_features, in_features))
            self.l_scale_weight = Parameter(torch.Tensor(out_features, in_features))
            self.t_loc_weight = Parameter(torch.Tensor(1))
            self.t_scale_weight = Parameter(torch.Tensor(1))
            
            
        self.register_buffer('eps_weight',
                             torch.Tensor(out_features, in_features),
                             persistent=False)
        self.register_buffer('prior_weight_mu',
                             torch.Tensor(out_features, in_features),
                             persistent=False)
        self.register_buffer('prior_weight_sigma',
                             torch.Tensor(out_features, in_features),
                             persistent=False)
        if bias:
            self.mu_bias = Parameter(torch.Tensor(out_features))
            self.rho_bias = Parameter(torch.Tensor(out_features))
            if prior_type == 'spike-and-slab':
                self.log_alpha_bias = Parameter(torch.Tensor(out_features)) # Spike-and-Slab prior
            elif prior_type == 'horseshoe':
                self.l_loc_bias = Parameter(torch.Tensor(out_features))
                self.l_scale_bias = Parameter(torch.Tensor(out_features))
                self.t_loc_bias = Parameter(torch.Tensor(1))
                self.t_scale_bias = Parameter(torch.Tensor(1))
                
            self.register_buffer(
                'eps_bias',
                torch.Tensor(out_features),
                persistent=False)
            self.register_buffer(
                'prior_bias_mu',
                torch.Tensor(out_features),
                persistent=False)
            self.register_buffer('prior_bias_sigma',
                                 torch.Tensor(out_features),
                                 persistent=False)
        else:
            if prior_type == 'spike-and-slab':
                self.register_parameter('log_alpha_bias', None) # Spike-and-Slab prior
            elif prior_type == 'horseshoe':
                self.register_parameter('l_loc_bias', None)
                self.register_parameter('l_scale_bias', None)
                self.register_parameter('t_loc_bias', None)
                self.register_parameter('t_scale_bias', None)
                
            self.register_buffer('prior_bias_mu', None, persistent=False)
            self.register_buffer('prior_bias_sigma', None, persistent=False)
            self.register_parameter('mu_bias', None)
            self.register_parameter('rho_bias', None)
            self.register_buffer('eps_bias', None, persistent=False)

        self.init_parameters()
        self.quant_prepare=False
    
    def prepare(self):
        self.qint_quant = nn.ModuleList([torch.quantization.QuantStub(
                                         QConfig(weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric), activation=MinMaxObserver.with_args(dtype=torch.qint8,qscheme=torch.per_tensor_symmetric))) for _ in range(5)])
        self.quint_quant = nn.ModuleList([torch.quantization.QuantStub(
                                         QConfig(weight=MinMaxObserver.with_args(dtype=torch.quint8), activation=MinMaxObserver.with_args(dtype=torch.quint8))) for _ in range(2)])
        self.dequant = torch.quantization.DeQuantStub()
        self.quant_prepare=True

    def init_parameters(self):
        self.prior_weight_mu.fill_(self.prior_mean)
        self.prior_weight_sigma.fill_(self.prior_variance)
        if self.prior_type == 'spike-and-slab':
            self.log_alpha_weight.data.normal_(mean=-3.0, std=0.1)  # Spike-and-Slab prior
        elif self.prior_type == 'horseshoe':
            # LogNormal 분포의 파라미터 초기화 (작은 양수 값에서 시작하도록)
            self.l_loc_weight.data.normal_(mean=-3.0, std=0.1)
            self.l_scale_weight.data.normal_(mean=-3.0, std=0.1)
            self.t_loc_weight.data.normal_(mean=-3.0, std=0.1)
            self.t_scale_weight.data.normal_(mean=-3.0, std=0.1)
            
        self.mu_weight.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
        self.rho_weight.data.normal_(mean=self.posterior_rho_init[0], std=0.1)
        if self.mu_bias is not None:
            if self.prior_type == 'spike-and-slab':
                self.log_alpha_bias.data.normal_(mean=-3.0, std=0.1) # Spike-and-Slab prior
            elif self.prior_type == 'horseshoe':
                self.l_loc_bias.data.normal_(mean=-3.0, std=0.1)
                self.l_scale_bias.data.normal_(mean=-3.0, std=0.1)
                self.t_loc_bias.data.normal_(mean=-3.0, std=0.1)
                self.t_scale_bias.data.normal_(mean=-3.0, std=0.1)
                
            self.prior_bias_mu.fill_(self.prior_mean)
            self.prior_bias_sigma.fill_(self.prior_variance)
            self.mu_bias.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init[0],
                                       std=0.1)

    def kl_loss(self):
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        
        if self.prior_type == 'horseshoe':
            kl = self.kl_div_horseshoe(
                mu_q = self.mu_weight,
                sigma_q = sigma_weight,
                l_loc = self.l_loc_weight,
                l_scale = self.l_scale_weight,
                t_loc = self.t_loc_weight,
                t_scale = self.t_scale_weight
            )
            if self.mu_bias is not None:
                sigma_bias = torch.log1p(torch.exp(self.rho_bias))
                kl += self.kl_div_horseshoe(
                    mu_q = self.mu_bias,
                    sigma_q = sigma_bias,
                    l_loc = self.l_loc_bias,
                    l_scale = self.l_scale_bias,
                    t_loc = self.t_loc_bias,
                    t_scale = self.t_scale_bias
                )
                
        elif self.prior_type == 'spike-and-slab':
            kl = self.kl_div_spike_and_slab(
                mu_q = self.mu_weight,
                sigma_q = sigma_weight,
                log_alpha = self.log_alpha_weight,
                mu_p = self.prior_weight_mu,
                sigma_p = self.prior_weight_sigma,
                pi_p = self.spike_and_slab_pi
            )
            if self.mu_bias is not None:
                sigma_bias = torch.log1p(torch.exp(self.rho_bias))
                kl += self.kl_div_spike_and_slab(
                    mu_q = self.mu_bias,
                    sigma_q = sigma_bias,
                    log_alpha = self.log_alpha_bias,
                    mu_p = self.prior_bias_mu,
                    sigma_p = self.prior_bias_sigma,
                    pi_p = self.spike_and_slab_pi
                )
        else:
            kl = self.kl_div(
                self.mu_weight,
                sigma_weight,
                self.prior_weight_mu,
                self.prior_weight_sigma,
                prior_type=self.prior_type)
            if self.mu_bias is not None:
                sigma_bias = torch.log1p(torch.exp(self.rho_bias))
                kl += self.kl_div(self.mu_bias, sigma_bias,
                                self.prior_bias_mu, self.prior_bias_sigma, prior_type=self.prior_type)
        return kl

    def forward(self, input, return_kl=True):
        if self.dnn_to_bnn_flag:
            return_kl = False
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        eps_weight = self.eps_weight.data.normal_()
        tmp_result = sigma_weight * eps_weight
        weight = self.mu_weight + tmp_result


        if return_kl:
            
            if self.prior_type == 'horseshoe':
                kl_weight = self.kl_div_horseshoe(
                    mu_q = self.mu_weight,
                    sigma_q = sigma_weight,
                    l_loc = self.l_loc_weight,
                    l_scale = self.l_scale_weight,
                    t_loc = self.t_loc_weight,
                    t_scale = self.t_scale_weight
                )
                
            elif self.prior_type == 'spike-and-slab':
                kl_weight = self.kl_div_spike_and_slab(
                    mu_q = self.mu_weight,
                    sigma_q = sigma_weight,
                    log_alpha = self.log_alpha_weight,
                    mu_p = self.prior_weight_mu,
                    sigma_p = self.prior_weight_sigma,
                    pi_p = self.spike_and_slab_pi
                )
            else:
                kl_weight = self.kl_div(self.mu_weight, sigma_weight,
                                    self.prior_weight_mu, self.prior_weight_sigma, prior_type = self.prior_type)
        bias = None

        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            bias = self.mu_bias + (sigma_bias * self.eps_bias.data.normal_())
            if return_kl:
                if self.prior_type == 'horseshoe':
                    kl_bias = self.kl_div_horseshoe(
                        mu_q = self.mu_bias,
                        sigma_q = sigma_bias,
                        l_loc = self.l_loc_bias,
                        l_scale = self.l_scale_bias,
                        t_loc = self.t_loc_bias,
                        t_scale = self.t_scale_bias
                    )
                    
                elif self.prior_type == 'spike-and-slab':
                    kl_bias = self.kl_div_spike_and_slab(
                        mu_q = self.mu_bias,
                        sigma_q = sigma_bias,
                        log_alpha = self.log_alpha_bias,
                        mu_p = self.prior_bias_mu,
                        sigma_p = self.prior_bias_sigma,
                        pi_p = self.spike_and_slab_pi
                    )
                else:
                    kl_bias = self.kl_div(self.mu_bias, sigma_bias, self.prior_bias_mu,
                                      self.prior_bias_sigma, prior_type = self.prior_type)

        out = F.linear(input, weight, bias)

        if self.quant_prepare:
            # quint8 quantstub
            input = self.quint_quant[0](input) # input
            out = self.quint_quant[1](out) # output

            # qint8 quantstub
            sigma_weight = self.qint_quant[0](sigma_weight) # weight
            mu_weight = self.qint_quant[1](self.mu_weight) # weight
            eps_weight = self.qint_quant[2](eps_weight) # random variable
            tmp_result =self.qint_quant[3](tmp_result) # multiply activation
            weight = self.qint_quant[4](weight) # add activatation


        if return_kl:
            if self.mu_bias is not None:
                kl = kl_weight + kl_bias
            else:
                kl = kl_weight

            return out, kl

        return out
