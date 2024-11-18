# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py


import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np


class SimpleStitchMoE(nn.Module):
    def __init__(self, input_size, output_size, stitch_layer_classes, noisy_gating=True) -> None:
        super(SimpleStitchMoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = len(stitch_layer_classes)
        self.output_size = output_size
        self.input_size = input_size
        self.experts = nn.ModuleList([stitch_layer_classes[i](self.input_size, self.output_size) for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, self.num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, self.num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        print(f"----- [MOE] Simple MOE SITCHING LAYER num experts: {self.num_experts} w_gate size: {self.w_gate.data.size()} noise : {self.noisy_gating}")
        
    def init_stitch_weights_bias(self, weight, bias, layer_id=None):
        if layer_id is None:
            for exp in self.experts:
                exp.init_stitch_weights_bias(weight, bias)
        else:
            print(f"Initializing Expert {layer_id}")
            self.experts[layer_id].init_stitch_weights_bias(weight, bias)
            
    def forward(self, x, noise_epsilon=1e-2):
        clean_logits = torch.einsum("bhw,we->bhe",x, self.w_gate)
        if self.noisy_gating and self.training:
            raw_noise_stddev = torch.einsum("bhw,we->bhe",x, self.w_noise)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        
        gates = nn.functional.softmax(logits, dim=2)
        exp_outputs = torch.stack([exp(x) for exp in self.experts],dim=-1)
        return torch.einsum("bhwe,bhe->bhw",exp_outputs,gates)
        
         