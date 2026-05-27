import os
import sys
import math
import time
import argparse
import threading
import itertools
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.set_float32_matmul_precision('high')


class TokenMemmapDataset(torch.utils.data.Dataset):
    def __init__(self, bin_path, block_size, token_start, token_end, samples_per_epoch=None, random_sampling=True):
        self.tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.token_start = int(token_start)
        self.token_end = int(token_end)
        self.max_start = self.token_end - self.token_start - self.block_size - 1
        if self.max_start <= 0:
            raise ValueError("Memmap split is too small for the requested block_size.")
        self.random_sampling = random_sampling
        self.samples_per_epoch = int(samples_per_epoch) if samples_per_epoch else self.max_start

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        if self.random_sampling:
            local_start = np.random.randint(0, self.max_start)
        else:
            local_start = idx % self.max_start
        start = self.token_start + local_start
        x = np.array(self.tokens[start:start + self.block_size], dtype=np.int64)
        y = np.int64(self.tokens[start + self.block_size])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

# ─── SECTION 1: BULLETPROOF DOMAIN-GUARDED PRIMITIVES ────────────────────────
def safe_divide(x1, x2, eps=1e-4):
    """Guards against division by zero and suppresses extreme vector scaling."""
    sign = torch.sign(x2)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    denom = torch.where(torch.abs(x2) < eps, sign * eps, x2)
    return x1 / denom

def safe_log(x, eps=1e-5):
    """Universal log(|x|) that natively avoids zero and negative domains."""
    return torch.log(torch.clamp(torch.abs(x), min=eps))

def safe_sqrt(x, eps=1e-5):
    """Fixes the PyTorch 0.0 gradient trap by ensuring finite derivatives."""
    return torch.sqrt(torch.clamp(x, min=eps))

def safe_arcsin(x, border_eps=1e-4):
    return torch.arcsin(torch.clamp(x, min=-1.0 + border_eps, max=1.0 - border_eps))

PRIM_LIST = [
    # Pure (parameter-free)
    'identity', 'pure_add', 'pure_sub', 'gated_mul', 'pure_div',
    # Unary math
    'sin', 'cos', 'tanh', 'arctan', 'arcsin', 'log', 'sqrt',
    # Affine-wrapped
    'affine_tanh', 'affine_sqrt', 'affine_cos', 'affine_log', 'affine_sin',
    'affine_relu', 'affine_gelu',
    # Structural
    'gated_affine', 'norm_affine', 'fourier_mix', 'soft_attention', 'shift_mix',
]
NUM_PRIMITIVES = len(PRIM_LIST)

# Registry for fused unary-affine primitives.
# To add one: (1) add the name to PRIM_LIST above, (2) append here.
# To remove one: remove from PRIM_LIST only — the has() guard skips it automatically.
# Slot indices are fixed so checkpoints stay compatible across removals.
UNARY_AFFINE_PRIMS = [
    ('affine_tanh', torch.tanh),   # slot 0
    ('affine_sqrt', safe_sqrt),    # slot 1
    ('affine_cos',  torch.cos),    # slot 2
    ('affine_log',  safe_log),     # slot 3
    ('affine_sin',  torch.sin),    # slot 4
]


# ─── SECTIONS 2 & 3: MODEL WITH BINARY ROUTING AND VALUE SAFETY RAILS ─────────
class SymbolicWorldModel(nn.Module):
    def __init__(self, block_size=32, num_layers=3, breadth=4, embed_dim=16, vocab_size=65, num_projection_slots=16):
        super().__init__()
        self.NUM_PROJECTION_SLOTS = num_projection_slots
        self.T = block_size
        self.L = num_layers
        self.B = breadth
        self.D = embed_dim
        self.vocab_size = vocab_size
        self.active_primitives = set(PRIM_LIST)
        self.total_slots = self.NUM_PROJECTION_SLOTS + (self.L * self.B)

        self.token_embeddings = nn.Embedding(self.vocab_size, self.D)
        self.pos_embeddings = nn.Parameter(torch.zeros(1, block_size, self.D))
        # v3 FIX (F): boost positional signal std so positions are not drowned
        # out by token embeddings inside summed routing bags.
        nn.init.normal_(self.pos_embeddings, std=0.5)

        # ─── LAYER 0 SPATIAL ROUTING & PRIMITIVE LOGITS ───────────────────────
        self.spatial_routing_logits = nn.Parameter(torch.zeros(self.NUM_PROJECTION_SLOTS, 2, block_size))
        self.spatial_primitive_logits = nn.Parameter(torch.zeros(self.NUM_PROJECTION_SLOTS, NUM_PRIMITIVES))

        # Inductive Bias focal initialization:
        with torch.no_grad():
            for b in range(self.NUM_PROJECTION_SLOTS):
                center = b * (block_size / self.NUM_PROJECTION_SLOTS)
                sigma = max(1.0, block_size / (2.0 * self.NUM_PROJECTION_SLOTS))
                for t in range(block_size):
                    dist = float(t - center)
                    penalty = - (dist ** 2) / (2.0 * (sigma ** 2))
                    self.spatial_routing_logits[b, :, t] = penalty

        # ─── DEEP ROUTING PARAMETERS (POOL SIZE = NUM_PROJECTION_SLOTS + l * B)
        self.primitive_logits = nn.Parameter(torch.zeros(self.L, self.B, NUM_PRIMITIVES))

        self.routing_logits = nn.ParameterList()
        for l in range(self.L):
            history_pool_size = self.NUM_PROJECTION_SLOTS + (l * self.B)
            r_logits = nn.Parameter(torch.zeros(self.B, 2, history_pool_size, 2))
            self.routing_logits.append(r_logits)

        # ─── v4 NEW: PER-EDGE WEIGHTED ROUTING ───────────────────────────────
        self.routing_edge_weights = nn.ParameterList()
        for l in range(self.L):
            history_pool_size = self.NUM_PROJECTION_SLOTS + (l * self.B)
            e_w = nn.Parameter(torch.randn(self.B, 2, history_pool_size) * 0.1)
            self.routing_edge_weights.append(e_w)
        # Per-channel routing bias added after the weighted sum.
        self.routing_biases = nn.Parameter(torch.zeros(self.L, self.B, 2, self.D))

        # ─── v3 DEEP PARAMETERS ──────────────────────────────────────────────
        def _w():
            return nn.Parameter(torch.randn(self.L, self.B, self.D, self.D) * (1.0 / math.sqrt(self.D)))
        def _b():
            return nn.Parameter(torch.zeros(self.L, self.B, self.D))

        has = self.active_primitives.__contains__
        if has('affine_relu') or has('affine_gelu'):
            self.affine_weights = _w()
            self.affine_biases = _b()

        # Fused unary-affine primitives (slot layout defined in UNARY_AFFINE_PRIMS)
        _NUM_UNARY = len(UNARY_AFFINE_PRIMS)
        if any(has(n) for n, _ in UNARY_AFFINE_PRIMS):
            self.unary_affine_weights = nn.Parameter(torch.randn(self.L, _NUM_UNARY, self.B, self.D, self.D) * (1.0 / math.sqrt(self.D)))
            self.unary_affine_biases  = nn.Parameter(torch.zeros(self.L, _NUM_UNARY, self.B, self.D))

        # New structural primitives
        if has('gated_affine'):
            self.gate1_weights, self.gate1_biases = _w(), _b()
            self.gate2_weights, self.gate2_biases = _w(), _b()
        if has('norm_affine'):
            self.norm_weights,  self.norm_biases  = _w(), _b()
        if has('shift_mix'):
            self.shift_weights = _w()
        if has('fourier_mix'):
            self.f1_weights, self.f1_biases = _w(), _b()
            self.f2_weights, self.f2_biases = _w(), _b()
        if has('soft_attention'):
            self.q_weights = _w()
            self.k_weights = _w()
            self.v_weights = _w()

        # ─── LAYER 0 PARAMETERIZED ROUTING WEIGHTS & BIASES ──────────────────
        def _w_sp():
            return nn.Parameter(torch.randn(self.NUM_PROJECTION_SLOTS, self.D, self.D) * (1.0 / math.sqrt(self.D)))
        def _b_sp():
            return nn.Parameter(torch.zeros(self.NUM_PROJECTION_SLOTS, self.D))

        if has('affine_relu') or has('affine_gelu'):
            self.spatial_affine_weights, self.spatial_affine_biases = _w_sp(), _b_sp()
        # Fused unary-affine spatial (slot layout defined in UNARY_AFFINE_PRIMS)
        if any(has(n) for n, _ in UNARY_AFFINE_PRIMS):
            self.spatial_unary_affine_weights = nn.Parameter(torch.randn(_NUM_UNARY, self.NUM_PROJECTION_SLOTS, self.D, self.D) * (1.0 / math.sqrt(self.D)))
            self.spatial_unary_affine_biases  = nn.Parameter(torch.zeros(_NUM_UNARY, self.NUM_PROJECTION_SLOTS, self.D))
        if has('gated_affine'):
            self.spatial_gate1_weights, self.spatial_gate1_biases = _w_sp(), _b_sp()
            self.spatial_gate2_weights, self.spatial_gate2_biases = _w_sp(), _b_sp()
        if has('norm_affine'):
            self.spatial_norm_weights, self.spatial_norm_biases = _w_sp(), _b_sp()
        if has('shift_mix'):
            self.spatial_shift_weights = _w_sp()
        if has('fourier_mix'):
            self.spatial_f1_weights, self.spatial_f1_biases = _w_sp(), _b_sp()
            self.spatial_f2_weights, self.spatial_f2_biases = _w_sp(), _b_sp()
        if has('soft_attention'):
            self.spatial_q_weights = _w_sp()
            self.spatial_k_weights = _w_sp()
            self.spatial_v_weights = _w_sp()

    def _stack_active_primitives(self, outputs_by_name):
        return torch.stack([outputs_by_name[name] for name in PRIM_LIST], dim=2)

    def _spatial_primitive_stack(self, x1, x2):
        has = self.active_primitives.__contains__
        outputs = {}

        if has('identity'): outputs['identity'] = x1
        if has('pure_add'): outputs['pure_add'] = x1 + x2
        if has('pure_sub'): outputs['pure_sub'] = x1 - x2
        if has('gated_mul'): outputs['gated_mul'] = x1 * x2
        if has('pure_div'): outputs['pure_div'] = safe_divide(x1, x2)
        if has('sin'): outputs['sin'] = torch.sin(x1)
        if has('cos'): outputs['cos'] = torch.cos(x1)
        if has('tanh'): outputs['tanh'] = torch.tanh(x1)
        if has('arctan'): outputs['arctan'] = torch.arctan(x1)
        if has('arcsin'): outputs['arcsin'] = safe_arcsin(x1)
        if has('log'): outputs['log'] = safe_log(x1)
        if has('sqrt'): outputs['sqrt'] = safe_sqrt(x1)

        if has('affine_relu') or has('affine_gelu'):
            affine_projected = torch.einsum('nbd,bde->nbe', x1, self.spatial_affine_weights) + self.spatial_affine_biases.unsqueeze(0)
            if has('affine_relu'): outputs['affine_relu'] = F.relu(affine_projected)
            if has('affine_gelu'): outputs['affine_gelu'] = F.gelu(affine_projected)
        if any(has(n) for n, _ in UNARY_AFFINE_PRIMS):
            # (N, P, SLOTS, D) — one batched projection for all unary-affine primitives
            all_unary = torch.einsum('nbd,pbde->npbe', x1, self.spatial_unary_affine_weights) + self.spatial_unary_affine_biases.unsqueeze(0)
            for p_idx, (prim_name, fn) in enumerate(UNARY_AFFINE_PRIMS):
                if has(prim_name):
                    outputs[prim_name] = fn(all_unary[:, p_idx])
        if has('gated_affine'):
            gate1_projected = torch.einsum('nbd,bde->nbe', x1, self.spatial_gate1_weights) + self.spatial_gate1_biases.unsqueeze(0)
            gate2_projected = torch.einsum('nbd,bde->nbe', x2, self.spatial_gate2_weights) + self.spatial_gate2_biases.unsqueeze(0)
            outputs['gated_affine'] = torch.sigmoid(gate1_projected) * gate2_projected
        if has('norm_affine'):
            normed_x1 = F.layer_norm(x1, (self.D,))
            outputs['norm_affine'] = torch.einsum('nbd,bde->nbe', normed_x1, self.spatial_norm_weights) + self.spatial_norm_biases.unsqueeze(0)
        if has('shift_mix'):
            outputs['shift_mix'] = torch.einsum('nbd,bde->nbe', x1, self.spatial_shift_weights) + x2
        if has('fourier_mix'):
            f1_projected = torch.einsum('nbd,bde->nbe', x1, self.spatial_f1_weights) + self.spatial_f1_biases.unsqueeze(0)
            f2_projected = torch.einsum('nbd,bde->nbe', x2, self.spatial_f2_weights) + self.spatial_f2_biases.unsqueeze(0)
            outputs['fourier_mix'] = torch.sin(f1_projected) * torch.cos(f2_projected)
        if has('soft_attention'):
            q_proj = torch.einsum('nbd,bde->nbe', x1, self.spatial_q_weights)
            k_proj = torch.einsum('nbd,bde->nbe', x2, self.spatial_k_weights)
            v_proj = torch.einsum('nbd,bde->nbe', x2, self.spatial_v_weights)
            score = torch.sum(q_proj * k_proj, dim=-1) / math.sqrt(self.D)
            weight = torch.sigmoid(score).unsqueeze(-1)
            outputs['soft_attention'] = weight * v_proj + (1.0 - weight) * x1

        return self._stack_active_primitives(outputs)

    def _deep_primitive_stack(self, l, x1, x2):
        has = self.active_primitives.__contains__
        outputs = {}

        if has('identity'): outputs['identity'] = x1
        if has('pure_add'): outputs['pure_add'] = x1 + x2
        if has('pure_sub'): outputs['pure_sub'] = x1 - x2
        if has('gated_mul'): outputs['gated_mul'] = x1 * x2
        if has('pure_div'): outputs['pure_div'] = safe_divide(x1, x2)
        if has('sin'): outputs['sin'] = torch.sin(x1)
        if has('cos'): outputs['cos'] = torch.cos(x1)
        if has('tanh'): outputs['tanh'] = torch.tanh(x1)
        if has('arctan'): outputs['arctan'] = torch.arctan(x1)
        if has('arcsin'): outputs['arcsin'] = safe_arcsin(x1)
        if has('log'): outputs['log'] = safe_log(x1)
        if has('sqrt'): outputs['sqrt'] = safe_sqrt(x1)

        if has('affine_relu') or has('affine_gelu'):
            affine_projected = torch.einsum('nbd,bde->nbe', x1, self.affine_weights[l]) + self.affine_biases[l].unsqueeze(0)
            if has('affine_relu'): outputs['affine_relu'] = F.relu(affine_projected)
            if has('affine_gelu'): outputs['affine_gelu'] = F.gelu(affine_projected)
        if any(has(n) for n, _ in UNARY_AFFINE_PRIMS):
            # (N, P, B, D) — one batched projection for all unary-affine primitives
            all_unary = torch.einsum('nbd,pbde->npbe', x1, self.unary_affine_weights[l]) + self.unary_affine_biases[l].unsqueeze(0)
            for p_idx, (prim_name, fn) in enumerate(UNARY_AFFINE_PRIMS):
                if has(prim_name):
                    outputs[prim_name] = fn(all_unary[:, p_idx])
        if has('gated_affine'):
            gate1_projected = torch.einsum('nbd,bde->nbe', x1, self.gate1_weights[l]) + self.gate1_biases[l].unsqueeze(0)
            gate2_projected = torch.einsum('nbd,bde->nbe', x2, self.gate2_weights[l]) + self.gate2_biases[l].unsqueeze(0)
            outputs['gated_affine'] = torch.sigmoid(gate1_projected) * gate2_projected
        if has('norm_affine'):
            normed_x1 = F.layer_norm(x1, (self.D,))
            outputs['norm_affine'] = torch.einsum('nbd,bde->nbe', normed_x1, self.norm_weights[l]) + self.norm_biases[l].unsqueeze(0)
        if has('shift_mix'):
            outputs['shift_mix'] = torch.einsum('nbd,bde->nbe', x1, self.shift_weights[l]) + x2
        if has('fourier_mix'):
            f1_projected = torch.einsum('nbd,bde->nbe', x1, self.f1_weights[l]) + self.f1_biases[l].unsqueeze(0)
            f2_projected = torch.einsum('nbd,bde->nbe', x2, self.f2_weights[l]) + self.f2_biases[l].unsqueeze(0)
            outputs['fourier_mix'] = torch.sin(f1_projected) * torch.cos(f2_projected)
        if has('soft_attention'):
            q_proj = torch.einsum('nbd,bde->nbe', x1, self.q_weights[l])
            k_proj = torch.einsum('nbd,bde->nbe', x2, self.k_weights[l])
            v_proj = torch.einsum('nbd,bde->nbe', x2, self.v_weights[l])
            score = torch.sum(q_proj * k_proj, dim=-1) / math.sqrt(self.D)
            weight = torch.sigmoid(score).unsqueeze(-1)
            outputs['soft_attention'] = weight * v_proj + (1.0 - weight) * x1

        return self._stack_active_primitives(outputs)

    def forward(self, token_ids, temperature=1.0):
        N, T_long = token_ids.shape
        device = token_ids.device

        # Embeddings & position encoding
        long_embeds = self.token_embeddings(token_ids) + self.pos_embeddings[:, :T_long, :]

        # Slice spatial routing logits dynamically along the time axis
        spatial_logits_sliced = self.spatial_routing_logits[:, :, :T_long]

        # Force Gumbel-Softmax to normalize across the time axis (dim=-1)
        spatial_mask = F.gumbel_softmax(spatial_logits_sliced, tau=temperature, hard=False, dim=-1) # (NUM_PROJECTION_SLOTS, 2, T_long)

        # Spatial Compression via explicit channel-wise reduction
        x1_raw = torch.einsum('bt,ntd->nbd', spatial_mask[:, 0, :], long_embeds) # Channel 0 mapping
        x2_raw = torch.einsum('bt,ntd->nbd', spatial_mask[:, 1, :], long_embeds) # Channel 1 mapping

        # Pre-norm Layer 0
        x1 = F.layer_norm(x1_raw, (self.D,))
        x2 = F.layer_norm(x2_raw, (self.D,))

        spatial_stacked = self._spatial_primitive_stack(x1, x2)

        # HARD SAFETY BOUNDARY
        spatial_stacked = torch.clamp(spatial_stacked, min=-20.0, max=20.0)

        spatial_prim_weights = F.gumbel_softmax(self.spatial_primitive_logits, tau=temperature, hard=False, dim=-1)
        compressed_slots = torch.sum(spatial_stacked * spatial_prim_weights.unsqueeze(0).unsqueeze(-1), dim=2)

        # Allocate Master Buffer and Seed it
        buffer = torch.empty((N, self.total_slots, self.D), device=device, dtype=torch.float32).zero_()
        buffer[:, :self.NUM_PROJECTION_SLOTS, :] = compressed_slots

        # Deep layer loop
        for l in range(self.L):
            history_cutoff = self.NUM_PROJECTION_SLOTS + (l * self.B)
            layer_start = history_cutoff
            layer_end = layer_start + self.B
            past_history = buffer[:, :history_cutoff, :]

            # v4: PER-EDGE WEIGHTED ROUTING
            route_gumbel = F.gumbel_softmax(self.routing_logits[l], tau=temperature, hard=False, dim=-1)
            route_mask = route_gumbel[:, :, :, 1]  # (B, 2, H), binary 0/1

            # Per-edge weights
            effective_weights = route_mask * self.routing_edge_weights[l]  # (B, 2, H)

            # Weighted sum over history
            layer_inputs = torch.einsum('bkh,nhd->nbkd', effective_weights, past_history)  # (N, B, 2, D)

            # Per-channel routing bias
            layer_inputs = layer_inputs + self.routing_biases[l].unsqueeze(0)  # (N, B, 2, D)

            x1_raw_deep = layer_inputs[:, :, 0, :]
            x2_raw_deep = layer_inputs[:, :, 1, :]

            # Pre-norm
            x1_deep = F.layer_norm(x1_raw_deep, (self.D,))
            x2_deep = F.layer_norm(x2_raw_deep, (self.D,))

            stacked_outputs = self._deep_primitive_stack(l, x1_deep, x2_deep)

            # HARD SAFETY BOUNDARY
            stacked_outputs = torch.clamp(stacked_outputs, min=-20.0, max=20.0)

            prim_weights = F.gumbel_softmax(self.primitive_logits[l], tau=temperature, hard=False, dim=-1)
            layer_outputs = torch.sum(stacked_outputs * prim_weights.unsqueeze(0).unsqueeze(-1), dim=2)

            # Residual connections
            if l > 0:
                prev_layer_start = self.NUM_PROJECTION_SLOTS + (l - 1) * self.B
                prev_layer_end = prev_layer_start + self.B
                prev_layer_output = buffer[:, prev_layer_start:prev_layer_end, :]
                layer_outputs = layer_outputs + prev_layer_output

            buffer[:, layer_start:layer_end, :] = layer_outputs

        return buffer


# ─── SECTION 4: EVALUATION AND UNIFIED ANNEALING TRAINING LOOP ───────────────
def compute_structural_entropy(model):
    total_entropy = 0.0
    total_elements = 0
    for l in range(model.L):
        probs = torch.softmax(model.routing_logits[l], dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        total_entropy += torch.sum(entropy)
        total_elements += entropy.numel()
    return total_entropy / max(1, total_elements)


# ─── EXTRACTION ENGINE ────────────────────────────────────────────────────────
def symbolic_slot_ref(model, h):
    if h < model.NUM_PROJECTION_SLOTS:
        return f"t_{h}"
    h_rem = h - model.NUM_PROJECTION_SLOTS
    return f"N_{h_rem // model.B}_{h_rem % model.B}"


def extract_symbolic_equations_hard(model):
    model.eval()
    slot_expressions = []

    with torch.no_grad():
        # 1. Parse Layer 0 spatial routing masks and primitive selections
        spatial_route_choices = torch.argmax(model.spatial_routing_logits, dim=-1) # (NUM_PROJECTION_SLOTS, 2)
        spatial_primitive_choices = torch.argmax(model.spatial_primitive_logits, dim=-1) # (NUM_PROJECTION_SLOTS,)

        for b in range(model.NUM_PROJECTION_SLOTS):
            def build_spatial_arg(channel):
                t_idx = spatial_route_choices[b, channel].item()
                return f"raw_t_{t_idx}"

            arg1_str = f"LN({build_spatial_arg(0)})"
            arg2_str = f"LN({build_spatial_arg(1)})"

            chosen_prim_name = PRIM_LIST[spatial_primitive_choices[b]]

            if chosen_prim_name == 'identity': node_expr = arg1_str
            elif chosen_prim_name == 'pure_add': node_expr = f"({arg1_str} + {arg2_str})"
            elif chosen_prim_name == 'pure_sub': node_expr = f"({arg1_str} - {arg2_str})"
            elif chosen_prim_name == 'gated_mul': node_expr = f"({arg1_str} * {arg2_str})"
            elif chosen_prim_name == 'pure_div': node_expr = f"({arg1_str} / {arg2_str})"
            elif chosen_prim_name in ['sin', 'cos', 'tanh', 'arctan', 'arcsin', 'log', 'sqrt']:
                node_expr = f"{chosen_prim_name}({arg1_str})"
            elif chosen_prim_name == 'affine_relu': node_expr = f"ReLU(W_sp_{b} * {arg1_str} + b_sp_{b})"
            elif chosen_prim_name == 'affine_gelu': node_expr = f"GELU(W_sp_{b} * {arg1_str} + b_sp_{b})"
            elif chosen_prim_name == 'affine_sin': node_expr = f"sin(W_sp_sin_{b} * {arg1_str} + b_sp_sin_{b})"
            elif chosen_prim_name == 'affine_cos': node_expr = f"cos(W_sp_cos_{b} * {arg1_str} + b_sp_cos_{b})"
            elif chosen_prim_name == 'affine_tanh': node_expr = f"tanh(W_sp_tanh_{b} * {arg1_str} + b_sp_tanh_{b})"
            elif chosen_prim_name == 'affine_log': node_expr = f"log(W_sp_log_{b} * {arg1_str} + b_sp_log_{b})"
            elif chosen_prim_name == 'affine_sqrt': node_expr = f"sqrt(W_sp_sqrt_{b} * {arg1_str} + b_sp_sqrt_{b})"
            elif chosen_prim_name == 'gated_affine': node_expr = f"sigmoid(W_sp_g1_{b} * {arg1_str} + b_sp_g1_{b}) * (W_sp_g2_{b} * {arg2_str} + b_sp_g2_{b})"
            elif chosen_prim_name == 'norm_affine': node_expr = f"(W_sp_norm_{b} * LN({arg1_str}) + b_sp_norm_{b})"
            elif chosen_prim_name == 'shift_mix': node_expr = f"(W_sp_shift_{b} * {arg1_str} + {arg2_str})"
            elif chosen_prim_name == 'fourier_mix': node_expr = f"sin(W_sp_f1_{b} * {arg1_str} + b_sp_f1_{b}) * cos(W_sp_f2_{b} * {arg2_str} + b_sp_f2_{b})"
            elif chosen_prim_name == 'soft_attention': node_expr = f"ATTN_sp_Q={b}({arg1_str}, {arg2_str})"
            else: node_expr = "0"

            slot_expressions.append(f"t_{b} = {node_expr}")

        # 2. Parse Deep hidden layer nodes referencing slots 0-31 as contextual base variables t_0 to t_31
        for l in range(model.L):
            history_cutoff = model.NUM_PROJECTION_SLOTS + (l * model.B)
            route_choices = torch.argmax(model.routing_logits[l], dim=-1)
            primitive_choices = torch.argmax(model.primitive_logits[l], dim=-1)
            edge_weights = model.routing_edge_weights[l]  # (B, 2, H)

            for b in range(model.B):
                def build_arg(channel):
                    comps = []
                    for h in range(history_cutoff):
                        if route_choices[b, channel, h] == 1:
                            w = edge_weights[b, channel, h].item()
                            comps.append(f"{w:.3g}*{symbolic_slot_ref(model, h)}")
                    comps.append(f"b_route_{l}_{b}_{channel}")
                    return "(" + " + ".join(comps) + ")"

                arg1_str = f"LN({build_arg(0)})"
                arg2_str = f"LN({build_arg(1)})"

                chosen_prim_name = PRIM_LIST[primitive_choices[b]]

                if chosen_prim_name == 'identity': node_expr = arg1_str
                elif chosen_prim_name == 'pure_add': node_expr = f"({arg1_str} + {arg2_str})"
                elif chosen_prim_name == 'pure_sub': node_expr = f"({arg1_str} - {arg2_str})"
                elif chosen_prim_name == 'gated_mul': node_expr = f"({arg1_str} * {arg2_str})"
                elif chosen_prim_name == 'pure_div': node_expr = f"({arg1_str} / {arg2_str})"
                elif chosen_prim_name in ['sin', 'cos', 'tanh', 'arctan', 'arcsin', 'log', 'sqrt']:
                    node_expr = f"{chosen_prim_name}({arg1_str})"
                elif chosen_prim_name == 'affine_relu': node_expr = f"ReLU(W_{l}_{b} * {arg1_str} + b_{l}_{b})"
                elif chosen_prim_name == 'affine_gelu': node_expr = f"GELU(W_{l}_{b} * {arg1_str} + b_{l}_{b})"
                elif chosen_prim_name == 'affine_sin': node_expr = f"sin(W_sin_{l}_{b} * {arg1_str} + b_sin_{l}_{b})"
                elif chosen_prim_name == 'affine_cos': node_expr = f"cos(W_cos_{l}_{b} * {arg1_str} + b_cos_{l}_{b})"
                elif chosen_prim_name == 'affine_tanh': node_expr = f"tanh(W_tanh_{l}_{b} * {arg1_str} + b_tanh_{l}_{b})"
                elif chosen_prim_name == 'affine_log': node_expr = f"log(W_log_{l}_{b} * {arg1_str} + b_log_{l}_{b})"
                elif chosen_prim_name == 'affine_sqrt': node_expr = f"sqrt(W_sqrt_{l}_{b} * {arg1_str} + b_sqrt_{l}_{b})"
                elif chosen_prim_name == 'gated_affine': node_expr = f"sigmoid(W_g1_{l}_{b} * {arg1_str} + b_g1_{l}_{b}) * (W_g2_{l}_{b} * {arg2_str} + b_g2_{l}_{b})"
                elif chosen_prim_name == 'norm_affine': node_expr = f"(W_norm_{l}_{b} * LN({arg1_str}) + b_norm_{l}_{b})"
                elif chosen_prim_name == 'shift_mix': node_expr = f"(W_shift_{l}_{b} * {arg1_str} + {arg2_str})"
                elif chosen_prim_name == 'fourier_mix': node_expr = f"sin(W_f1_{l}_{b} * {arg1_str} + b_f1_{l}_{b}) * cos(W_f2_{l}_{b} * {arg2_str} + b_f2_{l}_{b})"
                elif chosen_prim_name == 'soft_attention': node_expr = f"ATTN_Q={l}_{b}({arg1_str}, {arg2_str})"
                else: node_expr = "0"

                if l > 0:
                    prev_expr = symbolic_slot_ref(model, model.NUM_PROJECTION_SLOTS + (l - 1) * model.B + b)
                    node_expr = f"({node_expr} + {prev_expr})"

                slot_expressions.append(node_expr)

    return slot_expressions

def extract_symbolic_equations_soft(model, mixture_threshold=0.001):
    model.eval()
    slot_expressions = []

    with torch.no_grad():
        # 1. Spatial layer — soft routing over time positions, soft primitive mixture
        spatial_route_probs = torch.softmax(model.spatial_routing_logits, dim=-1)  # (SLOTS, 2, T)
        spatial_prim_probs_all = torch.softmax(model.spatial_primitive_logits, dim=-1)  # (SLOTS, P)

        for b in range(model.NUM_PROJECTION_SLOTS):
            def build_spatial_arg(channel, _b=b):
                probs = spatial_route_probs[_b, channel].cpu().numpy()
                sorted_t = sorted(range(len(probs)), key=lambda t: -probs[t])
                comps, cumsum = [], 0.0
                for t in sorted_t:
                    comps.append(f"{probs[t]:.2f}*tok_{t}")
                    cumsum += probs[t]
                    if cumsum >= 0.90:
                        break
                return "(" + " + ".join(comps) + ")"

            arg1_str = f"LN({build_spatial_arg(0)})"
            arg2_str = f"LN({build_spatial_arg(1)})"

            def render_spatial_primitive(prim_name):
                if prim_name == 'identity': return arg1_str
                elif prim_name == 'pure_add': return f"({arg1_str} + {arg2_str})"
                elif prim_name == 'pure_sub': return f"({arg1_str} - {arg2_str})"
                elif prim_name == 'gated_mul': return f"({arg1_str} * {arg2_str})"
                elif prim_name == 'pure_div': return f"({arg1_str} / {arg2_str})"
                elif prim_name in ['sin', 'cos', 'tanh', 'arctan', 'arcsin', 'log', 'sqrt']:
                    return f"{prim_name}({arg1_str})"
                elif prim_name == 'affine_relu': return f"ReLU(W_sp_{b} * {arg1_str} + b_sp_{b})"
                elif prim_name == 'affine_gelu': return f"GELU(W_sp_{b} * {arg1_str} + b_sp_{b})"
                elif prim_name == 'affine_sin': return f"sin(W_sp_sin_{b} * {arg1_str} + b_sp_sin_{b})"
                elif prim_name == 'affine_cos': return f"cos(W_sp_cos_{b} * {arg1_str} + b_sp_cos_{b})"
                elif prim_name == 'affine_tanh': return f"tanh(W_sp_tanh_{b} * {arg1_str} + b_sp_tanh_{b})"
                elif prim_name == 'affine_log': return f"log(W_sp_log_{b} * {arg1_str} + b_sp_log_{b})"
                elif prim_name == 'affine_sqrt': return f"sqrt(W_sp_sqrt_{b} * {arg1_str} + b_sp_sqrt_{b})"
                elif prim_name == 'gated_affine': return f"sigmoid(W_sp_g1_{b} * {arg1_str} + b_sp_g1_{b}) * (W_sp_g2_{b} * {arg2_str} + b_sp_g2_{b})"
                elif prim_name == 'norm_affine': return f"(W_sp_norm_{b} * LN({arg1_str}) + b_sp_norm_{b})"
                elif prim_name == 'shift_mix': return f"(W_sp_shift_{b} * {arg1_str} + {arg2_str})"
                elif prim_name == 'fourier_mix': return f"sin(W_sp_f1_{b} * {arg1_str} + b_sp_f1_{b}) * cos(W_sp_f2_{b} * {arg2_str} + b_sp_f2_{b})"
                elif prim_name == 'soft_attention': return f"ATTN_sp_Q={b}({arg1_str}, {arg2_str})"
                else: return "0"

            probs_b = spatial_prim_probs_all[b].cpu().numpy()
            kept_idxs = [i for i in range(NUM_PRIMITIVES) if probs_b[i] >= mixture_threshold]
            kept_idxs.sort(key=lambda i: -probs_b[i])

            if len(kept_idxs) == 0:
                top1 = int(probs_b.argmax())
                rendered = render_spatial_primitive(PRIM_LIST[top1])
                node_expr = f"<<uncommitted; top1={PRIM_LIST[top1]}@{probs_b[top1]*100:.1f}%>> {probs_b[top1]:.3g}*{rendered}"
            elif len(kept_idxs) == 1:
                only = kept_idxs[0]
                rendered = render_spatial_primitive(PRIM_LIST[only])
                if probs_b[only] >= 0.90:
                    node_expr = rendered
                else:
                    node_expr = f"{probs_b[only]:.3g}*{rendered}"
            else:
                parts = []
                for i in kept_idxs:
                    rendered = render_spatial_primitive(PRIM_LIST[i])
                    parts.append(f"{probs_b[i]:.3g}*{rendered}")
                node_expr = "(" + " + ".join(parts) + ")"

            slot_expressions.append(f"t_{b} = {node_expr}")

        # 2. Deep hidden layer nodes — soft routing probs × learned edge weights
        for l in range(model.L):
            history_cutoff = model.NUM_PROJECTION_SLOTS + (l * model.B)
            route_probs = torch.softmax(model.routing_logits[l], dim=-1)[:, :, :, 1]  # (B, 2, H)
            prim_probs_all = torch.softmax(model.primitive_logits[l], dim=-1)
            edge_weights = model.routing_edge_weights[l]  # (B, 2, H)

            for b in range(model.B):
                def build_arg(channel, _b=b, _l=l, _hc=history_cutoff):
                    effective = (route_probs[_b, channel] * edge_weights[_b, channel]).cpu().numpy()
                    total = float(abs(effective).sum()) + 1e-8
                    sorted_h = sorted(range(_hc), key=lambda h: -abs(effective[h]))
                    comps = []
                    for h in sorted_h:
                        if abs(effective[h]) / total < 0.02:
                            break
                        comps.append(f"{effective[h]:.3g}*{symbolic_slot_ref(model, h)}")
                    bias_val = model.routing_biases[_l, _b, channel].mean().item()
                    if abs(bias_val) > 1e-3:
                        comps.append(f"{bias_val:.3g}")
                    return "(" + " + ".join(comps) + ")" if comps else "(mixed)"

                arg1_str = f"LN({build_arg(0)})"
                arg2_str = f"LN({build_arg(1)})"

                def render_primitive(prim_name):
                    if prim_name == 'identity': return arg1_str
                    elif prim_name == 'pure_add': return f"({arg1_str} + {arg2_str})"
                    elif prim_name == 'pure_sub': return f"({arg1_str} - {arg2_str})"
                    elif prim_name == 'gated_mul': return f"({arg1_str} * {arg2_str})"
                    elif prim_name == 'pure_div': return f"({arg1_str} / {arg2_str})"
                    elif prim_name in ['sin', 'cos', 'tanh', 'arctan', 'arcsin', 'log', 'sqrt']:
                        return f"{prim_name}({arg1_str})"
                    elif prim_name == 'affine_relu': return f"ReLU(W_{l}_{b} * {arg1_str} + b_{l}_{b})"
                    elif prim_name == 'affine_gelu': return f"GELU(W_{l}_{b} * {arg1_str} + b_{l}_{b})"
                    elif prim_name == 'affine_sin': return f"sin(W_sin_{l}_{b} * {arg1_str} + b_sin_{l}_{b})"
                    elif prim_name == 'affine_cos': return f"cos(W_cos_{l}_{b} * {arg1_str} + b_cos_{l}_{b})"
                    elif prim_name == 'affine_tanh': return f"tanh(W_tanh_{l}_{b} * {arg1_str} + b_tanh_{l}_{b})"
                    elif prim_name == 'affine_log': return f"log(W_log_{l}_{b} * {arg1_str} + b_log_{l}_{b})"
                    elif prim_name == 'affine_sqrt': return f"sqrt(W_sqrt_{l}_{b} * {arg1_str} + b_sqrt_{l}_{b})"
                    elif prim_name == 'gated_affine': return f"sigmoid(W_g1_{l}_{b} * {arg1_str} + b_g1_{l}_{b}) * (W_g2_{l}_{b} * {arg2_str} + b_g2_{l}_{b})"
                    elif prim_name == 'norm_affine': return f"(W_norm_{l}_{b} * LN({arg1_str}) + b_norm_{l}_{b})"
                    elif prim_name == 'shift_mix': return f"(W_shift_{l}_{b} * {arg1_str} + {arg2_str})"
                    elif prim_name == 'fourier_mix': return f"sin(W_f1_{l}_{b} * {arg1_str} + b_f1_{l}_{b}) * cos(W_f2_{l}_{b} * {arg2_str} + b_f2_{l}_{b})"
                    elif prim_name == 'soft_attention': return f"ATTN_Q={l}_{b}({arg1_str}, {arg2_str})"
                    else: return "0"

                probs_b = prim_probs_all[b].cpu().numpy()
                kept_idxs = [i for i in range(NUM_PRIMITIVES) if probs_b[i] >= mixture_threshold]
                kept_idxs.sort(key=lambda i: -probs_b[i])

                if len(kept_idxs) == 0:
                    top1 = int(probs_b.argmax())
                    rendered = render_primitive(PRIM_LIST[top1])
                    node_expr = f"<<uncommitted; top1={PRIM_LIST[top1]}@{probs_b[top1]*100:.1f}%>> {probs_b[top1]:.3g}*{rendered}"
                elif len(kept_idxs) == 1:
                    only = kept_idxs[0]
                    rendered = render_primitive(PRIM_LIST[only])
                    if probs_b[only] >= 0.90:
                        node_expr = rendered
                    else:
                        node_expr = f"{probs_b[only]:.3g}*{rendered}"
                else:
                    parts = []
                    for i in kept_idxs:
                        rendered = render_primitive(PRIM_LIST[i])
                        parts.append(f"{probs_b[i]:.3g}*{rendered}")
                    node_expr = "(" + " + ".join(parts) + ")"

                if l > 0:
                    prev_expr = symbolic_slot_ref(model, model.NUM_PROJECTION_SLOTS + (l - 1) * model.B + b)
                    node_expr = f"({node_expr} + {prev_expr})"

                slot_expressions.append(node_expr)

    return slot_expressions

# # ─── TINY SHAKESPEARE BENCHMARK DATASET LOADER ──────────────────────────────
# def fetch_tinyshakespeare_benchmark(block_size=32, num_sequences=15000):
#     url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
#     file_path = "tinyshakespeare.txt"

#     if not os.path.exists(file_path):
#         print("Mirroring Tiny Shakespeare Benchmark text over network...")
#         try:
#             urllib.request.urlretrieve(url, file_path)
#         except Exception as e:
#             print(f"Network download dropped ({e}). Deploying fallback stream...")
#             fallback_text = ("First Citizen:\nBefore we proceed any further, hear me speak.\n" * 1000)
#             with open(file_path, "w", encoding="utf-8") as f:
#                 f.write(fallback_text)

#     with open(file_path, 'r', encoding='utf-8') as f:
#         text = f.read()

#     truncated_corpus = text[:150000]
#     chars = sorted(list(set(truncated_corpus)))
#     vocab_size = len(chars)

#     char_to_idx = {ch: i for i, ch in enumerate(chars)}
#     idx_to_char = {i: ch for i, ch in enumerate(chars)}
#     encoded_data = [char_to_idx[ch] for ch in truncated_corpus]

#     inputs, targets = [], []
#     for i in range(len(encoded_data) - block_size - 1):
#         if len(inputs) >= num_sequences:
#             break
#         inputs.append(encoded_data[i : i + block_size])
#         targets.append(encoded_data[i + block_size])

#     return np.array(inputs), np.array(targets), vocab_size, idx_to_char

def _resolve_tokenizer_mode(dataset_name, tokenizer_mode):
    if tokenizer_mode != "auto":
        return tokenizer_mode
    return "gpt2" if dataset_name == "fineweb" else "char"


def _load_hf_tokenizer(tokenizer_name, cache_dir=None):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "The GPT-style tokenizer needs the 'transformers' package. "
            "Install it with: pip install transformers"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, cache_dir=cache_dir)
    # We only borrow the tokenizer vocabulary; this model windows the token stream
    # itself, so GPT-2's native 1024-token model limit is not relevant here.
    tokenizer.model_max_length = int(1e30)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.sep_token_id
    return tokenizer, eos_token_id


def _encode_text_corpus(text, tokenizer_mode, tokenizer_name, cache_dir=None):
    if tokenizer_mode == "char":
        chars = sorted(list(set(text)))
        char_to_idx = {ch: i for i, ch in enumerate(chars)}
        idx_to_char = {i: ch for i, ch in enumerate(chars)}
        encoded_data = [char_to_idx[ch] for ch in text]
        return encoded_data, len(chars), idx_to_char

    if tokenizer_mode == "gpt2":
        tokenizer, eos_token_id = _load_hf_tokenizer(tokenizer_name, cache_dir=cache_dir)
        encoded_data = tokenizer.encode(text)
        if eos_token_id is not None:
            encoded_data.append(eos_token_id)
        return encoded_data, len(tokenizer), tokenizer

    raise ValueError(f"Unsupported tokenizer mode: {tokenizer_mode}")


def _split_and_window_tokens(encoded_data, block_size, val_split):
    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must be between 0 and 1.")
    if len(encoded_data) < (2 * block_size + 4):
        raise ValueError(
            f"Need more than {2 * block_size + 4} tokens to build train/val windows; "
            f"got {len(encoded_data)}."
        )

    split_idx = int(len(encoded_data) * (1.0 - val_split))
    train_tokens = encoded_data[:split_idx]
    val_tokens = encoded_data[split_idx:]
    if len(train_tokens) <= block_size + 1 or len(val_tokens) <= block_size + 1:
        raise ValueError(
            "Train and validation splits must each contain at least block_size + 2 tokens. "
            "Increase --fineweb-max-tokens, lower --block-size, or adjust --val-split."
        )

    def build_sequences(token_list):
        inputs, targets = [], []
        for i in range(len(token_list) - block_size - 1):
            inputs.append(token_list[i : i + block_size])
            targets.append(token_list[i + block_size])
        return np.array(inputs, dtype=np.int64), np.array(targets, dtype=np.int64)

    X_train, Y_train = build_sequences(train_tokens)
    X_val, Y_val = build_sequences(val_tokens)
    return X_train, Y_train, X_val, Y_val


# ─── TINY SHAKESPEARE BENCHMARK DATASET LOADER (WITH VALIDATION SPLIT) ───────
def fetch_tinyshakespeare_benchmark(block_size=32, val_split=0.1, tokenizer_mode="char", tokenizer_name="gpt2", hf_cache_dir=None):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    file_path = "tinyshakespeare.txt"

    if not os.path.exists(file_path):
        print("Mirroring Tiny Shakespeare Benchmark text over network...")
        try:
            urllib.request.urlretrieve(url, file_path)
        except Exception as e:
            print(f"Network download dropped ({e}). Deploying fallback stream...")
            fallback_text = ("First Citizen:\nBefore we proceed any further, hear me speak.\n" * 1000)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(fallback_text)

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    truncated_corpus = text[:300000]
    encoded_data, vocab_size, idx_to_char = _encode_text_corpus(
        truncated_corpus, tokenizer_mode, tokenizer_name, cache_dir=hf_cache_dir
    )
    X_train, Y_train, X_val, Y_val = _split_and_window_tokens(
        encoded_data, block_size, val_split
    )

    return X_train, Y_train, X_val, Y_val, vocab_size, idx_to_char, len(encoded_data)


def fetch_fineweb_benchmark(
    block_size=128,
    val_split=0.1,
    tokenizer_mode="gpt2",
    tokenizer_name="gpt2",
    fineweb_config="sample-10BT",
    max_tokens=200000,
    hf_cache_dir=None,
    samples_per_epoch=None,
    val_samples=None,
):
    if tokenizer_mode == "char":
        raise ValueError(
            "FineWeb should be used with a subword tokenizer. "
            "Use --tokenizer auto or --tokenizer gpt2."
        )

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "FineWeb loading needs the 'datasets' package. "
            "Install it with: pip install datasets"
        ) from exc

    os.makedirs(hf_cache_dir or ".", exist_ok=True)
    safe_config = fineweb_config.replace("/", "_").replace("\\", "_")
    safe_tokenizer = tokenizer_name.replace("/", "_").replace("\\", "_")
    bin_path = os.path.join(
        hf_cache_dir or ".",
        f"fineweb_{safe_config}_{safe_tokenizer}_{max_tokens}_tokens.uint16.bin",
    )

    tokenizer, eos_token_id = _load_hf_tokenizer(tokenizer_name, cache_dir=hf_cache_dir)
    if len(tokenizer) > np.iinfo(np.uint16).max:
        raise ValueError("The selected tokenizer vocabulary is too large for uint16 memmap storage.")

    if os.path.exists(bin_path):
        token_count = os.path.getsize(bin_path) // np.dtype(np.uint16).itemsize
        print(f"Using cached FineWeb token memmap: {bin_path} ({token_count:,} tokens)")
    else:
        print(
            f"Streaming FineWeb ({fineweb_config}) with tokenizer '{tokenizer_name}' "
            f"until {max_tokens:,} tokens are collected into {bin_path}..."
        )
        stream = load_dataset(
            "HuggingFaceFW/fineweb",
            name=fineweb_config,
            split="train",
            streaming=True,
            cache_dir=hf_cache_dir,
        )

        token_memmap = np.memmap(bin_path, dtype=np.uint16, mode="w+", shape=(max_tokens,))
        token_count = 0
        for row in stream:
            text = row.get("text", "")
            if not text:
                continue
            doc_tokens = tokenizer.encode(text)
            if eos_token_id is not None:
                doc_tokens.append(eos_token_id)
            if not doc_tokens:
                continue

            remaining = max_tokens - token_count
            if len(doc_tokens) > remaining:
                doc_tokens = doc_tokens[:remaining]
            token_memmap[token_count:token_count + len(doc_tokens)] = np.asarray(doc_tokens, dtype=np.uint16)
            token_count += len(doc_tokens)
            if token_count >= max_tokens:
                break

        token_memmap.flush()
        del token_memmap
        if token_count < max_tokens:
            print(f"FineWeb stream ended after {token_count:,} tokens.")
            trimmed = np.memmap(bin_path, dtype=np.uint16, mode="r+", shape=(max_tokens,))
            trimmed.flush()
            del trimmed
            os.truncate(bin_path, token_count * np.dtype(np.uint16).itemsize)

    if token_count < (2 * block_size + 4):
        raise ValueError(
            f"Need more than {2 * block_size + 4} tokens to build train/val windows; got {token_count}."
        )

    split_idx = int(token_count * (1.0 - val_split))
    train_dataset = TokenMemmapDataset(
        bin_path,
        block_size,
        token_start=0,
        token_end=split_idx,
        samples_per_epoch=samples_per_epoch,
        random_sampling=True,
    )
    val_dataset = TokenMemmapDataset(
        bin_path,
        block_size,
        token_start=split_idx,
        token_end=token_count,
        samples_per_epoch=val_samples,
        random_sampling=False,
    )
    return train_dataset, val_dataset, len(tokenizer), tokenizer, token_count, bin_path


def load_training_data(
    dataset_name,
    block_size,
    val_split,
    tokenizer_mode,
    tokenizer_name,
    fineweb_config,
    fineweb_max_tokens,
    hf_cache_dir,
    samples_per_epoch,
    val_samples,
):
    resolved_tokenizer = _resolve_tokenizer_mode(dataset_name, tokenizer_mode)

    if dataset_name == "tinyshakespeare":
        X_train, Y_train, X_val, Y_val, vocab_size, idx_to_char, token_count = fetch_tinyshakespeare_benchmark(
            block_size=block_size,
            val_split=val_split,
            tokenizer_mode=resolved_tokenizer,
            tokenizer_name=tokenizer_name,
            hf_cache_dir=hf_cache_dir,
        )
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(Y_train, dtype=torch.long))
        val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.long), torch.tensor(Y_val, dtype=torch.long))
        return train_dataset, val_dataset, vocab_size, idx_to_char, resolved_tokenizer, token_count, "in_memory"

    if dataset_name == "fineweb":
        train_dataset, val_dataset, vocab_size, tokenizer, token_count, bin_path = fetch_fineweb_benchmark(
            block_size=block_size,
            val_split=val_split,
            tokenizer_mode=resolved_tokenizer,
            tokenizer_name=tokenizer_name,
            fineweb_config=fineweb_config,
            max_tokens=fineweb_max_tokens,
            hf_cache_dir=hf_cache_dir,
            samples_per_epoch=samples_per_epoch,
            val_samples=val_samples,
        )
        return train_dataset, val_dataset, vocab_size, tokenizer, resolved_tokenizer, token_count, bin_path

    raise ValueError(f"Unsupported dataset: {dataset_name}")

import networkx as nx
import matplotlib.pyplot as plt

def visualize_network_topology(model, save_path="network_topology.png"):
    model.eval()
    G = nx.DiGraph()
    node_labels = {}
    node_layers = {}
    edge_prob_map = {}  # (src, dst) -> max routing prob across channels

    with torch.no_grad():
        # Spatial nodes — add explicitly with top primitive label
        sp_prim_probs = torch.softmax(model.spatial_primitive_logits, dim=-1)  # (SLOTS, P)
        sp_top_idx = sp_prim_probs.argmax(dim=-1)
        sp_top_conf = sp_prim_probs.max(dim=-1).values
        for b in range(model.NUM_PROJECTION_SLOTS):
            nid = f"t_{b}"
            prim = PRIM_LIST[sp_top_idx[b].item()]
            conf = sp_top_conf[b].item()
            node_labels[nid] = f"t_{b}\n{prim}\n{conf*100:.0f}%"
            node_layers[nid] = 0
            G.add_node(nid)

        # Deep nodes — soft routing probabilities drive edge visibility
        for l in range(model.L):
            history_cutoff = model.NUM_PROJECTION_SLOTS + (l * model.B)
            route_probs = torch.softmax(model.routing_logits[l], dim=-1)[:, :, :, 1]  # (B, 2, H)
            prim_probs = torch.softmax(model.primitive_logits[l], dim=-1)
            top_idx = prim_probs.argmax(dim=-1)
            top_conf = prim_probs.max(dim=-1).values

            for b in range(model.B):
                dst_id = f"N_{l}_{b}"
                prim = PRIM_LIST[top_idx[b].item()]
                conf = top_conf[b].item()
                node_labels[dst_id] = f"N{l}_{b}\n{prim}\n{conf*100:.0f}%"
                node_layers[dst_id] = l + 1
                G.add_node(dst_id)

                for channel in range(2):
                    for h in range(history_cutoff):
                        prob = route_probs[b, channel, h].item()
                        if prob < 0.05:  # skip near-zero routing connections
                            continue
                        if h < model.NUM_PROJECTION_SLOTS:
                            src_id = f"t_{h}"
                        else:
                            h_rem = h - model.NUM_PROJECTION_SLOTS
                            src_id = f"N_{h_rem // model.B}_{h_rem % model.B}"
                        G.add_edge(src_id, dst_id)
                        key = (src_id, dst_id)
                        edge_prob_map[key] = max(edge_prob_map.get(key, 0.0), prob)

    # Backbone: single O(V+E) ancestors call from the true output node
    output_head_id = f"N_{model.L-1}_{model.B-1}"
    active_backbone = (nx.ancestors(G, output_head_id) | {output_head_id}) if output_head_id in G else {output_head_id}

    # Stratified layout — one column per node in each layer
    layer_buckets = {}
    for node in G.nodes():
        layer = node_layers.get(node, 0)
        layer_buckets.setdefault(layer, []).append(node)

    pos = {}
    for layer, nodes in sorted(layer_buckets.items()):
        nodes_sorted = sorted(nodes)
        for idx, node in enumerate(nodes_sorted):
            pos[node] = (idx - (len(nodes_sorted) - 1) / 2.0, layer * 2.5)

    render_nodes = [n for n in G.nodes() if n in pos]
    render_edges = [(u, v) for u, v in G.edges() if u in pos and v in pos]

    node_colors, node_edge_colors = [], []
    for node in render_nodes:
        if node == output_head_id:
            node_colors.append('#d9534f'); node_edge_colors.append('#7a1a16')
        elif node in active_backbone:
            if node.startswith('t_'):
                node_colors.append('#5cb85c'); node_edge_colors.append('#245824')
            else:
                node_colors.append('#0275d8'); node_edge_colors.append('#013866')
        else:
            node_colors.append('#e6e6e6'); node_edge_colors.append('#b3b3b3')

    edge_colors, edge_widths = [], []
    for u, v in render_edges:
        prob = edge_prob_map.get((u, v), 0.05)
        in_backbone = u in active_backbone and v in active_backbone
        edge_colors.append('#4682B4' if in_backbone else '#d3d3d3')
        edge_widths.append(max(0.4, prob * 5) if in_backbone else 0.3)

    fig_w = max(14, model.NUM_PROJECTION_SLOTS * 0.6)
    fig_h = max(10, (model.L + 1) * 3.0)
    plt.figure(figsize=(fig_w, fig_h))
    nx.draw_networkx_nodes(G, pos, nodelist=render_nodes, node_color=node_colors,
                           edgecolors=node_edge_colors, node_size=900, node_shape='s', alpha=0.9)
    nx.draw_networkx_edges(G, pos, edgelist=render_edges, edge_color=edge_colors,
                           width=edge_widths, arrowsize=10, connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_labels(G, pos, labels={k: node_labels[k] for k in render_nodes if k in node_labels},
                            font_size=6, font_color='black', font_weight='bold')
    plt.title("Neuro-Symbolic Compiler Topology\n(Green=Spatial | Blue=Active | Red=Output | Gray=Inactive  |  edge width ∝ routing prob)",
              fontsize=11, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.clf()
    plt.close()
    print(f"Network topology saved to '{save_path}'.")


def generate_network_report(model, report_path, vocab_sz=None, block_size=None):
    import time as _time
    import math as _math
    import numpy as _np

    model.eval()
    lines = []
    def w(*args): lines.append(" ".join(str(a) for a in args))
    def sep(c='='): w(c * 90)

    sep()
    w("NEURO-SYMBOLIC COMPILER — POST-TRAINING NETWORK ANALYSIS")
    w(f"Generated : {_time.strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"Primitives: {NUM_PRIMITIVES}  |  Layers: {model.L}  |  Breadth: {model.B}  |  Spatial slots: {model.NUM_PROJECTION_SLOTS}  |  D: {model.D}")
    sep()
    w("")

    with torch.no_grad():

        # ── 1. MODEL FOOTPRINT ──────────────────────────────────────────────────
        sep('-')
        w("1. MODEL FOOTPRINT")
        sep('-')
        total_params   = sum(p.numel() for p in model.parameters())
        emb_params     = model.token_embeddings.weight.numel() + model.pos_embeddings.numel()
        routing_params = (sum(p.numel() for p in model.routing_logits)
                          + sum(p.numel() for p in model.routing_edge_weights)
                          + model.routing_biases.numel())
        struct_params  = (model.spatial_routing_logits.numel()
                          + model.spatial_primitive_logits.numel()
                          + model.primitive_logits.numel())
        compute_params = total_params - emb_params - routing_params - struct_params
        w(f"  Total parameters     : {total_params:>12,}")
        w(f"    Embedding          : {emb_params:>12,}  ({emb_params/total_params*100:.1f}%)")
        w(f"    Routing weights    : {routing_params:>12,}  ({routing_params/total_params*100:.1f}%)")
        w(f"    Structural logits  : {struct_params:>12,}  ({struct_params/total_params*100:.1f}%)")
        w(f"    Compute (W/b)      : {compute_params:>12,}  ({compute_params/total_params*100:.1f}%)")
        w(f"  Size @ FP32          : {total_params*4/1024/1024:.2f} MB")
        w(f"  Size @ BF16/FP16     : {total_params*2/1024/1024:.2f} MB")
        if vocab_sz:   w(f"  Vocab size           : {vocab_sz:,}")
        if block_size: w(f"  Context length       : {block_size}")
        w("")

        device = next(model.parameters()).device
        if device.type == "cuda" and block_size:
            _dx = torch.zeros(1, block_size, dtype=torch.long, device=device)
            for _ in range(5): model(_dx, temperature=1.0)
            torch.cuda.synchronize()
            t0 = _time.perf_counter()
            for _ in range(100): model(_dx, temperature=1.0)
            torch.cuda.synchronize()
            inf_ms = (_time.perf_counter() - t0) / 100 * 1000
            w(f"  GPU inference time   : {inf_ms:.3f} ms / sample  (batch=1, seq={block_size})")
            w(f"  Throughput estimate  : {1000/inf_ms:.0f} samples/sec")
            del _dx
        w("")

        # ── 2. NODE UTILIZATION BY LAYER ────────────────────────────────────────
        sep('-')
        w("2. NODE UTILIZATION BY LAYER")
        sep('-')
        w(f"  Thresholds: dominant=>50% | committed=>35% | split=<35%")
        w(f"  {'Layer':<18} {'Nodes':>5}  {'Dominant':>8}  {'Committed':>9}  {'Split':>5}")
        w(f"  {'-'*18} {'-'*5}  {'-'*8}  {'-'*9}  {'-'*5}")
        DOMINANT_T, COMMITTED_T = 0.50, 0.35

        def utilization_row(label, probs_matrix):
            n = probs_matrix.shape[0]
            dom = sum(1 for i in range(n) if probs_matrix[i].max() >= DOMINANT_T)
            com = sum(1 for i in range(n) if probs_matrix[i].max() >= COMMITTED_T)
            spl = n - com
            w(f"  {label:<18} {n:>5}  {dom:>8}  {com:>9}  {spl:>5}")
            return n, dom, com

        sp_probs = torch.softmax(model.spatial_primitive_logits, dim=-1).cpu().numpy()
        tn, td, tc = utilization_row("Spatial (slot 0)", sp_probs)
        total_n, total_d, total_c = tn, td, tc
        all_prim_probs = [sp_probs]

        for l in range(model.L):
            pp = torch.softmax(model.primitive_logits[l], dim=-1).cpu().numpy()
            tn, td, tc = utilization_row(f"Deep layer {l}", pp)
            total_n += tn; total_d += td; total_c += tc
            all_prim_probs.append(pp)

        w(f"  {'-'*18} {'-'*5}  {'-'*8}  {'-'*9}  {'-'*5}")
        w(f"  {'TOTAL':<18} {total_n:>5}  {total_d:>8}  {total_c:>9}  {total_n-total_c:>5}")
        w(f"  Network commitment rate: {total_c/total_n*100:.1f}%  |  dominance rate: {total_d/total_n*100:.1f}%")
        w("")

        # ── 3. TOP PRIMITIVES PER LAYER ─────────────────────────────────────────
        sep('-')
        w("3. TOP PRIMITIVES PER LAYER  (mean probability across nodes, top 5)")
        sep('-')

        all_probs_cat = _np.concatenate(all_prim_probs, axis=0)  # (total_nodes, P)

        def top_prims_block(label, pmat, n_top=5):
            avg = pmat.mean(axis=0)
            top = sorted(range(NUM_PRIMITIVES), key=lambda i: -avg[i])[:n_top]
            w(f"  {label}:")
            for i in top:
                bar = '█' * max(1, int(avg[i] * 40))
                w(f"    {PRIM_LIST[i]:<20} {avg[i]*100:5.1f}%  {bar}")
            w("")

        top_prims_block("Spatial layer", sp_probs)
        for l in range(model.L):
            top_prims_block(f"Deep layer {l}", all_prim_probs[l + 1])

        top_prims_block("GLOBAL (all layers)", all_probs_cat)

        # ── 4. PRIMITIVE DOMINANCE vs SPLIT — PER NODE ──────────────────────────
        sep('-')
        w("4. PRIMITIVE DOMINANCE vs SPLIT — PER NODE  (rel-entropy: 0=certain, 1=uniform)")
        sep('-')
        max_ent = _math.log(NUM_PRIMITIVES)
        w(f"  {'Node':<13} {'Top primitive':<22} {'Conf%':>6}  {'Rel-H':>6}  {'Status'}")
        w(f"  {'-'*13} {'-'*22} {'-'*6}  {'-'*6}  {'--------'}")

        def node_row(label, probs):
            ti = int(probs.argmax())
            conf = float(probs[ti])
            ent = float(-sum(p * _math.log(float(p) + 1e-9) for p in probs))
            rh = ent / max_ent
            status = "DOMINANT" if conf >= DOMINANT_T else ("committed" if conf >= COMMITTED_T else "split   ")
            w(f"  {label:<13} {PRIM_LIST[ti]:<22} {conf*100:6.1f}%  {rh:6.3f}  {status}")

        for b in range(model.NUM_PROJECTION_SLOTS):
            node_row(f"t_{b}", sp_probs[b])
        for l in range(model.L):
            pp = all_prim_probs[l + 1]
            for b in range(model.B):
                node_row(f"N_{l}_{b}", pp[b])
        w("")

        # ── 5. ROUTING UTILIZATION ───────────────────────────────────────────────
        sep('-')
        w("5. ROUTING UTILIZATION — active history slots per layer  (threshold 10%)")
        sep('-')
        ROUTE_T = 0.10
        w(f"  {'Layer':<14} {'Pool size':>9}  {'Possible':>8}  {'Active':>6}  {'Active%':>7}  {'Mean prob':>9}")
        w(f"  {'-'*14} {'-'*9}  {'-'*8}  {'-'*6}  {'-'*7}  {'-'*9}")
        for l in range(model.L):
            hc = model.NUM_PROJECTION_SLOTS + (l * model.B)
            rp = torch.softmax(model.routing_logits[l], dim=-1)[:, :, :, 1].cpu().numpy()
            possible = model.B * 2 * hc
            active = int((rp > ROUTE_T).sum())
            mean_p = float(rp.mean())
            w(f"  {'Deep layer '+str(l):<14} {hc:>9}  {possible:>8}  {active:>6}  {active/possible*100:7.1f}%  {mean_p*100:9.2f}%")
        w("")

        # ── 6. SPATIAL TOKEN POSITION USAGE ─────────────────────────────────────
        sep('-')
        w("6. SPATIAL ROUTING — AVERAGE STATIC POOLING WEIGHT PER TOKEN POSITION (across slots and channels)")
        sep('-')
        sp_route_probs = torch.softmax(model.spatial_routing_logits, dim=-1).cpu().numpy()  # (SLOTS, 2, T)
        mean_pos = sp_route_probs.mean(axis=(0, 1))  # (T,) avg across slots and channels
        top_pos = sorted(range(len(mean_pos)), key=lambda t: -mean_pos[t])[:10]
        w(f"  Top-10 token positions (0=oldest, T-1=most recent):")
        for t in top_pos:
            bar = '█' * max(1, int(mean_pos[t] * 200))
            w(f"    tok_{t:<4}  {mean_pos[t]*100:5.2f}%  {bar}")
        w("")

        # ── 7. DETAILED SYMBOLIC EQUATIONS ──────────────────────────────────────
        sep('-')
        w("7. DETAILED SYMBOLIC EQUATIONS — COMPLETE NETWORK")
        sep('-')
        equations = extract_symbolic_equations_soft(model)
        w("  [SPATIAL COMPRESSION LAYER]")
        w("  Each t_b compresses the token sequence into one D-dim vector.")
        w("")
        for b in range(model.NUM_PROJECTION_SLOTS):
            pp = sp_probs[b]
            ti = int(pp.argmax())
            conf = float(pp[ti])
            w(f"  t_{b} [{PRIM_LIST[ti]} @ {conf*100:.1f}%]")
            w(f"      = {equations[b]}")
        w("")
        eq_idx = model.NUM_PROJECTION_SLOTS
        for l in range(model.L):
            pp_l = all_prim_probs[l + 1]
            w(f"  [DEEP LAYER {l}]")
            for b in range(model.B):
                ti = int(pp_l[b].argmax())
                conf = float(pp_l[b][ti])
                residual = f" + N_{l-1}_{b}" if l > 0 else ""
                w(f"  N_{l}_{b} [{PRIM_LIST[ti]} @ {conf*100:.1f}%]")
                w(f"      = {equations[eq_idx]}{residual}")
                eq_idx += 1
            w("")

        sep('-')
        w("  [OUTPUT HEAD]")
        w(f"  Uses buffer slot -1 = N_{model.L-1}_{model.B-1}")
        w(f"  logits = N_{model.L-1}_{model.B-1} @ token_embeddings.T   (shape: vocab_size)")
        w(f"  prediction = argmax(softmax(logits))")
        w("")

        sep()
        w("END OF REPORT")
        sep()

    report_text = "\n".join(lines)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\nNetwork report written to '{report_path}'.")

    # Print first ~60 lines to console as a summary
    print("\n" + "=" * 90)
    for line in lines[:60]:
        print(line)
    if len(lines) > 60:
        print(f"... [{len(lines) - 60} more lines — see {report_path}]")


# ─── MAIN SYSTEM EXECUTION RUNNER ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the neuro-symbolic world model.")
    parser.add_argument(
        "--dataset",
        choices=["tinyshakespeare", "fineweb"],
        default="tinyshakespeare",
        help="Training corpus to use.",
    )
    parser.add_argument(
        "--tokenizer",
        choices=["auto", "char", "gpt2"],
        default="auto",
        help="Tokenizer mode. 'auto' uses char for Tiny Shakespeare and GPT-2 BPE for FineWeb.",
    )
    parser.add_argument(
        "--tokenizer-name",
        default="gpt2",
        help="Hugging Face tokenizer name used when --tokenizer is gpt2 or auto resolves to gpt2.",
    )
    parser.add_argument(
        "--fineweb-config",
        default="sample-10BT",
        help="FineWeb dataset configuration to stream from Hugging Face.",
    )
    parser.add_argument(
        "--fineweb-max-tokens",
        type=int,
        default=200000,
        help="Number of FineWeb tokens to stream before building train/validation windows.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default="fineweb_cache",
        help="Cache directory for Hugging Face tokenizer and dataset shard files.",
    )
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=None,
        help="Optional number of randomly sampled training windows per epoch.",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=None,
        help="Optional number of validation windows evaluated per epoch.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="searchgpt_checkpoint.pt",
        help="Where to save the trained model checkpoint.",
    )
    parser.add_argument("--block-size", type=int, default=128, help="Context length.")
    parser.add_argument("--epochs", type=int, default=600, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size.")
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Accumulate this many micro-batches before each optimizer step.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA mixed precision to reduce activation memory.",
    )
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation fraction.")
    args = parser.parse_args()
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be at least 1.")
    hf_cache_dir = os.path.abspath(args.hf_cache_dir) if args.hf_cache_dir else None

# ─── EXTRA CAPACITY SYMBOLIC GRID CONFIGURATION ──────────────────────────

    NUM_PROJECTION_SLOTS = 32
    BLOCK_SIZE = args.block_size # default 128
    NUM_LAYERS = 8
    BREADTH = 8     
    EMBED_DIM = 256  
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    # if args.dataset == "fineweb" and args.tokenizer in ["auto", "gpt2"]:
        # LR_CONTINUOUS = 3* 0.002 * (BATCH_SIZE / 512)
        # LR_STRUCTURAL = 3* 0.01 * (BATCH_SIZE / 512)

        # LR_CONTINUOUS_DECAY = 0.005 
        # LR_STRUCTURAL_DECAY = 0.001 * (BATCH_SIZE / 4096)
    # elif args.dataset == "tinyshakespeare" and args.tokenizer in ["auto", "char"]:
    # LR_CONTINUOUS = 0.002 * (BATCH_SIZE / 512)
    # LR_STRUCTURAL = 0.01 * (BATCH_SIZE / 512)

    # LR_CONTINUOUS_DECAY = 0.01 
    # LR_STRUCTURAL_DECAY = 0.001 * (BATCH_SIZE / 4096)


    LR_CONTINUOUS = 3* 0.002 * (BATCH_SIZE / 512)
    LR_STRUCTURAL = 3* 0.01  #(BATCH_SIZE / 512)
    GRAD_CLIP_NORM = 20.0

    

    # X_data, Y_data, vocab_sz, idx_to_char = fetch_tinyshakespeare_benchmark(block_size=BLOCK_SIZE)

    train_dataset, val_dataset, vocab_sz, idx_to_char, tokenizer_used, token_count, data_path = load_training_data(
        dataset_name=args.dataset,
        block_size=BLOCK_SIZE,
        val_split=args.val_split,
        tokenizer_mode=args.tokenizer,
        tokenizer_name=args.tokenizer_name,
        fineweb_config=args.fineweb_config,
        fineweb_max_tokens=args.fineweb_max_tokens,
        hf_cache_dir=hf_cache_dir,
        samples_per_epoch=args.samples_per_epoch,
        val_samples=args.val_samples,
    )

    # Effective per-epoch window counts: honour --samples-per-epoch / --val-samples
    # for TinyShakespeare; FineWeb already bakes this into TokenMemmapDataset.__len__().
    if args.dataset == "tinyshakespeare" and args.samples_per_epoch:
        train_samples_per_epoch = args.samples_per_epoch
    else:
        train_samples_per_epoch = len(train_dataset)

    if args.dataset == "tinyshakespeare" and args.val_samples:
        val_samples_per_epoch = args.val_samples
    else:
        val_samples_per_epoch = len(val_dataset)

    steps_per_epoch = train_samples_per_epoch // BATCH_SIZE

    # LR_CONTINUOUS_DECAY = 0.005 * (BATCH_SIZE / 512) / (steps_per_epoch/269)
    # LR_STRUCTURAL_DECAY = 0.001 * (BATCH_SIZE / 4096) / (steps_per_epoch/269)

    # Decays scale ONLY with steps_per_epoch (NOT batch)
    # REFERENCE_STEPS = 263
    # decay_scale = REFERENCE_STEPS / steps_per_epoch  

    # LR_CONTINUOUS_DECAY = 0.005 * decay_scale
    # LR_STRUCTURAL_DECAY = 0.0005 * decay_scale


    TARGET_PER_EPOCH_CONTINUOUS_DECAY = 0.00806
    TARGET_PER_EPOCH_STRUCTURAL_DECAY = 0.00101

    # LR_CONTINUOUS_DECAY = TARGET_PER_EPOCH_CONTINUOUS_DECAY / (LR_CONTINUOUS * steps_per_epoch)
    # LR_STRUCTURAL_DECAY = TARGET_PER_EPOCH_STRUCTURAL_DECAY / (LR_STRUCTURAL * steps_per_epoch)
    LR_STRUCTURAL_DECAY = 0.0
    LR_CONTINUOUS_DECAY = 0.0


    
    model = SymbolicWorldModel(
        block_size=BLOCK_SIZE, num_layers=NUM_LAYERS,
        breadth=BREADTH, embed_dim=EMBED_DIM, vocab_size=vocab_sz,
        num_projection_slots=NUM_PROJECTION_SLOTS
    )

    # ─── v4 ROUTING / LN / RESIDUAL CHANGE VERIFICATION ──────────────────────
    total_params = sum(p.numel() for p in model.parameters())
    edge_w_params = sum(p.numel() for p in model.routing_edge_weights)
    route_bias_params = model.routing_biases.numel()
    edge_w_shapes = [tuple(p.shape) for p in model.routing_edge_weights]
    print("-" * 90)
    print(f"[v4] Total model parameters = {total_params:,}")
    print(f"[v4] Params added by routing_edge_weights = {edge_w_params:,}; "
          f"by routing_biases = {route_bias_params:,}; "
          f"combined new routing params = {edge_w_params + route_bias_params:,}")
    print(f"[v4] Per-edge routing weights added: {edge_w_shapes}, "
          f"routing_biases {tuple(model.routing_biases.shape)}, "
          f"LN pre-norm enabled, transformer-style residual enabled.")
    print("-" * 90)

    effective_batch_size = BATCH_SIZE * args.grad_accum_steps
    print(f"DATASET={args.dataset}, TOKENIZER={tokenizer_used}, VOCAB_SIZE={vocab_sz:,}, TOKENS={token_count:,}, DATA_PATH={data_path}")
    print(f"TRAIN_WINDOWS_PER_EPOCH={train_samples_per_epoch:,}, VAL_WINDOWS_PER_EPOCH={val_samples_per_epoch:,}")
    print(f"HF_CACHE_DIR={hf_cache_dir}")
    print(f"MICRO_BATCH_SIZE={BATCH_SIZE}, GRAD_ACCUM_STEPS={args.grad_accum_steps}, EFFECTIVE_BATCH_SIZE={effective_batch_size}, AMP={args.amp}")
    print(f"BLOCK_SIZE={BLOCK_SIZE}, NUM_LAYERS={NUM_LAYERS}, BREADTH={BREADTH}, EMBED_DIM={EMBED_DIM}, BATCH_SIZE={BATCH_SIZE}, LR_CONTINUOUS={LR_CONTINUOUS:.4f}, LR_STRUCTURAL={LR_STRUCTURAL:.4f} , LR_CONTINUOUS_DECAY={LR_CONTINUOUS_DECAY}, LR_STRUCTURAL_DECAY={LR_STRUCTURAL_DECAY}, GRAD_CLIP_NORM={GRAD_CLIP_NORM}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    amp_enabled = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # BF16 has FP32 range; no loss scaling needed

    # dataset = torch.utils.data.TensorDataset(torch.tensor(X_data, dtype=torch.long), torch.tensor(Y_data, dtype=torch.long))
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # num_workers=0 on WSL: worker subprocesses trigger LLVM pthread_join failures.
    # TensorDataset (TinyShakespeare) is already in RAM so there is no I/O to overlap anyway.
    _dl_num_workers = 0 if os.name != 'nt' else 2
    _dl_kwargs = dict(batch_size=BATCH_SIZE, num_workers=_dl_num_workers, pin_memory=(device.type == "cuda"), persistent_workers=(_dl_num_workers > 0))

    if args.dataset == "tinyshakespeare" and args.samples_per_epoch:
        _train_sampler = torch.utils.data.RandomSampler(
            train_dataset, replacement=True, num_samples=args.samples_per_epoch
        )
        train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=_train_sampler, **_dl_kwargs)
    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, shuffle=(args.dataset != "fineweb"), **_dl_kwargs
        )

    if args.dataset == "tinyshakespeare" and args.val_samples:
        _val_sampler = torch.utils.data.RandomSampler(
            val_dataset, replacement=False, num_samples=args.val_samples
        )
        val_dataloader = torch.utils.data.DataLoader(val_dataset, sampler=_val_sampler, **_dl_kwargs)
    else:
        val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **_dl_kwargs)


    structural_params = []
    continuous_params = []
    for name, param in model.named_parameters():
        if 'logits' in name or 'spatial' in name:
            structural_params.append(param)
        else:
            continuous_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': continuous_params, 'lr': LR_CONTINUOUS, 'weight_decay': LR_CONTINUOUS_DECAY},
        {'params': structural_params, 'lr': LR_STRUCTURAL,  'weight_decay': LR_STRUCTURAL_DECAY}
    ])

    if hasattr(torch, 'compile') and device.type == "cuda" and os.name != 'nt':
        class _Spinner:
            def __init__(self, msg):
                self._msg = msg
                self._stop = threading.Event()
                self._t = threading.Thread(target=self._run, daemon=True)
            def _run(self):
                t0 = time.time()
                for ch in itertools.cycle(r'\|/-'):
                    if self._stop.is_set():
                        break
                    sys.stdout.write(f'\r{self._msg} {ch}  {time.time()-t0:.0f}s elapsed')
                    sys.stdout.flush()
                    time.sleep(0.15)
            def start(self):
                self._t.start()
                return self
            def stop(self, elapsed):
                self._stop.set()
                self._t.join()
                sys.stdout.write(f'\r{self._msg} done in {elapsed:.0f}s\n')
                sys.stdout.flush()

        model = torch.compile(model, mode="reduce-overhead", dynamic=False)
        _t0 = time.time()
        _sp = _Spinner("torch.compile: tracing graph...").start()
        # Warm-up with actual batch shape so CUDA graphs are captured correctly
        _dx = torch.zeros(BATCH_SIZE, BLOCK_SIZE, dtype=torch.long, device=device)
        _dy = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)
        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            _wbuf = model(_dx, temperature=2.0)
            _wloss = F.cross_entropy(_wbuf[:, -1, :] @ model.token_embeddings.weight.T, _dy)
        _wloss.backward()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        _sp.stop(time.time() - _t0)
        del _dx, _dy, _wbuf, _wloss

    loss_history = []
    val_loss_history = [] # Track validation alongside training loss
    entropy_history = []
    primitive_confidence_history = {p: [] for p in PRIM_LIST}

    print("-" * 90)
    print(f"Neuro-Symbolic Search Space Confirmed.")
    print(f"Hardware Compute Node Status: {device}")
    print("-" * 90)

    for epoch in range(1, EPOCHS + 1):
        progress = (epoch - 1) / float(max(1, EPOCHS - 1))
        # current_temp = 2.0 * (0.05 ** progress)

        # current_temp = 2.0 * (0.05 ** (progress ** 2))
        current_temp = 2.0


        current_entropy_lambda = 0.03 * (1.0 - progress)

        model.train()
        running_ce_tensor = torch.zeros((), device=device)
        optimizer.zero_grad(set_to_none=True)
        epoch_grad_norm = 0.0

        for batch_idx, (batch_x, batch_y) in enumerate(train_dataloader, start=1):
            batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                buffer = model(batch_x, temperature=current_temp)
                final_node_state = buffer[:, -1, :]
                logits = final_node_state @ model.token_embeddings.weight.T

                ce_loss = F.cross_entropy(logits, batch_y)
            routing_entropy = compute_structural_entropy(model)
            total_loss = ce_loss - (current_entropy_lambda * routing_entropy)
            scaled_loss = total_loss / args.grad_accum_steps

            scaled_loss.backward()

            if batch_idx % args.grad_accum_steps == 0 or batch_idx == len(train_dataloader):
                epoch_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM).item()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_ce_tensor += ce_loss.detach()

        epoch_ce = running_ce_tensor.item() / len(train_dataloader)
        epoch_entropy = routing_entropy.item()

        # ─── PHASE B: VALIDATION EVALUATION CYCLE ───
        model.eval()
        running_val_ce_tensor = torch.zeros((), device=device)

        with torch.no_grad():
            for val_x, val_y in val_dataloader:
                val_x, val_y = val_x.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
                
                # Pass current temperature so routing profiles match identically
                with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                    val_buffer = model(val_x, temperature=current_temp)
                    val_final_node = val_buffer[:, -1, :]
                    val_logits = val_final_node @ model.token_embeddings.weight.T
                
                    val_ce = F.cross_entropy(val_logits, val_y)
                running_val_ce_tensor += val_ce.detach()

        epoch_val_ce = running_val_ce_tensor.item() / len(val_dataloader)

        loss_history.append(epoch_ce)
        val_loss_history.append(epoch_val_ce)
        entropy_history.append(epoch_entropy)

        with torch.no_grad():
            probs = torch.softmax(model.primitive_logits[0, 0], dim=-1).cpu().numpy()
            for idx, name in enumerate(PRIM_LIST):
                primitive_confidence_history[name].append(probs[idx])

        if epoch % 1 == 0 or epoch == 1:
            favored_idx = np.argmax(probs)
            clipped = "CLIPPED" if epoch_grad_norm > GRAD_CLIP_NORM else "ok"
            print(f"Epoch {epoch:02d}/{EPOCHS} | CE Loss: {epoch_ce:.4f} | VAL CE LOSS: {epoch_val_ce:.4f} | Topology Entropy: {epoch_entropy:.4f} | Temp: {current_temp:.3f} | GradNorm: {epoch_grad_norm:.3f} [{clipped}]")
            print(f"   -> L0_N0 Leading Path: '{PRIM_LIST[favored_idx]}' ({probs[favored_idx]*100:.1f}% confidence)")


    model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
    checkpoint = {
        "model_state_dict": model_to_save.state_dict(),
        "model_config": {
            "block_size": BLOCK_SIZE,
            "num_layers": NUM_LAYERS,
            "breadth": BREADTH,
            "embed_dim": EMBED_DIM,
            "vocab_size": vocab_sz,
            "num_projection_slots": NUM_PROJECTION_SLOTS,
        },
        "prim_list": list(PRIM_LIST),
        "dataset": args.dataset,
        "tokenizer_mode": tokenizer_used,
        "tokenizer_name": args.tokenizer_name,
        "hf_cache_dir": hf_cache_dir,
        "idx_to_char": idx_to_char if tokenizer_used == "char" else None,
        "loss_history": loss_history,
        "val_loss_history": val_loss_history,
        "entropy_history": entropy_history,
    }
    torch.save(checkpoint, args.checkpoint_path)
    print(f"Checkpoint saved to '{args.checkpoint_path}'.")


    # Generate training metrics plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS + 1), loss_history, label='Train CE Loss', color='#d9534f', linewidth=2)
    plt.plot(range(1, EPOCHS + 1), val_loss_history, label='Val CE Loss', color='#f0ad4e', linewidth=2, linestyle='-.')
    plt.plot(range(1, EPOCHS + 1), entropy_history, label='Topology Entropy', color='#0275d8', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Magnitude')
    plt.title('Neuro-Symbolic Search Space Trajectory')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('optimization_profile.png', dpi=150)
    plt.clf()
    plt.close()

    # Unwrap compiled model for analysis
    analysis_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    visualize_network_topology(analysis_model, save_path="network_topology.png")
    generate_network_report(analysis_model, report_path="network_report.txt",
                            vocab_sz=vocab_sz, block_size=BLOCK_SIZE)
