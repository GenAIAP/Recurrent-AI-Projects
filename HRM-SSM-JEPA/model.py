# ==============================================================================
# ARCHITECTURE v7.2: CALIBRATED GATED-DELTANET SSM-HRM-JEPA (TINYSTORIES+IMAGES)
# ==============================================================================
import copy
import math
import os
import subprocess
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from transformers import AutoTokenizer

try:
    from datasets import load_dataset
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "datasets"]
    )
    from datasets import load_dataset


def setup_drive_workspace(folder_name: str = "SSM_HRM_JEPA_v1_4"):
    drive_root = "/content/drive/MyDrive"
    if not os.path.exists(drive_root):
        try:
            from google.colab import drive

            drive.mount("/content/drive")
        except Exception:
            drive_root = "."
    work_dir = os.path.join(drive_root, folder_name)
    os.makedirs(work_dir, exist_ok=True)
    return work_dir


WORK_DIR = setup_drive_workspace()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


def fast_encode(text: str, max_chars: int = 500, max_len: int = 80):
    return tokenizer.encode(
        " " + text[:max_chars].strip(), truncation=True, max_length=max_len
    )


def multi_sphere_norm(x: torch.Tensor, sphere_dim: int = 16) -> torch.Tensor:
    orig_shape = x.shape
    grouped = x.view(*orig_shape[:-1], -1, sphere_dim)
    return F.normalize(grouped, p=2, dim=-1).view(*orig_shape)


# === [SECTION 1: CO-SCALED CONFIGURATION] =====================================
class ScaleConfig:
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dec_layers: int,
        highlight_k: int,
        lr: float,
    ):
        self.vocab_size: int = tokenizer.vocab_size
        self.bos_token_id: int = tokenizer.bos_token_id or tokenizer.eos_token_id
        self.eos_token_id: int = tokenizer.eos_token_id
        self.pad_token_id: int = tokenizer.pad_token_id
        self.d_model: int = d_model
        self.d_state: int = d_model // 2
        self.d_domain: int = d_model // 2
        self.n_heads: int = n_heads
        self.dec_layers: int = dec_layers
        self.highlight_k: int = highlight_k
        self.top_k_mem: int = 2
        self.plan_horizons: int = 3
        self.seq_len: int = 64
        self.num_ideas: int = 4
        self.tokens_per_idea: int = 16
        self.vis_patches: int = 16
        self.vis_dim: int = 576
        self.hrm_max_ticks: int = 5
        self.act_threshold: float = 0.95
        self.jepa_temp: float = 0.1
        self.ema_decay: float = 0.99
        self.lr: float = lr
        self.batch_size: int = 24
        self.steps: int = 500
        self.eval_every: int = 100
        self.cache_size: int = 5000
        self.work_dir: str = WORK_DIR
        self.device: str = DEVICE


# === [SECTION 2: GATED DELTANET MATRIX-STATE SSM BACKBONE] ====================
class DeltaMatrixSSMBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.gate_proj = nn.Linear(d_model, n_heads)
        self.beta_proj = nn.Linear(d_model, n_heads)
        self.out_proj = nn.Linear(d_model, d_model)

        hidden_d = int(2.0 * d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_d),
            nn.GELU(),
            nn.Linear(hidden_d, d_model),
        )

    def forward(self, x: torch.Tensor, state: torch.Tensor = None):
        single_step = x.dim() == 2
        if single_step:
            x = x.unsqueeze(1)
        B, L, D = x.shape
        H, Dh = self.n_heads, self.d_head

        h = self.norm1(x)
        qkv = self.qkv_proj(h).view(B, L, 3, H, Dh)
        q = F.normalize(qkv[:, :, 0], p=2, dim=-1)
        k = F.normalize(qkv[:, :, 1], p=2, dim=-1)
        v = qkv[:, :, 2]

        log_decay = -F.softplus(self.gate_proj(h)).clamp(min=0.01, max=0.4)
        beta = torch.sigmoid(self.beta_proj(h))

        if single_step or L == 1:
            if state is None:
                state = torch.zeros(B, H, Dh, Dh, device=x.device, dtype=x.dtype)
            gamma = torch.exp(log_decay[:, 0]).view(B, H, 1, 1)
            beta_t = beta[:, 0].view(B, H, 1, 1)

            decayed_S = gamma * state
            v_pred = torch.einsum("bhde,bhd->bhe", decayed_S, k[:, 0])
            v_err = v[:, 0] - v_pred
            delta_outer = torch.einsum("bhd,bhe->bhde", k[:, 0], v_err)

            state = decayed_S + beta_t * delta_outer
            y = torch.einsum("bhde,bhd->bhe", state, q[:, 0]).reshape(B, 1, D)
        else:
            q_t = q.permute(0, 2, 1, 3)
            k_t = k.permute(0, 2, 1, 3)
            v_t = v.permute(0, 2, 1, 3) * beta.permute(0, 2, 1).unsqueeze(-1)

            cum_decay = torch.cumsum(log_decay.permute(0, 2, 1), dim=-1)
            diff = cum_decay.unsqueeze(-1) - cum_decay.unsqueeze(-2)
            causal_mask = torch.tril(
                torch.ones(L, L, device=x.device, dtype=torch.bool)
            )
            decay_mask = (
                torch.exp(diff.masked_fill(~causal_mask, -1e9))
                * causal_mask.float()
            )

            strict_past = torch.tril(
                torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=-1
            )
            kk_past = (
                torch.matmul(k_t, k_t.transpose(-1, -2)) * decay_mask
            ) * strict_past.float()
            v_delta = v_t - 0.5 * torch.matmul(kk_past, v_t)

            attn_linear = torch.matmul(q_t, k_t.transpose(-1, -2)) * decay_mask
            y_heads = torch.matmul(attn_linear, v_delta)
            y = y_heads.permute(0, 2, 1, 3).reshape(B, L, D)

            tail_decay = torch.exp(
                cum_decay[:, :, -1:] - cum_decay
            ).unsqueeze(-1)
            state = torch.matmul(k_t.transpose(-1, -2), v_delta * tail_decay)

        x = x + self.out_proj(y)
        out = x + self.ffn(self.norm2(x))
        return (out.squeeze(1), state) if single_step else (out, state)


class TextIdeaSlicer(nn.Module):
    def __init__(self, cfg: ScaleConfig):
        super().__init__()
        self.cfg = cfg
        self.ssm = DeltaMatrixSSMBlock(cfg.d_model, cfg.n_heads)
        self.boundary_head = nn.Linear(cfg.d_model, 1)

    def forward(self, tok_emb: torch.Tensor, num_chunks: int = None):
        B, N, D = tok_emb.shape
        L_c = self.cfg.tokens_per_idea
        M = num_chunks if num_chunks is not None else (N // L_c)
        h_tok, _ = self.ssm(tok_emb)
        b_prob = torch.sigmoid(self.boundary_head(h_tok)).squeeze(-1)
        ratio_loss = (b_prob.sum(dim=-1).mean() - float(M)) ** 2
        w_local = F.softmax(b_prob.view(B, M, L_c), dim=-1)
        idea_chunks = (
            h_tok.view(B, M, L_c, D) * w_local.unsqueeze(-1)
        ).sum(dim=2)
        w_slice = torch.zeros(
            B, M, N, device=tok_emb.device, dtype=tok_emb.dtype
        )
        for m in range(M):
            w_slice[:, m, m * L_c : (m + 1) * L_c] = w_local[:, m, :]
        return idea_chunks, h_tok, w_slice, ratio_loss


class VisualIdeaSlicer(nn.Module):
    def __init__(self, cfg: ScaleConfig):
        super().__init__()
        self.cfg = cfg
        self.vis_proj = nn.Linear(cfg.vis_dim, cfg.d_model)
        self.ssm = DeltaMatrixSSMBlock(cfg.d_model, cfg.n_heads)
        self.slot_queries = nn.Parameter(
            torch.randn(1, 256, cfg.d_model) * 0.02
        )
        self.null_vis_patches = nn.Parameter(
            torch.randn(1, cfg.vis_patches, cfg.vis_dim) * 0.02
        )

    def forward(
        self,
        vis_patches: torch.Tensor = None,
        has_vision_mask: torch.Tensor = None,
        batch_size: int = 1,
        num_chunks: int = None,
    ):
        M = num_chunks if num_chunks is not None else self.cfg.num_ideas
        if vis_patches is None:
            vis_patches = self.null_vis_patches.expand(batch_size, -1, -1)
        elif has_vision_mask is not None:
            mask = has_vision_mask.view(-1, 1, 1).float()
            vis_patches = (
                mask * vis_patches
                + (1.0 - mask)
                * self.null_vis_patches.expand(vis_patches.size(0), -1, -1)
            )
        h_vis, _ = self.ssm(self.vis_proj(vis_patches))
        queries = self.slot_queries[:, :M, :]
        w_vis = F.softmax(
            torch.matmul(queries, h_vis.transpose(-1, -2))
            / math.sqrt(self.cfg.d_model),
            dim=-1,
        )
        return torch.matmul(w_vis, h_vis), h_vis, w_vis


# === [SECTION 3: SPAN-COUPLED HIGHLIGHTER] ====================================
class FactorizedGistAndHighlighter(nn.Module):
    def __init__(self, cfg: ScaleConfig):
        super().__init__()
        self.cfg = cfg
        self.state_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_state),
        )
        self.domain_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_domain),
        )
        self.salience_scorer = nn.Sequential(
            nn.Linear(cfg.d_model * 2 + 1, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 1),
        )
        self.span_conv = nn.Conv1d(
            1, 1, kernel_size=3, padding=1, bias=False
        )
        with torch.no_grad():
            self.span_conv.weight.copy_(torch.tensor([[[0.25, 1.0, 0.25]]]))

        self.slot_pos_emb = nn.Parameter(
            torch.randn(1, 1, cfg.highlight_k, cfg.d_model) * 0.02
        )

    def encode_gist(self, idea_chunks: torch.Tensor):
        s_state = multi_sphere_norm(self.state_proj(idea_chunks), sphere_dim=16)
        s_domain = multi_sphere_norm(
            self.domain_proj(idea_chunks), sphere_dim=16
        )
        return torch.cat([s_state, s_domain], dim=-1), s_state, s_domain

    def highlight_tokens(self, h_tok, raw_emb, w_slice, raw_token_ids):
        B, M, N = w_slice.shape
        Kh, L_c = self.cfg.highlight_k, self.cfg.tokens_per_idea
        h_chunks = h_tok.view(B, M, L_c, self.cfg.d_model)
        chunk_mean = h_chunks.mean(dim=2, keepdim=True).expand_as(h_chunks)
        anomaly = (
            1.0
            - F.cosine_similarity(h_chunks, chunk_mean, dim=-1).reshape(B, N, 1)
        )
        raw_salience = self.salience_scorer(
            torch.cat([h_tok, raw_emb, anomaly], dim=-1)
        ).squeeze(-1)

        salience_logits = self.span_conv(raw_salience.unsqueeze(1)).squeeze(1)

        valid_tok = (
            (raw_token_ids != self.cfg.pad_token_id) & (raw_token_ids >= 256)
        ).float()
        masked_salience = salience_logits * valid_tok + (1.0 - valid_tok) * (
            -1e4
        )
        chunk_mask = torch.full_like(w_slice, -1e4)
        for m in range(M):
            chunk_mask[:, m, m * L_c : (m + 1) * L_c] = 0.0
        idea_tok_scores = masked_salience.unsqueeze(1) + chunk_mask

        if self.training:
            gumbel = -torch.log(
                -torch.log(
                    torch.rand_like(idea_tok_scores).clamp(1e-6, 1 - 1e-6)
                )
            )
            noisy_scores = idea_tok_scores + 0.05 * gumbel
        else:
            noisy_scores = idea_tok_scores

        _, raw_top_idx = torch.topk(noisy_scores, k=Kh, dim=-1)
        top_idx, _ = torch.sort(raw_top_idx, dim=-1)

        e_high_ids = torch.gather(
            raw_token_ids.unsqueeze(1).expand(B, M, N), dim=-1, index=top_idx
        )
        e_high_vecs = torch.gather(
            h_tok.unsqueeze(1).expand(B, M, N, self.cfg.d_model),
            dim=2,
            index=top_idx.unsqueeze(-1).expand(B, M, Kh, self.cfg.d_model),
        )
        picked_w = torch.gather(
            F.softmax(idea_tok_scores, dim=-1), dim=-1, index=top_idx
        ).unsqueeze(-1)
        e_high_vecs = (
            e_high_vecs * (picked_w - picked_w.detach() + 1.0)
            + self.slot_pos_emb
        )
        return e_high_vecs, e_high_ids, salience_logits


class EpisodicMemoryAndRouter(nn.Module):
    def __init__(self, cfg: ScaleConfig):
        super().__init__()
        self.cfg = cfg
        self.key_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.query_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.mem_compress = nn.Linear(cfg.d_model * 2, cfg.d_model)
        self.idea_rank_emb = nn.Parameter(
            torch.randn(1, cfg.top_k_mem, 1, cfg.d_model) * 0.02
        )

    def retrieve(self, query_state, s_bank, e_vecs_bank, e_ids_bank):
        B, M_curr, D = s_bank.shape
        k = min(self.cfg.top_k_mem, M_curr)
        scores = torch.matmul(
            self.query_proj(query_state).unsqueeze(1),
            self.key_proj(s_bank).transpose(-1, -2),
        ).squeeze(1) / math.sqrt(D)
        top_vals, top_idx = torch.topk(F.softmax(scores, dim=-1), k=k, dim=-1)
        top_weights = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp(
            min=1e-6
        )

        ret_s = torch.gather(
            s_bank, dim=1, index=top_idx.unsqueeze(-1).expand(B, k, D)
        )
        pooled_s = (ret_s * top_weights.unsqueeze(-1)).sum(dim=1)

        Kh = e_ids_bank.shape[-1]
        ret_high_vecs = torch.gather(
            e_vecs_bank,
            dim=1,
            index=top_idx.unsqueeze(-1).unsqueeze(-1).expand(B, k, Kh, D),
        )
        w_st = (top_weights - top_weights.detach() + 1.0).view(B, k, 1, 1)
        ret_high_vecs = (
            ret_high_vecs * w_st + self.idea_rank_emb[:, :k, :, :]
        )
        ret_high_ids = torch.gather(
            e_ids_bank, dim=1, index=top_idx.unsqueeze(-1).expand(B, k, Kh)
        )

        flat_high_vecs = ret_high_vecs.reshape(B, k * Kh, D)
        flat_high_ids = ret_high_ids.reshape(B, k * Kh)
        mem_context = self.mem_compress(
            torch.cat([pooled_s, flat_high_vecs.mean(dim=1)], dim=-1)
        )
        return mem_context, flat_high_vecs, flat_high_ids, scores, ret_s


# === [SECTION 4: HRM CORE WITH 3-STAGE MULTI-HORIZON JEPA PLANNER] ============
class HRMCore(nn.Module):
    def __init__(self, cfg: ScaleConfig):
        super().__init__()
        self.cfg = cfg
        self.h_ssm = DeltaMatrixSSMBlock(cfg.d_model, cfg.n_heads)
        self.l_think_ssm = DeltaMatrixSSMBlock(cfg.d_model, cfg.n_heads)
        self.halt_head = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, 1),
        )
        self.pred_state = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_state),
        )
        self.pred_domain = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_domain),
        )
        self.plan_proj = nn.Linear(
            cfg.d_model, cfg.plan_horizons * cfg.d_model
        )

    def step_idea(
        self, s_curr, h_state, router, s_bank, e_vecs_bank, e_ids_bank
    ):
        B, D = s_curr.shape
        h_out, h_state = self.h_ssm(s_curr, h_state)
        l_query = h_out
        l_mat_state = None
        T_max = self.cfg.hrm_max_ticks
        cum_halt = torch.zeros(B, 1, device=s_curr.device)
        still_running = torch.ones(B, 1, device=s_curr.device)
        expected_ticks = torch.zeros(B, 1, device=s_curr.device)
        l_accum = torch.zeros(B, D, device=s_curr.device)

        for tau in range(T_max):
            mem_ctx, ret_high_vecs, ret_high_ids, router_scores, ret_s = (
                router.retrieve(l_query, s_bank, e_vecs_bank, e_ids_bank)
            )
            l_out, l_mat_state = self.l_think_ssm(h_out + mem_ctx, l_mat_state)
            l_query = l_out
            raw_p = torch.sigmoid(
                self.halt_head(torch.cat([l_out, h_out], dim=-1))
            )
            p_halt = (
                (1.0 - cum_halt)
                if tau == T_max - 1
                else torch.minimum(raw_p * still_running, 1.0 - cum_halt)
            )
            l_accum = l_accum + p_halt * l_out
            expected_ticks = expected_ticks + still_running * (1.0 - raw_p)
            cum_halt = cum_halt + p_halt
            still_running = (cum_halt < self.cfg.act_threshold).float()
            if not self.training and still_running.sum() == 0:
                break

        pred_s_state = multi_sphere_norm(
            self.pred_state(l_accum), sphere_dim=16
        )
        pred_s_domain = multi_sphere_norm(
            self.pred_domain(l_accum), sphere_dim=16
        )
        pred_s = torch.cat([pred_s_state, pred_s_domain], dim=-1)

        raw_plan = self.plan_proj(l_accum).view(
            B, self.cfg.plan_horizons, D
        )
        plan_state = multi_sphere_norm(
            raw_plan[:, :, : self.cfg.d_state], sphere_dim=16
        )
        plan_dom = multi_sphere_norm(
            raw_plan[:, :, self.cfg.d_state :], sphere_dim=16
        )
        pred_plan = torch.cat([plan_state, plan_dom], dim=-1)

        return (
            pred_s,
            pred_s_state,
            pred_s_domain,
            pred_plan,
            plan_state,
            h_state,
            ret_high_vecs,
            ret_high_ids,
            router_scores,
            1.0 + expected_ticks.mean(),
            ret_s,
        )


# === [SECTION 5: CALIBRATED WORKSPACE VERBALIZER (FIXED LOGIT SCALING)] =======
class VerbalizerWithPointerGate(nn.Module):
    """Fixed in v7.2:

    Uses a properly scaled `vocab_head = nn.Linear(d_model, vocab_size)`
    (initialized with
    std=0.02) so Step 0 NLL starts at ~10.8 instead of 130.0 and `p_vocab` never
    saturates!
    """

    def __init__(self, cfg: ScaleConfig):
        super().__init__()
        self.cfg = cfg
        self.dec_layers = nn.ModuleList(
            [
                DeltaMatrixSSMBlock(cfg.d_model, cfg.n_heads)
                for _ in range(cfg.dec_layers)
            ]
        )
        self.ws_ln = nn.LayerNorm(cfg.d_model)
        self.ws_q = nn.Linear(cfg.d_model, cfg.d_model)
        self.ws_k = nn.Linear(cfg.d_model, cfg.d_model)
        self.ws_v = nn.Linear(cfg.d_model, cfg.d_model)
        self.vocab_head = nn.Linear(cfg.d_model, cfg.vocab_size)
        nn.init.normal_(self.vocab_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.vocab_head.bias)
        self.ptr_query = nn.Linear(cfg.d_model, cfg.d_model)
        self.copy_gate = nn.Linear(cfg.d_model * 2, 1)

    def forward(
        self,
        workspace_slots,
        shifted_in_emb,
        ret_high_vecs,
        ret_high_ids,
        used_slot_mask=None,
    ):
        B, L_c, D = shifted_in_emb.shape
        ws_summary = workspace_slots[:, 0, :]
        dec_h = shifted_in_emb + ws_summary.unsqueeze(1)
        for layer in self.dec_layers:
            dec_h, _ = layer(dec_h, None)

        ws_norm = self.ws_ln(workspace_slots)
        q_ws = self.ws_q(dec_h)
        k_ws = self.ws_k(ws_norm)
        v_ws = self.ws_v(ws_norm)
        ws_attn = F.softmax(
            torch.matmul(q_ws, k_ws.transpose(-1, -2)) / math.sqrt(D), dim=-1
        )
        ws_ctx = torch.matmul(ws_attn, v_ws)

        ptr_scores = torch.matmul(
            self.ptr_query(dec_h), ret_high_vecs.transpose(-1, -2)
        ) / math.sqrt(D)
        if used_slot_mask is not None:
            ptr_scores = ptr_scores - used_slot_mask.unsqueeze(1) * 1e4
        p_ptr = F.softmax(ptr_scores, dim=-1)
        ctx_ptr = torch.matmul(p_ptr, ret_high_vecs)

        enriched_h = dec_h + ws_ctx + 0.5 * ctx_ptr
        p_vocab = F.softmax(self.vocab_head(enriched_h), dim=-1)

        p_copy = torch.sigmoid(
            self.copy_gate(torch.cat([enriched_h, ctx_ptr], dim=-1))
        ).clamp(1e-6, 1.0 - 1e-6)

        final_probs = ((1.0 - p_copy) * p_vocab).scatter_add(
            dim=-1,
            index=ret_high_ids.unsqueeze(1).expand(
                B, L_c, ret_high_ids.size(-1)
            ),
            src=p_copy * p_ptr,
        )
        return final_probs, p_copy, p_ptr

    @torch.no_grad()
    def generate_autoregressive(
        self,
        workspace_slots,
        tok_emb_layer,
        ret_high_vecs,
        ret_high_ids,
        max_len=16,
        return_telemetry: bool = False,
    ):
        B, device = workspace_slots.size(0), workspace_slots.device
        cur_ids = torch.full(
            (B, 1), self.cfg.bos_token_id, dtype=torch.long, device=device
        )
        used_slot_mask = torch.zeros_like(ret_high_ids, dtype=torch.float)
        generated_ids, copy_gates = [], []
        for _ in range(max_len):
            probs, p_copy, _ = self.forward(
                workspace_slots,
                tok_emb_layer(cur_ids),
                ret_high_vecs,
                ret_high_ids,
                used_slot_mask,
            )
            if generated_ids:
                prev_stack = torch.cat(generated_ids, dim=1)
                probs = probs.clone()
                probs[:, -1, :].scatter_(
                    1,
                    prev_stack,
                    probs[:, -1, :].gather(1, prev_stack) * 0.4,
                )

            next_id = probs[:, -1, :].argmax(dim=-1, keepdim=True)
            gate_step = p_copy[:, -1, :]
            copied_now = (next_id == ret_high_ids).float() * (
                gate_step > 0.5
            ).float()
            used_slot_mask = (used_slot_mask + copied_now).clamp(max=1.0)
            generated_ids.append(next_id)
            copy_gates.append(gate_step)
            cur_ids = torch.cat([cur_ids, next_id], dim=1)

        gen_out = torch.cat(generated_ids, dim=1)
        if return_telemetry:
            return gen_out, torch.cat(copy_gates, dim=1)
        return gen_out


def contrastive_hypersphere_loss(z_a, z_b, temp=0.1):
    if z_a.size(0) < 2:
        return (1.0 - F.cosine_similarity(z_a, z_b, dim=-1)).mean()
    B = z_a.size(0)
    z_a_n = F.normalize(z_a, p=2, dim=-1)
    z_b_n = F.normalize(z_b, p=2, dim=-1)
    logits = torch.matmul(z_a_n, z_b_n.transpose(0, 1)) / temp
    labels = torch.arange(B, device=z_a.device)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.transpose(0, 1), labels)
    )


def compute_self_supervised_tfidf_targets(input_ids, cfg: ScaleConfig):
    B, N = input_ids.shape
    M, L_c = cfg.num_ideas, cfg.tokens_per_idea
    chunks = input_ids.view(B, M, L_c)
    chunk_freq = (
        (input_ids.view(B, N, 1, 1) == chunks.unsqueeze(1))
        .any(-1)
        .float()
        .sum(-1)
    )
    batch_doc_freq = (
        (input_ids.view(B, N, 1, 1) == input_ids.view(1, 1, B, N))
        .any(-1)
        .float()
        .sum(-1)
    )
    not_pad = (input_ids != cfg.pad_token_id).float()
    rec = (
        (chunk_freq >= 2.0).float()
        * (batch_doc_freq <= 3.0).float()
        * (input_ids >= 1000).float()
    )
    uniq = (batch_doc_freq <= 2.0).float() * (input_ids >= 2000).float()
    return torch.clamp(rec + 0.5 * uniq, max=1.0) * not_pad


class MultimodalSSMHrmJepa(nn.Module):
    def __init__(self, cfg: ScaleConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        self.text_slicer = TextIdeaSlicer(cfg)
        self.vis_slicer = VisualIdeaSlicer(cfg)
        self.encoder = FactorizedGistAndHighlighter(cfg)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.router = EpisodicMemoryAndRouter(cfg)
        self.hrm = HRMCore(cfg)
        self.verbalizer = VerbalizerWithPointerGate(cfg)

    @torch.no_grad()
    def update_ema(self):
        m = self.cfg.ema_decay
        for p_on, p_tgt in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            p_tgt.data.mul_(m).add_(p_on.data, alpha=1.0 - m)

    def forward(
        self, input_ids, vis_patches, has_vision_mask, ptr_aux_scale: float = 0.4
    ):
        B, N = input_ids.shape
        M, L_c = self.cfg.num_ideas, self.cfg.tokens_per_idea
        raw_emb = self.tok_emb(input_ids)

        ctx_emb = raw_emb[:, : 3 * L_c, :]
        ans_emb = raw_emb[:, 3 * L_c :, :]
        txt_chunks_ctx, h_tok_ctx, w_slice_ctx, ratio_loss = self.text_slicer(
            ctx_emb, 3
        )
        txt_chunks_ans, h_tok_ans, w_slice_ans, _ = self.text_slicer(ans_emb, 1)

        txt_chunks = torch.cat([txt_chunks_ctx, txt_chunks_ans], dim=1)
        h_tok = torch.cat([h_tok_ctx, h_tok_ans], dim=1)
        w_slice = torch.zeros(B, M, N, device=raw_emb.device, dtype=raw_emb.dtype)
        w_slice[:, :3, : 3 * L_c] = w_slice_ctx
        w_slice[:, 3:, 3 * L_c :] = w_slice_ans

        vis_chunks, _, _ = self.vis_slicer(vis_patches, has_vision_mask, B, M)

        _, s_txt_state, s_txt_dom = self.encoder.encode_gist(txt_chunks)
        _, s_vis_state, _ = self.encoder.encode_gist(vis_chunks)
        s_multi = torch.cat(
            [
                multi_sphere_norm(s_txt_state + s_vis_state, sphere_dim=16),
                s_txt_dom,
            ],
            dim=-1,
        )
        e_high_vecs, e_high_ids, salience_logits = (
            self.encoder.highlight_tokens(h_tok, raw_emb, w_slice, input_ids)
        )
        highlight_loss = F.binary_cross_entropy_with_logits(
            salience_logits,
            compute_self_supervised_tfidf_targets(input_ids, self.cfg),
            pos_weight=torch.tensor([10.0], device=input_ids.device),
        )

        with torch.no_grad():
            _, tgt_txt_state, tgt_txt_dom = self.target_encoder.encode_gist(
                txt_chunks
            )
            _, tgt_vis_state, _ = self.target_encoder.encode_gist(vis_chunks)
            target_multi_state = multi_sphere_norm(
                tgt_txt_state + tgt_vis_state, sphere_dim=16
            )
            sub_spans = torch.stack(
                [
                    h_tok_ans[:, 0:5, :].mean(dim=1),
                    h_tok_ans[:, 5:10, :].mean(dim=1),
                    h_tok_ans[:, 10:16, :].mean(dim=1),
                ],
                dim=1,
            )
            _, tgt_sub_states, _ = self.target_encoder.encode_gist(sub_spans)

        vis_idx = has_vision_mask.nonzero(as_tuple=True)[0]
        anchor_loss = (
            contrastive_hypersphere_loss(
                s_txt_state[vis_idx].mean(1),
                s_vis_state[vis_idx].mean(1),
                self.cfg.jepa_temp,
            )
            if len(vis_idx) >= 2
            else torch.tensor(0.0, device=input_ids.device)
        )

        h_state, jepa_loss, act_role_loss = None, 0.0, 0.0
        target_ticks = [2.0, 2.0, 4.5]
        for i in range(M - 1):
            mem_end = i if i > 0 else 1
            (
                pred_s,
                pred_state,
                pred_dom,
                pred_plan,
                plan_state,
                h_state,
                ret_vecs,
                ret_ids,
                router_scores,
                avg_t,
                ret_s,
            ) = self.hrm.step_idea(
                s_multi[:, i, :],
                h_state,
                self.router,
                s_multi[:, :mem_end, :],
                e_high_vecs[:, :mem_end, :, :],
                e_high_ids[:, :mem_end, :],
            )
            act_role_loss += (avg_t - target_ticks[i]) ** 2
            jepa_loss += contrastive_hypersphere_loss(
                pred_state, target_multi_state[:, i + 1, :], self.cfg.jepa_temp
            ) + 0.3 * (
                1.0
                - F.cosine_similarity(
                    pred_dom, tgt_txt_dom[:, i + 1, :], dim=-1
                ).mean()
            )

        plan_jepa_loss = 0.0
        for s_idx in range(self.cfg.plan_horizons):
            plan_jepa_loss += contrastive_hypersphere_loss(
                plan_state[:, s_idx, :],
                tgt_sub_states[:, s_idx, :],
                self.cfg.jepa_temp,
            )
        plan_jepa_loss = plan_jepa_loss / float(self.cfg.plan_horizons)

        ans_target_ids = input_ids[:, 3 * L_c : 4 * L_c]
        bos_emb = self.tok_emb(
            torch.full((B, 1), self.cfg.bos_token_id, device=input_ids.device)
        )
        shifted_in_emb = torch.cat(
            [bos_emb, raw_emb[:, 3 * L_c : 4 * L_c - 1, :]], dim=1
        )

        is_pad = ans_target_ids == self.cfg.pad_token_id
        prev_is_pad = F.pad(is_pad[:, :-1], (1, 0), value=False)
        ans_labels = ans_target_ids.masked_fill(prev_is_pad, -100)

        q_turn_tokens = h_tok_ctx[:, 2 * L_c : 3 * L_c, :]
        workspace_slots = torch.cat(
            [
                pred_s.unsqueeze(1),
                pred_plan,
                ret_s,
                vis_chunks[:, :3, :],
                ret_vecs,
                q_turn_tokens,
            ],
            dim=1,
        )

        probs, p_copy, p_ptr = self.verbalizer(
            workspace_slots, shifted_in_emb, ret_vecs, ret_ids
        )

        answer_nll = F.nll_loss(
            torch.log(probs.clamp(min=1e-8)).reshape(-1, self.cfg.vocab_size),
            ans_labels.reshape(-1),
            ignore_index=-100,
        )

        matches = (
            ans_target_ids.unsqueeze(-1) == ret_ids.unsqueeze(1)
        ).float() * (~is_pad).unsqueeze(-1).float()
        has_match = matches.max(dim=-1, keepdim=True)[0]
        active_pos = (~prev_is_pad).unsqueeze(-1).float()
        ptr_aux = (
            F.binary_cross_entropy(p_copy, has_match, reduction="none")
            * active_pos
        ).sum() / active_pos.sum().clamp(min=1.0)
        match_norm = matches / matches.sum(dim=-1, keepdim=True).clamp(
            min=1e-6
        )
        ptr_aux += (
            1.5
            * (
                -(match_norm * torch.log(p_ptr.clamp(min=1e-8))).sum(-1)
                * has_match.squeeze(-1)
            ).mean()
        )

        bank_matches = (
            (
                (
                    ans_target_ids.view(B, 1, 1, L_c)
                    == e_high_ids[:, :2, :].unsqueeze(-1)
                )
                & (~is_pad).view(B, 1, 1, L_c)
            )
            .any(-1)
            .any(-1)
            .float()
        )
        if bank_matches.sum() > 0:
            ptr_aux += F.binary_cross_entropy(
                F.softmax(router_scores, dim=-1).clamp(1e-6, 1.0 - 1e-6),
                bank_matches.clamp(max=1.0),
            )

        steps_c = float(M - 1)
        total_loss = (
            0.20 * (jepa_loss / steps_c)
            + 0.15 * plan_jepa_loss
            + answer_nll
            + ptr_aux_scale * ptr_aux
            + 0.35 * highlight_loss
            + 0.15 * anchor_loss
            + 0.04 * (act_role_loss / steps_c)
            + 0.02 * ratio_loss
        )
        return total_loss, answer_nll

    @torch.no_grad()
    def generate_reply(
        self,
        context_tokens,
        vis_patches=None,
        has_vision_mask=None,
        return_telemetry: bool = False,
    ):
        B, N_ctx = context_tokens.shape
        L_c = self.cfg.tokens_per_idea
        M_ctx = N_ctx // L_c
        raw_emb = self.tok_emb(context_tokens)
        txt_chunks, h_tok, w_slice, _ = self.text_slicer(raw_emb, M_ctx)
        vis_chunks, _, _ = self.vis_slicer(
            vis_patches, has_vision_mask, B, M_ctx
        )
        _, s_txt_state, s_txt_dom = self.encoder.encode_gist(txt_chunks)
        _, s_vis_state, _ = self.encoder.encode_gist(vis_chunks)
        s_multi = torch.cat(
            [
                multi_sphere_norm(s_txt_state + s_vis_state, sphere_dim=16),
                s_txt_dom,
            ],
            dim=-1,
        )
        e_high_vecs, e_high_ids, _ = self.encoder.highlight_tokens(
            h_tok, raw_emb, w_slice, context_tokens
        )
        h_state = None
        for i in range(M_ctx):
            mem_end = i if i > 0 else 1
            (
                pred_s,
                _,
                _,
                pred_plan,
                _,
                h_state,
                ret_vecs,
                ret_ids,
                _,
                _,
                ret_s,
            ) = self.hrm.step_idea(
                s_multi[:, i, :],
                h_state,
                self.router,
                s_multi[:, :mem_end, :],
                e_high_vecs[:, :mem_end, :, :],
                e_high_ids[:, :mem_end, :],
            )
        q_turn_tokens = h_tok[:, (M_ctx - 1) * L_c : M_ctx * L_c, :]
        workspace_slots = torch.cat(
            [
                pred_s.unsqueeze(1),
                pred_plan,
                ret_s,
                vis_chunks[:, :3, :],
                ret_vecs,
                q_turn_tokens,
            ],
            dim=1,
        )
        if return_telemetry:
            gen_ids, gates = self.verbalizer.generate_autoregressive(
                workspace_slots,
                self.tok_emb,
                ret_vecs,
                ret_ids,
                max_len=L_c,
                return_telemetry=True,
            )
            return gen_ids, gates, ret_ids
        return self.verbalizer.generate_autoregressive(
            workspace_slots, self.tok_emb, ret_vecs, ret_ids, max_len=L_c
        )

    @torch.no_grad()
    def step_incremental_turn(
        self,
        new_turn_tokens,
        h_state,
        s_bank,
        e_vecs_bank,
        e_ids_bank,
        vis_chunks_3,
    ):
        B, L_c = new_turn_tokens.shape
        raw_emb = self.tok_emb(new_turn_tokens)
        txt_chunk, h_tok, w_slice, _ = self.text_slicer(raw_emb, 1)
        _, s_txt_state, s_txt_dom = self.encoder.encode_gist(txt_chunk)
        _, s_vis_state, _ = self.encoder.encode_gist(vis_chunks_3[:, :1, :])
        s_curr = torch.cat(
            [
                multi_sphere_norm(s_txt_state + s_vis_state, sphere_dim=16),
                s_txt_dom,
            ],
            dim=-1,
        ).squeeze(1)
        e_vecs, e_ids, _ = self.encoder.highlight_tokens(
            h_tok, raw_emb, w_slice, new_turn_tokens
        )

        pred_s, _, _, pred_plan, _, h_state, ret_vecs, ret_ids, _, _, ret_s = (
            self.hrm.step_idea(
                s_curr, h_state, self.router, s_bank, e_vecs_bank, e_ids_bank
            )
        )
        workspace_slots = torch.cat(
            [
                pred_s.unsqueeze(1),
                pred_plan,
                ret_s,
                vis_chunks_3,
                ret_vecs,
                h_tok,
            ],
            dim=1,
        )
        return self.verbalizer.generate_autoregressive(
            workspace_slots, self.tok_emb, ret_vecs, ret_ids, max_len=L_c
        )


# === [SECTION 6: CALIBRATED ISO-PARAMETER CLASSICAL TRANSFORMER] ==============
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_hidden: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, d_model),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        h = self.ln1(x)
        attn_out, _ = self.attn(
            h, h, h, attn_mask=attn_mask, need_weights=False
        )
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class ClassicalTransformerLM(nn.Module):
    def __init__(self, cfg: ScaleConfig, n_layers: int, mlp_hidden: int):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        self.pos_emb = nn.Embedding(2560, cfg.d_model)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        self.vis_proj = nn.Linear(cfg.vis_dim, cfg.d_model)
        self.null_vis_patches = nn.Parameter(
            torch.randn(1, cfg.vis_patches, cfg.vis_dim) * 0.02
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(cfg.d_model, cfg.n_heads, mlp_hidden)
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lm_head.bias)

    def _forward_seq(self, input_ids, vis_patches=None, has_vision_mask=None):
        B, N = input_ids.shape
        if vis_patches is None:
            vis_patches = self.null_vis_patches.expand(B, -1, -1)
        elif has_vision_mask is not None:
            mask = has_vision_mask.view(-1, 1, 1).float()
            vis_patches = (
                mask * vis_patches
                + (1.0 - mask) * self.null_vis_patches.expand(B, -1, -1)
            )
        vis_toks = self.vis_proj(vis_patches)
        txt_toks = self.tok_emb(input_ids)
        x = torch.cat([vis_toks, txt_toks], dim=1)
        L_tot = x.size(1)
        pos = torch.arange(L_tot, device=x.device).unsqueeze(0)
        x = x + self.pos_emb(pos)
        causal_mask = torch.triu(
            torch.full((L_tot, L_tot), float("-inf"), device=x.device),
            diagonal=1,
        )
        for blk in self.blocks:
            x = blk(x, causal_mask)
        return self.lm_head(self.ln_f(x))

    def forward(
        self, input_ids, vis_patches, has_vision_mask, ptr_aux_scale: float = 0.0
    ):
        L_c = self.cfg.tokens_per_idea
        P = self.cfg.vis_patches
        logits = self._forward_seq(input_ids, vis_patches, has_vision_mask)
        ans_logits = logits[:, P + 3 * L_c - 1 : P + 4 * L_c - 1, :]
        ans_target_ids = input_ids[:, 3 * L_c : 4 * L_c]
        is_pad = ans_target_ids == self.cfg.pad_token_id
        prev_is_pad = F.pad(is_pad[:, :-1], (1, 0), value=False)
        ans_labels = ans_target_ids.masked_fill(prev_is_pad, -100)
        answer_nll = F.cross_entropy(
            ans_logits.reshape(-1, self.cfg.vocab_size),
            ans_labels.reshape(-1),
            ignore_index=-100,
        )
        return answer_nll, answer_nll

    @torch.no_grad()
    def generate_reply(
        self, context_tokens, vis_patches=None, has_vision_mask=None
    ):
        L_c = self.cfg.tokens_per_idea
        cur_ids = context_tokens
        gen = []
        for _ in range(L_c):
            logits = self._forward_seq(cur_ids, vis_patches, has_vision_mask)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            if gen:
                prev_stack = torch.cat(gen, dim=1)
                probs.scatter_(
                    1, prev_stack, probs.gather(1, prev_stack) * 0.4
                )
            next_id = probs.argmax(dim=-1, keepdim=True)
            gen.append(next_id)
            cur_ids = torch.cat([cur_ids, next_id], dim=1)
        return torch.cat(gen, dim=1)


def count_core_params(model: nn.Module) -> int:
    total = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "tok_emb" in name or "vocab_head" in name or "lm_head" in name:
            continue
        total += p.numel()
    return total


def build_iso_param_transformer(
    cfg: ScaleConfig, target_core_params: int, n_layers: int
) -> ClassicalTransformerLM:
    dummy = ClassicalTransformerLM(cfg, n_layers=n_layers, mlp_hidden=1)
    p_at_1 = count_core_params(dummy)
    delta_per_unit = n_layers * (2 * cfg.d_model + 1)
    exact_mlp_hidden = max(
        32, int(round(1 + (target_core_params - p_at_1) / float(delta_per_unit)))
    )
    return ClassicalTransformerLM(
        cfg, n_layers=n_layers, mlp_hidden=exact_mlp_hidden
    )


# === [SECTION 7: TINYSTORIES + IMAGES ONLY DATASET (LOADS CACHED IN <1s)] =====
class TinyStoriesAndImagesDataset:
    def __init__(self, work_dir: str, cache_size: int = 5000):
        vis_cache_path = os.path.join(
            work_dir, f"flickr8k_mobilenet_cache_{cache_size}.pt"
        )
        stories_cache_path = os.path.join(
            work_dir, f"tinystories_cache_{cache_size}.pt"
        )

        cached_vis = torch.load(vis_cache_path, map_location="cpu")
        stories_toks = torch.load(stories_cache_path, map_location="cpu")

        n_v, n_s = len(cached_vis["captions"]), len(stories_toks)
        v_cut, s_cut = int(0.9 * n_v), int(0.9 * n_s)

        self.splits = {
            "train": {
                "vis_f": cached_vis["vis_features"][:v_cut],
                "caps": cached_vis["captions"][:v_cut],
                "stories": stories_toks[:s_cut],
            },
            "val": {
                "vis_f": cached_vis["vis_features"][v_cut:],
                "caps": cached_vis["captions"][v_cut:],
                "stories": stories_toks[s_cut:],
            },
        }
        print(
            f"Loaded TinyStories + Images in <1s | Train: {s_cut+v_cut:,} | Unseen Val: {(n_s-s_cut)+(n_v-v_cut):,}"
        )

    def _pack(self, ids, L_c=16):
        pad = tokenizer.pad_token_id
        return (ids[: L_c - 1] + [pad] * L_c)[:L_c]

    def get_batch(
        self, split: str, batch_size: int, gen: torch.Generator = None
    ):
        sp = self.splits[split]
        vis_list, token_rows, has_vis = [], [], []
        for b_i in range(batch_size):
            if b_i % 2 == 0:
                idx = int(
                    torch.randint(
                        0, len(sp["stories"]), (1,), generator=gen
                    ).item()
                )
                vis_list.append(torch.zeros(16, 576))
                has_vis.append(0.0)
                token_rows.append(sp["stories"][idx])
            else:
                idx = int(
                    torch.randint(
                        0, len(sp["caps"]), (1,), generator=gen
                    ).item()
                )
                vis_list.append(sp["vis_f"][idx])
                has_vis.append(1.0)
                cap_ids = fast_encode(sp["caps"][idx], max_len=32)
                p0 = self._pack(
                    fast_encode("Once upon a time in this picture,")
                    + cap_ids[:6]
                )
                p1 = self._pack(
                    fast_encode("Looking closely at the scene, we see")
                    + cap_ids[:6]
                )
                p2 = self._pack(
                    fast_encode("Describe what is happening in the picture:")
                )
                p3 = self._pack(cap_ids)
                token_rows.append(p0 + p1 + p2 + p3)

        return (
            torch.tensor(token_rows, dtype=torch.long, device=DEVICE),
            torch.stack(vis_list).to(DEVICE),
            torch.tensor(has_vis, dtype=torch.float, device=DEVICE),
        )


@torch.no_grad()
def evaluate_on_fixed_val(model: nn.Module, val_batches: list):
    model.eval()
    nlls, recalls = [], []
    pad_id = tokenizer.pad_token_id
    for tokens, vis_patches, vis_mask in val_batches:
        _, ans_nll = model(tokens, vis_patches, vis_mask, ptr_aux_scale=0.0)
        nlls.append(ans_nll.item())

        gen_ids = model.generate_reply(tokens[:, :48], vis_patches, vis_mask)
        ctx_ids = tokens[:, :48]
        ans_true = tokens[:, 48:64]

        for b in range(tokens.size(0)):
            c_set = {
                int(t) for t in ctx_ids[b].tolist() if t >= 1000 and t != pad_id
            }
            a_set = {
                int(t)
                for t in ans_true[b].tolist()
                if t >= 1000 and t != pad_id
            }
            shared = c_set & a_set
            if shared:
                g_set = {int(t) for t in gen_ids[b].tolist()}
                recalls.append(len(shared & g_set) / float(len(shared)))

    model.train()
    return float(np.mean(nlls)), (
        float(np.mean(recalls)) * 100.0 if recalls else 0.0
    )


def decode_clean(tok_list: list, eos_id: int) -> str:
    out = []
    for t in tok_list:
        if t == eos_id:
            break
        out.append(t)
    return tokenizer.decode(out).replace("\n", " ").strip()


def run_v7_2_tinystories_experiment():
    dataset = TinyStoriesAndImagesDataset(WORK_DIR)

    val_gen = torch.Generator().manual_seed(12345)
    val_batches = [dataset.get_batch("val", 24, val_gen) for _ in range(8)]

    scales = [
        ("Scale 1 (Tiny)", 64, 2, 1, 2, 2, 2.0e-3),
        ("Scale 2 (Small)", 128, 4, 2, 4, 3, 1.6e-3),
        ("Scale 3 (Medium)", 256, 8, 3, 6, 4, 1.2e-3),
    ]

    results = {"SSM-HRM-JEPA": [], "Transformer": []}
    medium_curves = {"steps": [], "SSM-HRM-JEPA": [], "Transformer": []}
    top_d = scales[-1][1]
    top_hl = scales[-1][4]
    top_heads = scales[-1][2]

    print("=" * 94)
    print(
        f"RUNNING v7.2 CALIBRATED TINYSTORIES + IMAGES ISO-PARAM BENCHMARK ON {DEVICE.upper()}"
    )
    print("=" * 94)

    for (
        label,
        d_model,
        n_heads,
        dec_layers,
        hl_k,
        tfm_layers,
        lr,
    ) in scales:
        cfg = ScaleConfig(
            d_model=d_model,
            n_heads=n_heads,
            dec_layers=dec_layers,
            highlight_k=hl_k,
            lr=lr,
        )

        torch.manual_seed(42)
        ssm_model = MultimodalSSMHrmJepa(cfg).to(DEVICE)
        target_core = count_core_params(ssm_model)

        torch.manual_seed(42)
        tfm_model = build_iso_param_transformer(
            cfg, target_core_params=target_core, n_layers=tfm_layers
        ).to(DEVICE)

        for arch_name, model in [
            ("SSM-HRM-JEPA", ssm_model),
            ("Transformer", tfm_model),
        ]:
            core_p = count_core_params(model)
            tot_p = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            opt = torch.optim.AdamW(
                model.parameters(), lr=cfg.lr, weight_decay=1e-4
            )
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=cfg.steps, eta_min=cfg.lr * 0.1
            )

            train_gen = torch.Generator().manual_seed(999)
            t0 = time.time()
            model.train()

            for step in range(1, cfg.steps + 1):
                tokens, vis_patches, vis_mask = dataset.get_batch(
                    "train", cfg.batch_size, train_gen
                )
                progress = step / float(cfg.steps)
                ptr_scale = 0.02 + 0.35 * (
                    0.5 * (1.0 + math.cos(math.pi * progress))
                )

                opt.zero_grad()
                loss, _ = model(
                    tokens, vis_patches, vis_mask, ptr_aux_scale=ptr_scale
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                if arch_name == "SSM-HRM-JEPA":
                    model.update_ema()

                if (
                    step == 1
                    or step % cfg.eval_every == 0
                    or step == cfg.steps
                ):
                    val_nll, val_rec = evaluate_on_fixed_val(
                        model, val_batches
                    )
                    if d_model == top_d:
                        if arch_name == "SSM-HRM-JEPA":
                            medium_curves["steps"].append(step)
                        medium_curves[arch_name].append(val_nll)
                    if step == cfg.steps:
                        dt = time.time() - t0
                        print(
                            f"[{label:16s}] {arch_name:13s} | Core: {core_p/1e3:8.1f}K "
                            f"(Tot: {tot_p/1e6:5.2f}M) | Val NLL: {val_nll:.3f} | "
                            f"Story/Img Entity Recall: {val_rec:5.1f}% | {dt:.1f}s"
                        )
                        results[arch_name].append(
                            {
                                "label": label,
                                "d_model": d_model,
                                "core_params": core_p,
                                "total_params": tot_p,
                                "val_nll": val_nll,
                                "val_recall": val_rec,
                            }
                        )

            if d_model != top_d:
                del model, opt, sched
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                if arch_name == "SSM-HRM-JEPA":
                    med_ssm = model
                else:
                    med_tfm = model

    # === [SECTION 8: CONTEXT-LENGTH SCALING BENCHMARK (N = 64 .. 2048)] =======
    print(
        f"\nBenchmarking Context-Length Scaling (N = 64 -> 2048 tokens) at Scale 3 (d={top_d})..."
    )
    ctx_lengths = [64, 256, 512, 1024, 2048]
    ctx_metrics = {
        "N": ctx_lengths,
        "ssm_ms": [],
        "tfm_ms": [],
        "ssm_mb": [],
        "tfm_mb": [],
    }

    med_ssm.eval()
    med_tfm.eval()
    dh_top = top_d // top_heads
    with torch.no_grad():
        for N in ctx_lengths:
            M_turns = N // 16
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            s_bank = torch.randn(1, M_turns, top_d, device=DEVICE)
            e_vecs = torch.randn(1, M_turns, top_hl, top_d, device=DEVICE)
            e_ids = torch.randint(
                100, 10000, (1, M_turns, top_hl), device=DEVICE
            )
            new_turn = torch.randint(100, 10000, (1, 16), device=DEVICE)
            h_st = torch.zeros(1, top_heads, dh_top, dh_top, device=DEVICE)
            v_chunks_3 = torch.randn(1, 3, top_d, device=DEVICE)

            _ = med_ssm.step_incremental_turn(
                new_turn, h_st, s_bank, e_vecs, e_ids, v_chunks_3
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                _ = med_ssm.step_incremental_turn(
                    new_turn, h_st, s_bank, e_vecs, e_ids, v_chunks_3
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                ssm_mb = torch.cuda.max_memory_allocated() / (1024**2)
            else:
                ssm_mb = 0.0
            ssm_ms = (time.perf_counter() - t0) / 10.0 * 1000.0

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            tfm_ctx = torch.randint(100, 10000, (1, N), device=DEVICE)
            _ = med_tfm.generate_reply(tfm_ctx)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                tfm_mb = torch.cuda.max_memory_allocated() / (1024**2)
            else:
                tfm_mb = 0.0
            t0 = time.perf_counter()
            for _ in range(10):
                _ = med_tfm.generate_reply(tfm_ctx)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            tfm_ms = (time.perf_counter() - t0) / 10.0 * 1000.0

            ctx_metrics["ssm_ms"].append(ssm_ms)
            ctx_metrics["tfm_ms"].append(tfm_ms)
            ctx_metrics["ssm_mb"].append(ssm_mb)
            ctx_metrics["tfm_mb"].append(tfm_mb)
            print(
                f"  N={N:4d} toks | SSM-HRM Latency: {ssm_ms:6.2f} ms (Peak {ssm_mb:5.1f} MB) | "
                f"Transformer Latency: {tfm_ms:6.2f} ms (Peak {tfm_mb:5.1f} MB)"
            )

    # === [SECTION 9: PLOT 4-PANEL TINYSTORIES + IMAGES SCALING FIGURE] ========
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=140)
    fig.suptitle(
        "TinyStories + Images Scaling Laws: Calibrated SSM-HRM-JEPA (v7.2) vs. Transformer",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    c_ssm, c_tfm = "#0D9488", "#E11D48"

    ax = axes[0, 0]
    for name, color, marker in [
        ("SSM-HRM-JEPA", c_ssm, "o"),
        ("Transformer", c_tfm, "s"),
    ]:
        xs = np.array([r["core_params"] for r in results[name]], dtype=float)
        ys = np.array([r["val_nll"] for r in results[name]], dtype=float)
        slope, intercept = np.polyfit(np.log10(xs), np.log10(ys), 1)
        x_line = np.linspace(xs.min() * 0.85, xs.max() * 1.15, 50)
        y_line = (10**intercept) * (x_line**slope)
        ax.plot(x_line / 1e3, y_line, "--", color=color, alpha=0.55)
        ax.scatter(
            xs / 1e3,
            ys,
            color=color,
            marker=marker,
            s=85,
            zorder=4,
            label=f"{name} ($\\alpha={-slope:.3f}$)",
        )
        for r in results[name]:
            ax.annotate(
                f"{r['core_params']/1e3:.0f}K",
                (r["core_params"] / 1e3, r["val_nll"]),
                textcoords="offset points",
                xytext=(5, 6),
                fontsize=8,
                color=color,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(
        "A. Iso-Parameter Scaling Law on TinyStories+Images (Val NLL ↓)",
        fontweight="bold",
    )
    ax.set_xlabel("Core Reasoning Parameters (Thousands, Log Scale)")
    ax.set_ylabel("Validation Continuation NLL (Nats, Log Scale)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()

    ax = axes[0, 1]
    x_pos = np.arange(len(scales))
    w_bar = 0.34
    rec_ssm = [r["val_recall"] for r in results["SSM-HRM-JEPA"]]
    rec_tfm = [r["val_recall"] for r in results["Transformer"]]
    b1 = ax.bar(
        x_pos - w_bar / 2,
        rec_ssm,
        w_bar,
        color=c_ssm,
        label="SSM-HRM-JEPA (v7.2)",
    )
    b2 = ax.bar(
        x_pos + w_bar / 2,
        rec_tfm,
        w_bar,
        color=c_tfm,
        label="Classical Transformer",
    )
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}%",
                (bar.get_x() + bar.get_width() / 2, h),
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )
    ax.set_xticks(x_pos)
    ax.set_xticklabels([s[0] for s in scales])
    ax.set_ylim(0, 108)
    ax.set_title(
        "B. Character & Object Persistence in Story Generation (↑)",
        fontweight="bold",
    )
    ax.set_ylabel("Unseen Story/Image Entity Recall (%)")
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(
        medium_curves["steps"],
        medium_curves["SSM-HRM-JEPA"],
        "-o",
        color=c_ssm,
        lw=2,
        label="SSM-HRM-JEPA (v7.2 Scale 3)",
    )
    ax.plot(
        medium_curves["steps"],
        medium_curves["Transformer"],
        "-s",
        color=c_tfm,
        lw=2,
        label="Classical Transformer (Scale 3)",
    )
    ax.set_title(
        "C. Training Compute Efficiency (Scale 3 Iso-Parameter)",
        fontweight="bold",
    )
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Unseen Validation NLL (Nats ↓)")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(
        ctx_metrics["N"],
        ctx_metrics["ssm_ms"],
        "-o",
        color=c_ssm,
        lw=2.2,
        label="SSM-HRM-JEPA (O(1) DeltaNet + Top-k Bank)",
    )
    ax.plot(
        ctx_metrics["N"],
        ctx_metrics["tfm_ms"],
        "-s",
        color=c_tfm,
        lw=2.2,
        label="Classical Transformer (O(N²) Attention)",
    )
    ax.set_title(
        "D. Context-Length Scaling: Turn Latency (N = 64 → 2048)",
        fontweight="bold",
    )
    ax.set_xlabel("Story/Chat History Length (Tokens)")
    ax.set_ylabel("Generation Latency per Turn (ms ↓)")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = os.path.join(
        WORK_DIR, "scaling_law_v7_2_tinystories_vs_transformer.png"
    )
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved v7.2 TinyStories+Images Scaling Plot to Drive: {plot_path}")
    plt.show()

    # === [SECTION 10: SIDE-BY-SIDE GENERATIVE SHOWCASE (UNSEEN VALIDATION)] ===
    print("\n" + "=" * 94)
    print(
        "SIDE-BY-SIDE GENERATIVE SHOWCASE ON UNSEEN VALIDATION TINYSTORIES & IMAGES (SCALE 3)"
    )
    print("=" * 94)
    val_tokens, val_vis, val_mask = val_batches[0]
    with torch.no_grad():
        ssm_gen, ssm_gates, ssm_mem = med_ssm.generate_reply(
            val_tokens[:, :48], val_vis, val_mask, return_telemetry=True
        )
        tfm_gen = med_tfm.generate_reply(val_tokens[:, :48], val_vis, val_mask)

    sample_indices = [0, 2, 4, 1, 3, 5]
    for idx in sample_indices:
        kind = (
            "UNSEEN TINYSTORIES CONTINUATION"
            if idx % 2 == 0
            else "UNSEEN IMAGE-TO-TEXT GENERATION"
        )
        prompt_str = decode_clean(
            val_tokens[idx, :48].tolist(), tokenizer.eos_token_id
        )
        gt_str = decode_clean(
            val_tokens[idx, 48:64].tolist(), tokenizer.eos_token_id
        )
        ssm_str = decode_clean(ssm_gen[idx].tolist(), tokenizer.eos_token_id)
        tfm_str = decode_clean(tfm_gen[idx].tolist(), tokenizer.eos_token_id)
        mem_words = [
            tokenizer.decode([tid]).strip()
            for tid in ssm_mem[idx].tolist()[:6]
        ]

        print(f"[{kind} — Sample #{idx}]")
        print(f"  Prompt (48 toks)      : {prompt_str}")
        print(f"  Ground-Truth (16 toks): {gt_str}")
        print(f"  SSM Highlighted Memory: {mem_words}")
        print(f"  SSM-HRM-JEPA (v7.2)   : {ssm_str}")
        print(f"  Classical Transformer : {tfm_str}\n")


if __name__ == "__main__":
    run_v7_2_tinystories_experiment()
