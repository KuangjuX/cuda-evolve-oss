"""Reference implementation for Dynamic Sparse Attention forward pass."""

import torch


def dsa_forward_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.Tensor,
    indices_blk_siz: int,
    scale: float,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    token2batch_q: torch.Tensor = None,
    sliding_window: tuple = (-1, -1),
) -> tuple:
    """Pure PyTorch block-sparse causal attention with GQA and variable-length sequences.

    Uses a two-pass approach per (token, head): first gather all valid KV positions
    and scores, then compute softmax + weighted sum.
    """
    seq_len, n_heads, head_dim = q.shape
    seq_len_kv, n_heads_kv, _ = k.shape
    _, n_heads_block, block_count = block_indices.shape
    kv_group = n_heads // n_heads_kv
    batch_size = len(cu_seqlens_q) - 1

    if token2batch_q is None:
        token2batch_q = torch.repeat_interleave(
            torch.diff(cu_seqlens_q), output_size=seq_len
        )

    out = torch.zeros(seq_len, n_heads, head_dim, device=q.device, dtype=torch.float32)
    lse = torch.full((seq_len, n_heads), float("-inf"), device=q.device, dtype=torch.float32)

    for b in range(batch_size):
        q_start = cu_seqlens_q[b].item()
        q_end = cu_seqlens_q[b + 1].item()
        k_start = cu_seqlens_k[b].item()
        k_end = cu_seqlens_k[b + 1].item()
        sl_q = q_end - q_start
        sl_kv = k_end - k_start

        q_batch = q[q_start:q_end].float()
        k_batch = k[k_start:k_end].float()
        v_batch = v[k_start:k_end].float()

        for qi in range(sl_q):
            global_qi = q_start + qi
            causal_offset = sl_kv - sl_q
            causal_right = qi + causal_offset

            left_bound = 0
            right_bound = causal_right
            if sliding_window[0] != -1:
                left_bound = max(left_bound, qi + causal_offset - sliding_window[0])
            if sliding_window[1] != -1:
                right_bound = min(right_bound, qi + causal_offset + sliding_window[1])

            for h in range(n_heads):
                kv_h = h // kv_group
                hb = h // (kv_group if n_heads_block == n_heads_kv else (n_heads_kv // n_heads_block * kv_group))

                all_scores = []
                all_v = []

                for bc in range(block_count):
                    blk_idx = block_indices[global_qi, min(hb, n_heads_block - 1), bc].item()
                    blk_start = blk_idx * indices_blk_siz
                    blk_end = min(blk_start + indices_blk_siz, sl_kv)

                    for ki in range(blk_start, blk_end):
                        if ki < left_bound or ki > right_bound or ki < 0 or ki >= sl_kv:
                            continue
                        score = (q_batch[qi, h, :] * scale) @ k_batch[ki, kv_h, :]
                        all_scores.append(score.item())
                        all_v.append(v_batch[ki, kv_h, :])

                if len(all_scores) > 0:
                    scores_t = torch.tensor(all_scores, device=q.device, dtype=torch.float32)
                    v_stack = torch.stack(all_v)
                    m = scores_t.max()
                    exp_s = torch.exp(scores_t - m)
                    denom = exp_s.sum()
                    out[global_qi, h, :] = (exp_s.unsqueeze(1) * v_stack).sum(0) / denom
                    lse[global_qi, h] = m + torch.log(denom)

    out = out.to(q.dtype)
    return out, lse
