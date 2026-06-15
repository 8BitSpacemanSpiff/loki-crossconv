import torch


def _greedy_css_batched(M, B):
    """Greedy max-volume row subset for attention-weighted values.

    M: (H, P, d), where H is the number of KV heads after GQA pooling.
    Returns indices of shape (H, B) into P.
    """
    H, P, d = M.shape
    B = min(B, P)
    R = M.clone().float()
    norms = (R * R).sum(-1)
    idx = torch.full((H, B), -1, dtype=torch.long, device=M.device)
    ar = torch.arange(H, device=M.device)
    for t in range(B):
        j = norms.argmax(1)
        idx[:, t] = j
        qv = R[ar, j]
        qn = qv / (qv.norm(dim=1, keepdim=True) + 1e-12)
        coeff = torch.einsum("hpd,hd->hp", R, qn)
        R = R - coeff[..., None] * qn[:, None, :]
        norms = (R * R).sum(-1)
        norms[ar, j] = -1.0
    return idx


def _pool_gqa(A, num_groups):
    nh, L = A.shape
    return A.view(nh // num_groups, num_groups, L).sum(1)


def _expand_gqa(keep_kv, num_groups):
    return keep_kv.repeat_interleave(num_groups, dim=0)


def build_keep_mask(A_win, A_acc, V_heads, args, num_groups):
    """Build a one-shot post-prefill keep mask.

    A_win, A_acc: (num_heads, L) question-window and accumulated attention.
    V_heads: (num_heads, L, d) values after repeat_kv.
    Returns: (num_heads, L) boolean keep mask.
    """
    nh, L = A_win.shape
    dev = A_win.device
    kvh = nh // num_groups
    V_kv = V_heads.view(kvh, num_groups, L, V_heads.shape[-1])[:, 0]

    Bt = int(args.evict_budget * L) if args.evict_budget < 1 else int(args.evict_budget)
    S = min(args.sink_tokens, L)
    Rn = min(args.recent_tokens, L)
    forced = torch.zeros(L, dtype=torch.bool, device=dev)
    forced[:S] = True
    forced[L - Rn:] = True
    n_sel = max(Bt - int(forced.sum().item()), 0)

    keep_kv = forced[None, :].repeat(kvh, 1).clone()
    if n_sel == 0:
        return _expand_gqa(keep_kv, num_groups)

    score = _pool_gqa(A_win, num_groups)
    score = score.masked_fill(forced[None, :], -1.0)

    if args.evict_method == "snapkv":
        sp = torch.nn.functional.max_pool1d(
            score.clamp_min(0)[:, None],
            kernel_size=args.pool_kernel,
            stride=1,
            padding=args.pool_kernel // 2,
        )[:, 0]
        sel = sp.topk(n_sel, dim=-1).indices
        keep_kv.scatter_(1, sel, True)

    elif args.evict_method == "h2o":
        sacc = _pool_gqa(A_acc, num_groups).masked_fill(forced[None, :], -1.0)
        sel = sacc.topk(n_sel, dim=-1).indices
        keep_kv.scatter_(1, sel, True)

    elif args.evict_method == "value_residual":
        cm = max(args.cand_mult, 1)
        P = min(cm * n_sel, L)
        cand = score.topk(P, dim=-1).indices
        a_c = score.gather(1, cand).clamp_min(0)
        v_c = torch.gather(V_kv, 1, cand[..., None].expand(-1, -1, V_kv.shape[-1]))
        M = a_c.sqrt()[..., None] * v_c
        pick = _greedy_css_batched(M, n_sel)
        sel = torch.gather(cand, 1, pick)
        keep_kv.scatter_(1, sel, True)

    elif args.evict_method == "full":
        keep_kv[:] = True

    else:
        raise ValueError(args.evict_method)

    return _expand_gqa(keep_kv, num_groups)
