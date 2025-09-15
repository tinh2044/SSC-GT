import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_dct_basis(n: int, device=None, dtype=torch.float32):
    """Create DCT-II orthonormal basis matrix (n x n)."""
    if device is None:
        device = torch.device("cpu")
    k = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)  # (n,1)
    i = torch.arange(n, device=device, dtype=dtype).unsqueeze(0)  # (1,n)
    B = torch.cos(math.pi * (i + 0.5) * k / n)  # (n,n)
    B[0, :] *= 1.0 / math.sqrt(2.0)
    B = B * math.sqrt(2.0 / n)
    return B  # (n,n)


def unfold_patches(x: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    """Extract patches using F.unfold. Returns (B, C*ks*ks, Np)."""
    patches = F.unfold(x, kernel_size=kernel_size, stride=stride, padding=0)
    return patches


def fold_patches(
    patches: torch.Tensor, out_size: Tuple[int, int], kernel_size: int, stride: int
) -> torch.Tensor:
    """Fold patches (B, C*ks*ks, Np) => (B,C,H,W) using F.fold."""
    return F.fold(
        patches,
        output_size=(out_size[0], out_size[1]),
        kernel_size=kernel_size,
        stride=stride,
    )


class MultiScaleTokenizer(nn.Module):
    def __init__(self, in_ch=3, d_h=128, p_h=8, s_h=None, d_l=128, p_l=32):
        super().__init__()
        self.p_h = p_h
        # Remove overlap by default: stride equals patch size to reduce token count and FLOPs
        self.s_h = s_h if s_h is not None else p_h
        self.p_l = p_l

        self.conv_h = nn.Conv2d(
            in_ch, d_h, kernel_size=p_h, stride=self.s_h, padding=0, bias=False
        )
        self.norm_h = nn.LayerNorm(d_h)
        self.conv_l = nn.Conv2d(
            in_ch, d_l, kernel_size=p_l, stride=p_l, padding=0, bias=False
        )
        self.norm_l = nn.LayerNorm(d_l)

        nn.init.kaiming_normal_(self.conv_h.weight, a=0, mode="fan_out")
        nn.init.kaiming_normal_(self.conv_l.weight, a=0, mode="fan_out")

    def forward(self, x: torch.Tensor):
        B, C, H0, W0 = x.shape

        feat_h = self.conv_h(x)  # (B,d_h,H_h,W_h)
        B_, d_h, H_h, W_h = feat_h.shape
        tokens_h = feat_h.flatten(2).transpose(1, 2).contiguous()  # (B,N_h,d_h)
        tokens_h = self.norm_h(tokens_h)
        patches_h = unfold_patches(
            x, kernel_size=self.p_h, stride=self.s_h
        )  # (B, 3*p_h*p_h, N_h)

        feat_l = self.conv_l(x)
        _, d_l, H_l, W_l = feat_l.shape
        tokens_l = feat_l.flatten(2).transpose(1, 2).contiguous()
        tokens_l = self.norm_l(tokens_l)
        patches_l = unfold_patches(x, kernel_size=self.p_l, stride=self.p_l)

        return {
            "feat_h_map": feat_h,
            "tokens_h": tokens_h,
            "patches_h": patches_h,
            "H_h": H_h,
            "W_h": W_h,
            "feat_l_map": feat_l,
            "tokens_l": tokens_l,
            "patches_l": patches_l,
            "H_l": H_l,
            "W_l": W_l,
            "H0": H0,
            "W0": W0,
            "s_h": self.s_h,
            "p_h": self.p_h,
            "p_l": self.p_l,
        }


class SALA(nn.Module):
    """
    SALA: combines local windowed multi-head attention and global spectral attention (DCT).
    Input:
      tokens: (B,N,d)
      patches: (B, 3*ks*ks, N)  (raw RGB patches)
    """

    def __init__(
        self,
        d_model: int,
        head_dim: int = 32,
        num_heads: int = None,
        window_size: int = 7,
        patch_size: int = 8,
        spectral_stride: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        if num_heads is None:
            num_heads = max(1, d_model // head_dim)
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * self.num_heads == d_model

        # spatial projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # spectral path
        self.p_size = patch_size
        self.register_buffer("dct_basis", make_dct_basis(self.p_size))
        self.spec_proj = nn.Linear(3 * self.p_size * self.p_size, d_model)
        self.spec_q = nn.Linear(d_model, d_model)
        self.spec_k = nn.Linear(d_model, d_model)
        self.spec_v = nn.Linear(d_model, d_model)
        self.spec_out = nn.Linear(d_model, d_model)

        # gating between spatial & spectral
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.window_size = window_size
        # Downsample factor for spectral attention to reduce global O(N^2) cost
        # When >1, compute spectral attention on a strided grid and upsample back
        self.spectral_stride = max(1, spectral_stride)

    def dct_on_patches(self, patches: torch.Tensor):
        """
        patches: (B, 3*ks*ks, N)
        returns: (B, N, d_model)
        """
        B, Cks, N = patches.shape
        ks = self.p_size
        assert Cks == 3 * ks * ks
        device = patches.device
        # reshape to (B,3,ks,ks,N)
        patches = patches.view(B, 3, ks, ks, N)
        Bmat = self.dct_basis.to(device=device, dtype=patches.dtype)  # (ks,ks)

        # compute DCT per-sample in batch (vectorized using einsum)
        # p: (B,3,ks,ks,N) -> apply Bmat over spatial dims
        # First multiply left: (ks,ks) x (3,ks,ks,N) -> (3,ks,ks,N)
        # Use einsum to broadcast matmul across appropriate dims:
        # result shape after two einsums -> (B,3,ks,ks,N) then flatten to (B,N,3*ks*ks)
        F_ch = torch.einsum("ij, b c j k n -> b c i k n", Bmat, patches)
        F_ch = torch.einsum("b c i j n, k j -> b c i k n", F_ch, Bmat.t())
        F_flat = F_ch.reshape(B, 3 * ks * ks, N).permute(0, 2, 1)  # (B, N, 3*ks*ks)
        spec_desc = self.spec_proj(F_flat)  # (B,N,d_model)
        return spec_desc

    def local_window_attention(self, tokens: torch.Tensor, H: int, W: int):
        """
        tokens: (B,N,d)
        returns: (B,N,d)
        """
        B, N, d = tokens.shape
        assert N == H * W
        ws = self.window_size
        pad_h = (ws - (H % ws)) % ws
        pad_w = (ws - (W % ws)) % ws
        x = tokens.reshape(B, H, W, d)  # safe reshape
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h)).permute(0, 2, 3, 1)
            Hp = H + pad_h
            Wp = W + pad_w
        else:
            Hp, Wp = H, W

        # group windows and compute attention per window
        xw = (
            x.reshape(B, Hp // ws, ws, Wp // ws, ws, d)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
        )
        Bn, Wh, Ww, ws1, ws2, d = xw.shape
        xw = xw.reshape(B * (Hp // ws) * (Wp // ws), ws * ws, d)

        q = self.q_proj(xw)
        k = self.k_proj(xw)
        v = self.v_proj(xw)
        # multi-head
        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(xw.shape[0], xw.shape[1], d)
        out = self.out_proj(out)

        out = (
            out.reshape(B, Hp // ws, Wp // ws, ws, ws, d)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
        )
        out = out.reshape(B, Hp, Wp, d)
        out = out[:, :H, :W, :].contiguous().reshape(B, H * W, d)
        return out

    def spectral_attention(self, spec_desc: torch.Tensor):
        q = self.spec_q(spec_desc)
        k = self.spec_k(spec_desc)
        v = self.spec_v(spec_desc)
        scale = 1.0 / math.sqrt(self.d_model)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = self.spec_out(out)
        return out

    def forward(self, tokens: torch.Tensor, patches: torch.Tensor, H: int, W: int):
        sp_out = self.local_window_attention(tokens, H, W)  # (B,N,d)
        spec_desc = self.dct_on_patches(patches)  # (B,N,d)
        if self.spectral_stride > 1 and (H * W) > 0:
            # Downsample descriptors on token grid, run attention, then upsample back
            B, N, D = spec_desc.shape
            s = self.spectral_stride
            # reshape to B,H,W,D
            spec_hw = spec_desc.view(B, H, W, D)
            # strided sampling
            Hs = (H + s - 1) // s
            Ws = (W + s - 1) // s
            spec_ds = spec_hw[:, ::s, ::s, :]  # (B,Hs,Ws,D)
            spec_ds = spec_ds.contiguous().view(B, Hs * Ws, D)
            spec_out_ds = self.spectral_attention(spec_ds)  # (B,Hs*Ws,D)
            # upsample back to H,W
            spec_out_ds = (
                spec_out_ds.view(B, Hs, Ws, D).permute(0, 3, 1, 2).contiguous()
            )
            spec_out_full = (
                F.interpolate(spec_out_ds, size=(H, W), mode="nearest")
                .permute(0, 2, 3, 1)
                .contiguous()
            )
            spec_out = spec_out_full.view(B, N, D)
        else:
            spec_out = self.spectral_attention(spec_desc)  # (B,N,d)
        alpha = torch.sigmoid(self.alpha)
        fused = alpha * sp_out + (1 - alpha) * spec_out
        out = tokens + fused
        out = F.layer_norm(out, out.shape[-1:])
        return out


class TwoLayerGCN(nn.Module):
    def __init__(self, d_node: int):
        super().__init__()
        self.lin1 = nn.Linear(d_node, d_node, bias=False)
        self.lin2 = nn.Linear(d_node, d_node, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, H_nodes: torch.Tensor, A: torch.Tensor):
        x = torch.matmul(A, H_nodes)
        x = self.act(self.lin1(x))
        x = torch.matmul(A, x)
        x = self.act(self.lin2(x))
        return x


class ACGA(nn.Module):
    def __init__(
        self,
        token_dim: int,
        node_dim: int = 64,
        max_nodes: int = 64,
        score_hidden: int = 64,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.node_dim = node_dim
        self.max_nodes = max_nodes
        self.score_mlp = nn.Sequential(
            nn.Linear(token_dim, score_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(score_hidden, 1),
        )
        self.t2n = nn.Linear(token_dim, node_dim, bias=False)
        self.n2t = nn.Linear(node_dim, token_dim, bias=False)
        self.gcn = TwoLayerGCN(node_dim)

    def select_nodes(self, token_feats: torch.Tensor):
        B, N, d = token_feats.shape
        scores = self.score_mlp(token_feats).squeeze(-1)
        means = scores.mean(dim=1, keepdim=True)
        stds = scores.std(dim=1, unbiased=False, keepdim=True)
        thr = means + 0.5 * stds
        selected = scores > thr
        nodes_list = []
        idx_list = []
        for b in range(B):
            sel_idx = torch.nonzero(selected[b]).squeeze(-1)
            if sel_idx.numel() == 0:
                k = min(self.max_nodes, max(4, N // 16))
                _, topk = torch.topk(scores[b], k=k)
                sel_idx = topk
            else:
                if sel_idx.numel() > self.max_nodes:
                    _, topk = torch.topk(scores[b][sel_idx], k=self.max_nodes)
                    sel_idx = sel_idx[topk]
            idx_list.append(sel_idx)
            nodes_list.append(self.t2n(token_feats[b, sel_idx, :]))
        M = max([n.shape[0] for n in nodes_list])
        M = min(M, self.max_nodes)
        H0 = token_feats.new_zeros((B, M, self.node_dim))
        mask = torch.zeros(B, M, dtype=torch.bool, device=token_feats.device)
        for b in range(B):
            cur = nodes_list[b]
            cur_n = cur.shape[0]
            take = min(cur_n, M)
            H0[b, :take, :] = cur[:take, :]
            mask[b, :take] = True
            idx_list[b] = idx_list[b][:take]
        return H0, mask, idx_list

    def adjacency_from_nodes(self, H_nodes: torch.Tensor, mask: torch.Tensor):
        B, M, d = H_nodes.shape
        eps = 1e-6
        norm = H_nodes.norm(dim=-1, keepdim=True).clamp(min=eps)
        Hn = H_nodes / norm
        A = torch.matmul(Hn, Hn.transpose(-2, -1))
        A = F.relu(A)
        maskf = mask.float().unsqueeze(1) * mask.float().unsqueeze(2)
        A = A * maskf
        A = A + torch.eye(M, device=A.device).unsqueeze(0) * mask.unsqueeze(1).float()
        rowsum = A.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        A_norm = A / rowsum
        return A_norm

    def forward(self, token_feats: torch.Tensor):
        B, N, d = token_feats.shape
        H0, mask, idx_list = self.select_nodes(token_feats)
        A = self.adjacency_from_nodes(H0, mask)
        Hg = self.gcn(H0, A)
        token_proj = self.t2n(token_feats)
        logits = torch.matmul(token_proj, Hg.transpose(-2, -1))
        attn = F.softmax(logits / math.sqrt(self.node_dim), dim=-1)
        injected = torch.matmul(attn, Hg)
        back = self.n2t(injected)
        out = token_feats + back
        return out


class BACEP(nn.Module):
    def __init__(self, p_h: int, token_stride: int):
        super().__init__()
        self.p_h = p_h
        self.stride = token_stride
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
        )
        sobel_y = sobel_x.t()
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))
        self.comb_conv = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
        )

    def forward(self, x_img: torch.Tensor, feat_h_map: torch.Tensor):
        B, _, H0, W0 = x_img.shape
        gray = x_img.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        edges = torch.sqrt(gx * gx + gy * gy + 1e-8)
        _, d_h, H_h, W_h = feat_h_map.shape
        feat_up = F.interpolate(
            feat_h_map, size=(H0, W0), mode="bilinear", align_corners=False
        )
        feat_mag = feat_up.pow(2).mean(dim=1, keepdim=True)
        comb = torch.cat([edges, feat_mag], dim=1)
        mask_map = torch.sigmoid(self.comb_conv(comb))
        patches = unfold_patches(mask_map, kernel_size=self.p_h, stride=self.stride)
        e_tokens = patches.mean(dim=1)
        return e_tokens.clamp(0.0, 1.0), mask_map


class CrossAttentionFuse(nn.Module):
    def __init__(self, q_dim: int, kv_dim: int, out_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        assert self.head_dim * num_heads == out_dim
        self.q_proj = nn.Linear(q_dim, out_dim, bias=False)
        self.k_proj = nn.Linear(kv_dim, out_dim, bias=False)
        self.v_proj = nn.Linear(kv_dim, out_dim, bias=False)
        self.out = nn.Linear(out_dim, out_dim)

    def forward(
        self,
        Q_tokens: torch.Tensor,
        K_tokens: torch.Tensor,
        V_tokens: torch.Tensor,
        weight_K: Optional[torch.Tensor] = None,
    ):
        B, Nq, _ = Q_tokens.shape
        Nk = K_tokens.shape[1]
        q = (
            self.q_proj(Q_tokens)
            .view(B, Nq, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k_proj(K_tokens)
            .view(B, Nk, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v_proj(V_tokens)
            .view(B, Nk, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        if weight_K is not None:
            wk = weight_K.unsqueeze(1).unsqueeze(-1)
            k = k * wk
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = (
            out.permute(0, 2, 1, 3)
            .contiguous()
            .reshape(B, Nq, self.num_heads * self.head_dim)
        )
        out = self.out(out)
        return out


class ADPD(nn.Module):
    def __init__(self, d_h: int, d_l: int):
        super().__init__()
        self.coarse2fine_attn = CrossAttentionFuse(
            q_dim=d_h, kv_dim=d_l, out_dim=d_h, num_heads=8
        )
        self.fine2coarse_attn = CrossAttentionFuse(
            q_dim=d_l, kv_dim=d_h, out_dim=d_l, num_heads=8
        )
        self.ln_h = nn.LayerNorm(d_h)
        self.ln_l = nn.LayerNorm(d_l)
        self.ffn_h = nn.Sequential(
            nn.Linear(d_h, d_h * 2), nn.GELU(), nn.Linear(d_h * 2, d_h)
        )
        self.ffn_l = nn.Sequential(
            nn.Linear(d_l, d_l * 2), nn.GELU(), nn.Linear(d_l * 2, d_l)
        )
        self.lambda_h = nn.Parameter(torch.tensor(1.0))
        self.lambda_l = nn.Parameter(torch.tensor(0.5))

    def forward(
        self, tokens_h: torch.Tensor, tokens_l: torch.Tensor, e_tokens: torch.Tensor
    ):
        U_h = self.coarse2fine_attn(tokens_h, tokens_l, tokens_l)
        U_l = self.fine2coarse_attn(tokens_l, tokens_h, tokens_h, weight_K=e_tokens)
        F_h = self.ln_h(tokens_h + self.lambda_h * U_h + self.ffn_h(tokens_h))
        F_l = self.ln_l(tokens_l + self.lambda_l * U_l + self.ffn_l(tokens_l))
        return F_h, F_l


class PatchDecoder(nn.Module):
    def __init__(self, d_token: int, patch_size: int, out_channels: int):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.lin = nn.Linear(d_token, out_channels * patch_size * patch_size)

    def forward(self, tokens: torch.Tensor, H0: int, W0: int, stride: int):
        B, N, d = tokens.shape
        p = self.patch_size
        patches = self.lin(tokens)  # (B,N, out_ch*p*p)
        patches = patches.transpose(1, 2).contiguous()  # (B, out_ch*p*p, N)
        feat_map = fold_patches(
            patches, out_size=(H0, W0), kernel_size=p, stride=stride
        )
        ones = torch.ones((B, 1, H0, W0), device=tokens.device, dtype=tokens.dtype)
        coverage = fold_patches(
            F.unfold(ones, kernel_size=p, stride=stride),
            out_size=(H0, W0),
            kernel_size=p,
            stride=stride,
        )
        coverage = coverage.clamp(min=1.0)
        feat_map = feat_map / coverage
        return feat_map


class MaskHead(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        # Lightweight head: 1x1 bottleneck -> depthwise 3x3 -> 1x1 to mask
        mid = max(16, in_ch // 4)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, fmap: torch.Tensor):
        return self.net(fmap)


class SSC_GT(nn.Module):
    def __init__(
        self, in_ch=3, d_h=128, p_h=8, s_h=None, d_l=128, p_l=32, window_size=7
    ):
        super().__init__()
        self.tokenizer = MultiScaleTokenizer(
            in_ch=in_ch, d_h=d_h, p_h=p_h, s_h=s_h, d_l=d_l, p_l=p_l
        )
        self.sala_h = SALA(
            d_model=d_h,
            head_dim=32,
            window_size=window_size,
            patch_size=p_h,
            spectral_stride=2,
        )
        self.sala_l = SALA(
            d_model=d_l,
            head_dim=32,
            window_size=max(3, window_size // 2),
            patch_size=p_l,
            spectral_stride=1,
        )
        self.acga_h = ACGA(token_dim=d_h, node_dim=64, max_nodes=64)
        self.acga_l = ACGA(token_dim=d_l, node_dim=64, max_nodes=64)
        self.bacep = BACEP(p_h=p_h, token_stride=self.tokenizer.s_h)
        self.adpd = ADPD(d_h=d_h, d_l=d_l)
        # Reduce decoder channels to shrink full-res compute
        self.decoder = PatchDecoder(d_token=d_h, patch_size=p_h, out_channels=64)
        # Depthwise-separable fusion with bottleneck
        fusion_in_ch = 64 + d_l
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_in_ch, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1, groups=96, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.head_low = MaskHead(in_ch=d_l)
        self.head_high = MaskHead(in_ch=64)
        self.head_fusion = MaskHead(in_ch=64)

    def forward(self, img: torch.Tensor):
        t = self.tokenizer(img)
        feat_h_map = t["feat_h_map"]
        tokens_h = t["tokens_h"]
        patches_h = t["patches_h"]
        H_h = t["H_h"]
        W_h = t["W_h"]
        feat_l_map = t["feat_l_map"]
        tokens_l = t["tokens_l"]
        patches_l = t["patches_l"]
        H_l = t["H_l"]
        W_l = t["W_l"]
        H0 = t["H0"]
        W0 = t["W0"]

        th = self.sala_h(tokens_h, patches_h, H_h, W_h)
        tl = self.sala_l(tokens_l, patches_l, H_l, W_l)

        gh = self.acga_h(th)
        gl = self.acga_l(tl)

        e_tokens, edge_map = self.bacep(img, feat_h_map)

        Eh = gh * (1.0 + 1.0 * e_tokens.unsqueeze(-1))

        Fh, Fl = self.adpd(Eh, gl, e_tokens)

        recon_high = self.decoder(Fh, H0=H0, W0=W0, stride=t["s_h"])
        B = Fh.shape[0]
        Fl_map = Fl.transpose(1, 2).contiguous().reshape(B, Fl.shape[2], H_l, W_l)
        Fl_up = F.interpolate(
            Fl_map, size=(H0, W0), mode="bilinear", align_corners=False
        )

        high_mask = self.head_high(recon_high)
        low_mask = self.head_low(Fl_up)
        fusion_in = torch.cat([recon_high, Fl_up], dim=1)
        fusion_feat = self.fusion_conv(fusion_in)
        fusion_mask = self.head_fusion(fusion_feat)

        return [low_mask, high_mask, fusion_mask]


def get_model(**kwargs):
    return SSC_GT(**kwargs)


if __name__ == "__main__":
    device = torch.device("cpu")
    model = get_model().to(device)
    x = torch.randn(1, 3, 512, 512, device=device)
    out = model(x)
    print("Outputs:")
    for i, o in enumerate(out):
        print(f"  mask[{i}]:", o.shape)
