import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import Encoder


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


class SALA(nn.Module):
    """
    SALA: Local window attention directly on feature maps (B,C,H,W).
    Spectral branch omitted to avoid patch dependence; returns (B,C,H,W).
    """

    def __init__(
        self,
        d_model: int,
        head_dim: int = 32,
        num_heads: int = None,
        window_size: int = 7,
    ):
        super().__init__()
        self.d_model = d_model
        if num_heads is None:
            num_heads = max(1, d_model // head_dim)
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * self.num_heads == d_model
        self.window_size = window_size

        # 1x1 conv projections for q, k, v and output
        self.q_proj = nn.Conv2d(d_model, d_model, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(d_model, d_model, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(d_model, d_model, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(d_model, d_model, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(d_model)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        ws = self.window_size
        pad_h = (ws - (H % ws)) % ws
        pad_w = (ws - (W % ws)) % ws

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if pad_h > 0 or pad_w > 0:
            q = F.pad(q, (0, pad_w, 0, pad_h))
            k = F.pad(k, (0, pad_w, 0, pad_h))
            v = F.pad(v, (0, pad_w, 0, pad_h))
        Hp, Wp = q.shape[-2], q.shape[-1]

        # reshape into windows and heads
        def to_windows(t: torch.Tensor):
            t = t.view(B, self.num_heads, self.head_dim, Hp, Wp)
            t = t.unfold(3, ws, ws).unfold(4, ws, ws)  # (B,heads,hd, nH, nW, ws, ws)
            nH, nW = t.shape[3], t.shape[4]
            t = t.permute(0, 1, 3, 4, 2, 5, 6).contiguous()  # (B,heads,nH,nW,hd,ws,ws)
            t = t.view(B, self.num_heads, nH * nW, self.head_dim, ws * ws)
            t = t.permute(0, 1, 2, 4, 3).contiguous()  # (B,heads,nWin,T,hd)
            return t, nH, nW

        q_w, nH, nW = to_windows(q)
        k_w, _, _ = to_windows(k)
        v_w, _, _ = to_windows(v)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q_w, k_w.transpose(-2, -1)) * scale  # (B,heads,nWin,T,T)
        attn = F.softmax(attn, dim=-1)
        out_w = torch.matmul(attn, v_w)  # (B,heads,nWin,T,hd)

        # fold windows back
        out_w = out_w.permute(0, 1, 4, 2, 3).contiguous()  # (B,heads,hd,nWin,T)
        out_w = out_w.view(B, self.num_heads * self.head_dim, nH, nW, ws, ws)
        out = (
            out_w.permute(0, 1, 2, 4, 3, 5)
            .contiguous()
            .view(B, self.num_heads * self.head_dim, nH * ws, nW * ws)
        )
        out = out[:, :, :H, :W]
        out = self.out_proj(out)
        out = self.norm(x + out)
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
    """
    ACGA: operate directly on feature map (B,C,H,W).
    - Score per spatial location with 1x1 conv.
    - Select top-M spatial nodes per sample.
    - Build cosine adjacency and run 2-layer GCN in node space.
    - Broadcast back to full map via attention (conv projections), reshape to (B,C,H,W).
    """

    def __init__(
        self,
        in_channels: int,
        node_dim: int = 64,
        max_nodes: int = 64,
        score_hidden: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.node_dim = node_dim
        self.max_nodes = max_nodes
        self.score_conv = nn.Sequential(
            nn.Conv2d(in_channels, score_hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(score_hidden, 1, kernel_size=1),
        )
        self.t2n = nn.Conv2d(in_channels, node_dim, kernel_size=1, bias=False)
        self.n2t = nn.Conv2d(node_dim, in_channels, kernel_size=1, bias=False)
        self.gcn = TwoLayerGCN(node_dim)

    def select_nodes(self, fmap: torch.Tensor):
        B, C, H, W = fmap.shape
        scores = self.score_conv(fmap).view(B, -1)  # (B, H*W)
        idxs = []
        H0_list = []
        proj = self.t2n(fmap)  # (B,node_dim,H,W)
        for b in range(B):
            k = min(self.max_nodes, max(4, (H * W) // 16))
            vals, topk = torch.topk(scores[b], k=k)
            idxs.append(topk)
            y = topk // W
            x = topk % W
            feats = proj[b, :, y, x].transpose(0, 1)  # (k, node_dim)
            H0_list.append(feats)
        M = max([h.shape[0] for h in H0_list])
        H0 = fmap.new_zeros((B, M, self.node_dim))
        mask = torch.zeros(B, M, dtype=torch.bool, device=fmap.device)
        for b in range(B):
            cur = H0_list[b]
            take = min(cur.shape[0], M)
            H0[b, :take, :] = cur[:take, :]
            mask[b, :take] = True
        return H0, mask, idxs, proj

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

    def forward(self, fmap: torch.Tensor):
        B, C, H, W = fmap.shape
        H0, mask, idxs, proj = self.select_nodes(fmap)  # H0: (B,M,node_dim)
        A = self.adjacency_from_nodes(H0, mask)
        Hg = self.gcn(H0, A)  # (B,M,node_dim)
        # Token-to-node attention via conv-projected features
        tokens_proj = proj.view(B, self.node_dim, H * W).transpose(
            1, 2
        )  # (B,N,node_dim)
        logits = torch.matmul(tokens_proj, Hg.transpose(1, 2)) / math.sqrt(
            self.node_dim
        )
        attn = F.softmax(logits, dim=-1)  # (B,N,M)
        injected = torch.matmul(attn, Hg)  # (B,N,node_dim)
        injected = injected.transpose(1, 2).view(B, self.node_dim, H, W)
        back = self.n2t(injected)
        out = fmap + back
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
            nn.Conv2d(2, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
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
        # Average mask into the token grid of size (H_h, W_h) to get one scalar per token map
        e_map = F.adaptive_avg_pool2d(mask_map, output_size=(H_h, W_h))  # (B,1,H_h,W_h)
        return e_map.clamp(0.0, 1.0), mask_map


class CrossAttentionFuse(nn.Module):
    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        out_dim: int,
        num_heads: int = 8,
        attn_q_chunk: int = 1024,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        assert self.head_dim * num_heads == out_dim
        self.q_proj = nn.Linear(q_dim, out_dim, bias=False)
        self.k_proj = nn.Linear(kv_dim, out_dim, bias=False)
        self.v_proj = nn.Linear(kv_dim, out_dim, bias=False)
        self.out = nn.Linear(out_dim, out_dim)
        self.attn_q_chunk = attn_q_chunk

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
        # Chunked attention over queries to reduce peak memory
        out_chunks = []
        chunk = max(1, self.attn_q_chunk)
        k_t = k.transpose(
            -2, -1
        )  # (B,heads,hd,Nk)-> but here (B,heads,Nk,hd)^T already used
        for qs in range(0, Nq, chunk):
            qe = min(Nq, qs + chunk)
            q_chunk = q[:, :, qs:qe, :]  # (B,heads,qc,hd)
            scores = (
                torch.matmul(q_chunk, k.transpose(-2, -1)) * scale
            )  # (B,heads,qc,Nk)
            attn = F.softmax(scores, dim=-1)
            out_chunk = torch.matmul(attn, v)  # (B,heads,qc,hd)
            out_chunks.append(out_chunk)
        out = torch.cat(out_chunks, dim=2)
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
            q_dim=d_h, kv_dim=d_l, out_dim=d_h, num_heads=8, attn_q_chunk=2048
        )
        self.fine2coarse_attn = CrossAttentionFuse(
            q_dim=d_l, kv_dim=d_h, out_dim=d_l, num_heads=8, attn_q_chunk=2048
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


class FIEMHead(nn.Module):
    def __init__(self, ch_high: int, ch_low: int, mid_ch: int = 64, max_steps: int = 4):
        super().__init__()
        self.high_proj = nn.Sequential(
            nn.Conv2d(ch_high, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.low_proj = nn.Sequential(
            nn.Conv2d(ch_low, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.up_low = nn.ModuleList(
            [UpSampling2x(mid_ch, mid_ch) for _ in range(max_steps)]
        )
        self.up_high = nn.ModuleList(
            [UpSampling2x(mid_ch, mid_ch) for _ in range(max_steps)]
        )
        self.up_fuse = nn.ModuleList(
            [UpSampling2x(mid_ch, mid_ch) for _ in range(max_steps)]
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(mid_ch * 2, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_ch, mid_ch, kernel_size=3, padding=1, groups=mid_ch, bias=False
            ),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )

        self.head_high = MaskHead(in_ch=mid_ch)
        self.head_low = MaskHead(in_ch=mid_ch)
        self.head_fusion = MaskHead(in_ch=mid_ch)

    def _steps(self, src: int, dst: int) -> int:
        scale = max(1, dst // src)
        return int(math.log2(scale)) if scale > 0 else 0

    def forward(self, gh: torch.Tensor, gl: torch.Tensor, H0: int, W0: int):
        B, Ch, Hh, Wh = gh.shape
        _, Cl, Hl, Wl = gl.shape
        xh = self.high_proj(gh)  # (B,mid,Hh,Wh)
        xl = self.low_proj(gl)  # (B,mid,Hl,Wl)

        steps_l2h = self._steps(Hl, Hh)
        for i in range(steps_l2h):
            xl = self.up_low[i](xl)

        fuse = self.fuse(torch.cat([xh, xl], dim=1))  # (B,mid,Hh,Wh)

        steps_h2in = self._steps(Hh, H0)
        up_high = xh
        up_fuse = fuse
        for i in range(steps_h2in):
            up_high = self.up_high[i](up_high)
            up_fuse = self.up_fuse[i](up_fuse)

        steps_l2in = self._steps(Hl, H0)
        low_up = self.low_proj(gl)
        for i in range(steps_l2in):
            low_up = self.up_low[i](low_up)

        high_mask = self.head_high(up_high)
        low_mask = self.head_low(low_up)
        fusion_mask = self.head_fusion(up_fuse)
        return low_mask, high_mask, fusion_mask


class UpSampling2x(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(UpSampling2x, self).__init__()
        temp_chs = out_chs * 4  # for PixelShuffle
        self.up_module = nn.Sequential(
            nn.Conv2d(in_chs, temp_chs, 1, bias=False),
            nn.BatchNorm2d(temp_chs),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
        )

    def forward(self, features):
        return self.up_module(features)


class GroupFusion(nn.Module):
    def __init__(self, in_chs, out_chs, end=False):  # 768, 384
        super(GroupFusion, self).__init__()

        if end:
            tmp_chs = in_chs * 2
        else:
            tmp_chs = in_chs
        self.gf1 = nn.Sequential(
            nn.Conv2d(in_chs * 2, in_chs, 1, bias=False),
            nn.BatchNorm2d(in_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_chs, in_chs, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_chs, in_chs, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_chs),
            nn.ReLU(inplace=True),
        )

        self.gf2 = nn.Sequential(
            nn.Conv2d(in_chs * 2, tmp_chs, 1, bias=False),
            nn.BatchNorm2d(tmp_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(tmp_chs, tmp_chs, 3, padding=1, bias=False),
            nn.BatchNorm2d(tmp_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(tmp_chs, tmp_chs, 3, padding=1, bias=False),
            nn.BatchNorm2d(tmp_chs),
            nn.ReLU(inplace=True),
        )

        self.gf3 = nn.Sequential(
            nn.Conv2d(in_chs * 2, tmp_chs, 1, bias=False),
            nn.BatchNorm2d(tmp_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(tmp_chs, tmp_chs, 3, padding=1, bias=False),
            nn.BatchNorm2d(tmp_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(tmp_chs, tmp_chs, 3, padding=1, bias=False),
            nn.BatchNorm2d(tmp_chs),
            nn.ReLU(inplace=True),
        )
        self.up2x_1 = UpSampling2x(tmp_chs, out_chs)
        self.up2x_2 = UpSampling2x(tmp_chs, out_chs)

    def forward(self, f_up1, f_up2, f_down1, f_down2):  # [768,24,24]
        fc1 = torch.cat((f_down1, f_down2), dim=1)
        f_tmp = self.gf1(fc1)

        out1 = self.gf2(torch.cat((f_tmp, f_up1), dim=1))
        out2 = self.gf3(torch.cat((f_tmp, f_up2), dim=1))

        return self.up2x_1(out1), self.up2x_2(out2)  # [384,48,48]


class SSC_GT(nn.Module):
    def __init__(
        self,
        in_ch=3,
        d_h=128,
        p_h=8,
        s_h=None,
        d_l=128,
        p_l=32,
        window_size=7,
        encoder={},
    ):
        super().__init__()
        self.encoder = Encoder(**encoder)
        self.sala_h = SALA(d_model=d_h, head_dim=32, window_size=window_size)
        self.sala_l = SALA(
            d_model=d_l, head_dim=32, window_size=max(3, window_size // 2)
        )
        self.acga_h = ACGA(in_channels=d_h, node_dim=64, max_nodes=64)
        self.acga_l = ACGA(in_channels=d_l, node_dim=64, max_nodes=64)
        self.bacep = BACEP(p_h=p_h, token_stride=p_h)
        self.adpd = ADPD(d_h=d_h, d_l=d_l)
        self.high_proj = nn.Sequential(
            nn.Conv2d(d_h, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.up_high = nn.ModuleList([UpSampling2x(64, 64) for _ in range(3)])
        self.up_low = nn.ModuleList([UpSampling2x(d_l, d_l) for _ in range(4)])
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
        self.fiem = FIEMHead(ch_high=d_h, ch_low=d_l, mid_ch=64, max_steps=4)

        self.gf1_1 = GroupFusion(320, 128)
        self.gf1_2 = GroupFusion(128, 64)
        self.gf1_3 = GroupFusion(64, 64, end=True)

        self.gf2_2 = GroupFusion(128, 64)
        self.gf2_3 = GroupFusion(64, 64, end=True)

        self.out_F1 = nn.Sequential(
            nn.Conv2d(128, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.out_F2 = nn.Sequential(
            nn.Conv2d(128, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def cim_decoder(self, tokens):
        f = []
        size = [96, 96, 48, 48, 24, 24, 24, 24, 48, 48, 96, 96]
        for i in range(len(tokens)):
            b, _, c = tokens[i].shape
            f.append(
                tokens[i].permute(0, 2, 1).view(b, c, size[i], size[i]).contiguous()
            )

        f1_1, f1_2 = self.gf1_1(f[7], f[4], f[5], f[6])

        f2_1, f2_2 = self.gf1_2(f[9], f[8], f1_1, f1_2)
        f2_3, f2_4 = self.gf2_2(f[3], f[2], f1_1, f1_2)

        f3_1, f3_2 = self.gf1_3(f[11], f[10], f2_2, f2_1)
        f3_3, f3_4 = self.gf2_3(f[1], f[0], f2_3, f2_4)

        fout1 = self.out_F1(torch.cat([f3_1, f3_2], dim=1))  # (B,128,96,96)
        fout2 = self.out_F2(torch.cat([f3_3, f3_4], dim=1))  # (B,128,48,48)
        return fout1, fout2  # high, low

    def forward(self, img: torch.Tensor):
        x = self.encoder(img)
        enc_high, enc_low = self.cim_decoder(x)  # (B,128,96,96) and (B,128,48,48)
        B, C, H0, W0 = img.shape
        th = self.sala_h(enc_high)
        tl = self.sala_l(enc_low)
        gh = self.acga_h(th)
        gl = self.acga_l(tl)
        low_mask, high_mask, fusion_mask = self.fiem(gh, gl, H0, W0)
        return [low_mask, high_mask, fusion_mask]


def get_model(**kwargs):
    return SSC_GT(**kwargs)


if __name__ == "__main__":
    device = torch.device("cpu")
    model = get_model().to(device)
    x = torch.randn(1, 3, 384, 384, device=device)
    out = model(x)
    print("Outputs:")
    for i, o in enumerate(out):
        print(f"  mask[{i}]:", o.shape)
    print(f"number of parameters: {sum(p.numel() for p in model.parameters()):,}")
