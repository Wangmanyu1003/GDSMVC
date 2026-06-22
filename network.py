
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, low_feature_dim):
        super(Encoder, self).__init__()
        self.encoder_layer = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, low_feature_dim),
        )

    def forward(self, x):
        return self.encoder_layer(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, low_feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(low_feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class Network(nn.Module):
    def __init__(self, view, input_size, low_feature_dim, high_feature_dim, class_num, device):
        super(Network, self).__init__()
        self.device = device
        self.view = view
        self.low_feature_dim = low_feature_dim
        self.high_feature_dim = high_feature_dim
        self.cluster_num = class_num  # 或者你喜欢的命名

        # encoders / decoders
        self.encoders = nn.ModuleList([Encoder(input_size[v], low_feature_dim) for v in range(view)])
        self.decoders = nn.ModuleList([Decoder(input_size[v], low_feature_dim) for v in range(view)])

        # learnable global view weights (un-normalized)
        self.view_weights = nn.Parameter(torch.ones(view, dtype=torch.float32))

        # ---- SE MLP: 用于从 z_i -> u_i (输出维度 = high_feature_dim) ----
        self.se_mlp = nn.Sequential(
            nn.Linear(low_feature_dim * view, low_feature_dim * view // 2),
            nn.Linear(low_feature_dim * view // 2, low_feature_dim),
        )

        self.sf_learn = nn.Sequential(
            nn.Linear(low_feature_dim, high_feature_dim),
        )

        # 门控 mlp
        self.gate_mlp = nn.Sequential(
            nn.Linear(low_feature_dim, high_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(high_feature_dim, low_feature_dim),
        )

        # label_p 输出概率
        self.label_p = nn.Sequential(
            nn.Linear(low_feature_dim, high_feature_dim),
            nn.Linear(high_feature_dim, class_num),
            nn.Softmax(dim=1)
        )

        self.label_G = nn.Sequential(
            nn.Linear(low_feature_dim, high_feature_dim),
            nn.Linear(high_feature_dim, class_num),
            nn.Softmax(dim=1)
        )

        # 每个视图的可学习 gamma（初始化为 0.5）
        self.gammas = nn.Parameter(torch.full((view,), 0.5, dtype=torch.float32))

        # 1) 自注意力：所有视图共用同一层（embed_dim = low_feature_dim）
        self.mha_intra = nn.MultiheadAttention(
            embed_dim=self.low_feature_dim, num_heads=4, batch_first=True
        )

        # 2) 跨视图注意力：以“第 v 视图”为 query 的层，给每个视图一层（也可共用一层）
        self.mha_cross = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=self.low_feature_dim, num_heads=4, batch_first=True)
            for _ in range(self.view)
        ])

    def split_by_gate_single(self, Z_v: torch.Tensor, Z_list: list, v: int):
        """
        用 Transformer 注意力机制改进的门控分割
        Z_v: (N, d), Z_list: list of (N, d)
        返回 S_v, P_v, G_v, Context_v （均 (N, d)）
        """
        device = Z_v.device
        N, d = Z_v.shape
        T = len(Z_list)

        # ---------- 1) 自注意力 Intra^v ----------
        # MHA(batch_first=True) 期望输入 (B, L, D)，这里 B=1, L=N, D=d
        Intra, _ = self.mha_intra(Z_v.unsqueeze(0), Z_v.unsqueeze(0), Z_v.unsqueeze(0))
        Intra = Intra.squeeze(0)  # (N, d)

        # ---------- 2) 跨视图注意力 Cross^v ----------
        if T <= 1:
            Cross = torch.zeros_like(Intra, device=device)
        else:
            acc = torch.zeros_like(Intra, device=device)
            cnt = 0
            for u_idx, Z_u in enumerate(Z_list):
                if u_idx == v:
                    continue
                cross_out, _ = self.mha_cross[v](
                    Z_v.unsqueeze(0),  # query
                    Z_u.unsqueeze(0),  # key
                    Z_u.unsqueeze(0)  # value
                )
                acc = acc + cross_out.squeeze(0)
                cnt += 1
            Cross = acc / max(cnt, 1)

        # ========== 3) Context^v 组合 ==========
        gamma_v = torch.clamp(self.gammas[v], 0.0, 1.0)
        Context = gamma_v * Intra + (1.0 - gamma_v) * Cross

        # （可选）残差归一化，使数值稳定
        Context = F.normalize(Context + Z_v, p=2, dim=1)

        # ========== 4) 门控计算 ==========
        cat = torch.cat([Context], dim=1)  # (N, d)
        gate_feats = self.gate_mlp(cat)  # (N, d)
        G = torch.sigmoid(gate_feats)  # (N, d)

        # ========== 5) 分解 ==========
        S_v = G * Z_v
        P_v = (1.0 - G) * Z_v

        return S_v, P_v, G, Context

    def fuse_views_se1(self, S_raw_list, normalize: str = "softmax"):
        """
        Per-view SE + view-weight fusion：
        每个视图先做通道SE增强，再用视图级权重融合到 (N, d)。

        参数
        ----
        S_raw_list : List[Tensor]，长度 T；每个 (N, d)
        normalize  : 视图权重规范化方式（"softmax"/"sigmoid"/"l2"/"none"）

        返回
        ----
        S_bar : Tensor，(N, d)
        """
        assert isinstance(S_raw_list, (list, tuple)) and len(S_raw_list) > 0
        N, d = S_raw_list[0].shape
        device = S_raw_list[0].device
        dtype = S_raw_list[0].dtype
        T = len(S_raw_list)

        # ------- 1) 懒加载：每视图一个 SE( d -> d/r -> d ) -------
        # 结构：x -> Linear(d,d//r) -> ReLU -> Linear(d//r,d) -> sigmoid -> scale
        # 采用 ModuleList 便于动态 T
        if not hasattr(self, "_pv_se_list") or len(self._pv_se_list) != T \
                or getattr(self, "_pv_se_d", None) != d:
            r = 4
            hidden = max(1, d // r)
            self._pv_se_list = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d, hidden, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden, d, bias=True)
                ) for _ in range(T)
            ])
            self._pv_se_list.to(device=device)
            # 共享一套 dtype
            for m in self._pv_se_list:
                m[0].to(dtype=dtype);
                m[2].to(dtype=dtype)
            self._pv_se_d = d

        # ------- 2) 逐视图通道 SE 增强 -------
        S_tilde_list = []
        for v in range(T):
            S_v = S_raw_list[v].to(device=device, dtype=dtype)  # (N, d)
            g_v = self._pv_se_list[v](S_v)  # (N, d)
            gate_v = torch.sigmoid(g_v)  # (N, d)
            S_tilde_v = gate_v * S_v  # (N, d)
            S_tilde_list.append(S_tilde_v)

        # 堆叠：(N, T, d)
        S_tilde = torch.stack(S_tilde_list, dim=1)

        # ------- 3) 视图级权重（标量权重，稳健省参） -------
        # 用共享的线性头对每视图特征产生一个 score（比单纯 avg 更灵活）
        if not hasattr(self, "_view_score") or self._view_score.in_features != d:
            self._view_score = nn.Linear(d, 1, bias=True).to(device=device, dtype=dtype)

        # scores: (N, T, 1) -> squeeze 为 (N, T)
        scores = self._view_score(S_tilde)  # (N, T, 1)
        scores = scores.squeeze(-1)  # (N, T)

        # 归一化为权重 a: (N, T)
        if normalize == "softmax":
            a = torch.softmax(scores, dim=1)
        elif normalize == "sigmoid":
            a_ = torch.sigmoid(scores)
            a = a_ / (a_.sum(dim=1, keepdim=True) + 1e-12)
        elif normalize == "l2":
            a = scores / (scores.norm(p=2, dim=1, keepdim=True) + 1e-12)
            a = (a + 1) / 2
            a = a / (a.sum(dim=1, keepdim=True) + 1e-12)
        elif normalize == "none":
            # 不规范化：退化为等权
            a = torch.full_like(scores, 1.0 / T)
        else:
            a = torch.softmax(scores, dim=1)  # 安全回退

        a = a.unsqueeze(-1)  # (N, T, 1)

        # ------- 4) 融合：视图加权求和 -> (N, d) -------
        S_bar = (a * S_tilde).sum(dim=1)  # (N, d)
        S_bar = F.normalize(S_bar, p=2, dim=1)

        return S_bar


    def forward(self, xs):
        import torch.nn.functional as Ff
        # ---- 关键改动开始 ----
        def _freeze(module):
            flags = [P.requires_grad for P in module.parameters()]
            for P in module.parameters(): P.requires_grad = False
            return flags

        def _unfreeze(module, flags):
            for P, f in zip(module.parameters(), flags): P.requires_grad = f
        device = self.device
        T = self.view
        eps = 1e-12

        # 1) 编码/重构/投影
        Z_list, xrs = [], []
        for v in range(T):
            x = xs[v].to(device) if isinstance(xs[v], torch.Tensor) else xs[v]
            z_low = self.encoders[v](x)  # (N, low_feature_dim)
            xr = self.decoders[v](z_low)  # (N, input_dim_v)
            xrs.append(xr)
            z_high = Ff.normalize(z_low, p=2, dim=1)
            Z_list.append(z_high)

        # 2) 门控拆分
        pfs, sfs, ss, ps, hs = [], [], [], [],[]
        for v in range(T):
            Z_v = Z_list[v]
            S_v, P_v, _, _ = self.split_by_gate_single(Z_v, Z_list, v)
            s = Ff.normalize(S_v, p=2, dim=1)
            p = Ff.normalize(P_v, p=2, dim=1)
            sfs.append(s)
            pfs.append(p)
            h = self.label_p(s)
            hs.append(h)

            # q 分支，临时冻结 label_p，使其当常数用
            _flags = _freeze(self.label_p)
            p1 = self.label_p(p)
            ps.append(p1)
            _unfreeze(self.label_p, _flags)

        # 3) 视图级 SE 融合
        S_bar = self.fuse_views_se1(sfs, normalize="softmax")  # (N,d)
        S = S_bar
        #S = self.se_mlp(S_bar)
        # l 分支，允许训练 label_p
        L = self.label_G(S_bar)

        # ---- 关键改动结束 ----

        return xrs, pfs, sfs, ps, S, hs,L




