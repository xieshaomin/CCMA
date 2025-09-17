from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

import torch.nn.functional as F
from torchvision.transforms import Resize



class CrossModelAtt(nn.Module):
    def __init__(self, feature_dim, height, width):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, img_feat, text_feat):
        """
        img_feat: [B, C, H, W] 图像特征
        text_feat: [B, C, H, W] 文本特征
        """
        #img_feat 的形状为 [B, C, H, W] 其中B是批量大小，C是通道数，H和W是特征图的高度和宽度
        B, C, H, W = img_feat.shape

        #1 特征图展平
        q = img_feat.view(B, C, -1)  # [B, C, H*W]
        k = text_feat.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]

        attention_map = torch.bmm(q, k)  # [B, C, C]
        attention_map = self.softmax(attention_map)  # [B, C, C]

        v = text_feat.view(B, C, -1)  # [B, C, H*W]
        attention_info = torch.bmm(attention_map, v)  # [B, C, H*W]

        attention_info = attention_info.view(B,C,H,W)
        output = self.gamma * attention_info + img_feat  # [B, C, H, W]
        return output 

class CrossModelAtt_Text(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, text_feat, img_feat):
        B, C, H_t, W_t = text_feat.shape
        _, _, H, W = img_feat.shape

        q = text_feat.view(B, -1, C)                # [B, 77, 512]
        k = img_feat.view(B, -1, C).transpose(1, 2) # [B, 512, 192]
        attention_map = torch.bmm(q, k)            # [B, 77, 192]
        attention_map = self.softmax(attention_map)

        v = img_feat.view(B, -1, C)                # [B, 192, 512]
        attention_info = torch.bmm(attention_map, v)  # [B, 77, 512]
        attention_info = attention_info.transpose(1, 2).contiguous().view(B, C, H_t, W_t)

        output = self.gamma * attention_info + text_feat
        return output

class FusionModule(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_proj = nn.Linear(embed_dim, embed_dim * 2)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, query_feats, context_feats):
        dtype = self.q_proj.weight.dtype
        query_feats = query_feats.to(dtype=dtype)
        context_feats = context_feats.to(dtype=dtype)

        q = self.q_proj(query_feats)
        k, v = self.kv_proj(context_feats).chunk(2, dim=-1)
        attn_output, _ = self.attn(q, k, v)

        attn_output = attn_output + query_feats.to(dtype=attn_output.dtype)
        attn_output = self.norm(attn_output.to(dtype=self.norm.weight.dtype))

        ffn_input = attn_output.to(dtype=self.ffn[0].weight.dtype)
        ffn_output = self.ffn(ffn_input)

        output = ffn_output + attn_output

        # ✅ 修复 view 报错：加 .contiguous() 或用 .reshape()
        L = output.size(0)
        H = int(L ** 0.5)
        W = int(L / H)
        return output.transpose(0, 1).contiguous().view(-1, 1, H, W) 

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# RMSNorm Layer
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    """
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.weight is not None}'


# Differential Attention Module
class DiffAttention(nn.Module):
    """
    Differential Attention Module.

    Given an input tensor X ∈ ℝ^(B×N×d_model), we first compute the linear projections:
        Q = X Wᵠ, K = X Wᵏ, V = X Wᵛ

    The queries and keys are then reshaped and split into two parts:
        Q → [Q₁; Q₂] ∈ ℝ^(B, N, 2·h_effective, d_head)
        K → [K₁; K₂] ∈ ℝ^(B, N, 2·h_effective, d_head)
    with h_effective = num_heads // 2 and d_head = d_model / num_heads.

    The value projection is reshaped to:
        V ∈ ℝ^(B, N, h_effective, 2·d_head)

    We then compute two attention maps:
        A₁ = softmax((Q₁ K₁ᵀ) / √d_head)
        A₂ = softmax((Q₂ K₂ᵀ) / √d_head)

    A learnable scalar λ is computed via:
        λ = exp(λ_{q1} ⋅ λ_{k1}) − exp(λ_{q2} ⋅ λ_{k2}) + λ_init

    Finally, the differential attention output is:
        DiffAttn(X) = (A₁ − λ · A₂) · V

    The per-head outputs are then normalized headwise with RMSNorm and projected back to d_model.

    Args:
        dim (int): Embedding dimension (d_model).
        num_heads (int): Number of heads in the original transformer (must be even).
        qkv_bias (bool): If True, add a bias term to the Q, K, V projections.
        attn_drop (float): Dropout probability after softmax.
        proj_drop (float): Dropout probability after the output projection.
        lambda_init (float): Initial constant for lambda re-parameterization.
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, attn_drop: float = 0., proj_drop: float = 0., lambda_init: float = 0.8):
        super().__init__()
        if num_heads % 2 != 0:
            raise ValueError("num_heads must be even for Differential Attention.")
        self.dim = dim
        self.num_heads = num_heads  # original number of heads
        self.effective_heads = num_heads // 2  # differential attention operates on half as many heads
        self.head_dim = dim // num_heads  # per-head dimension
        self.scaling = self.head_dim ** -0.5

        # Linear projections for Q, K, V: mapping from dim → dim.
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim, bias=True)  # final output projection

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # RMSNorm for headwise normalization on outputs (each head's output has dimension 2·head_dim)
        self.diff_norm = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

        # Learnable lambda parameters (shared across all heads)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_init = lambda_init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, N, d_model).

        Returns:
            Tensor of shape (B, N, d_model) after applying differential attention.
        """
        B, N, _ = x.shape

        # Compute Q, K, V projections.
        q = self.q_proj(x)  # shape: (B, N, d_model)
        k = self.k_proj(x)  # shape: (B, N, d_model)
        v = self.v_proj(x)  # shape: (B, N, d_model)

        # Reshape Q and K into (B, N, 2 * h_effective, head_dim)
        q = q.view(B, N, 2 * self.effective_heads, self.head_dim)
        k = k.view(B, N, 2 * self.effective_heads, self.head_dim)
        # Reshape V into (B, N, h_effective, 2 * head_dim)
        v = v.view(B, N, self.effective_heads, 2 * self.head_dim)

        # Transpose to bring head dimension forward.
        # q, k: (B, 2 * h_effective, N, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        # v: (B, h_effective, N, 2 * head_dim)
        v = v.transpose(1, 2)

        # Scale Q.
        q = q * self.scaling

        # Compute raw attention scores: (B, 2 * h_effective, N, N)
        attn_scores = torch.matmul(q, k.transpose(-1, -2))

        # Compute attention probabilities.
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_drop(attn_probs)

        # Reshape to separate the two halves: (B, h_effective, 2, N, N)
        attn_probs = attn_probs.view(B, self.effective_heads, 2, N, N)

        # Compute lambda via re-parameterization.
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2))
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # Differential attention: subtract the second attention map scaled by lambda_full.
        diff_attn = attn_probs[:, :, 0, :, :] - lambda_full * attn_probs[:, :, 1, :, :]  # shape: (B, h_effective, N, N)

        # Multiply the differential attention weights with V.
        attn_output = torch.matmul(diff_attn, v)  # shape: (B, h_effective, N, 2 * head_dim)

        # Apply RMSNorm (headwise normalization) and scale by (1 - lambda_init)
        attn_output = self.diff_norm(attn_output) * (1 - self.lambda_init)

        # Concatenate heads: reshape from (B, h_effective, N, 2 * head_dim) → (B, N, 2 * h_effective * head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, 2 * self.effective_heads * self.head_dim)

        # Final linear projection.
        x_out = self.out_proj(attn_output)
        x_out = self.proj_drop(x_out)
        return x_out

class CCMA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 

        self.cross_modal_att = CrossModelAtt(feature_dim=512, height=16, width=12)
        self.cross_modal_att_text = CrossModelAtt_Text()

        self.fusion_module_tg = FusionModule(embed_dim=512) 
        self.DiffAttention = DiffAttention(dim=12, num_heads=12, qkv_bias=True, attn_drop=0.1, proj_drop=0.1, lambda_init=0.8).cuda()
        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def forward(self, batch):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, text_feats = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        
        # 特征融合
        # 1. 提取 patch tokens 并 reshape 成 2D 图像特征图
        patch_tokens = image_feats[:, 1:, :]  # [B, 192, 512]
        B, N, C = patch_tokens.shape
        H, W = 16, 12  # 显式设置
        # assert H * W == N, "Patch tokens数量无法reshape为H×W"
        img_feat_2d = patch_tokens.permute(0, 2, 1).contiguous().view(B, C, H, W)  # [B, 512, H, W]

        # 2. 将文本 token 聚合成与图像匹配的格式（你可以使用 mean pooling 或 reshape）
        # 下面是简单平均+reshape为图像特征形状
        text_tokens = text_feats  # [B, 77, 512]
        text_feat_flat = text_tokens.mean(dim=1)  # [B, 512]
        text_feat_2d = text_feat_flat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)  # [B, 512, H, W]

        # 3. 跨模态融合
        fused_feat_img = self.cross_modal_att(img_feat_2d, text_feat_2d)  # [B, 512, H, W]


        patch_tokens = image_feats[:, 1:, :]  # [B, 192, 512]
        B, N, C = patch_tokens.shape

        # 设置图像特征图的 H, W（16×12 是 192 的因子）
        H, W = 16, 12
        assert H * W == N, "Patch tokens数量无法reshape为H×W"

        # 图像特征 reshape 成 2D 格式 [B, 512, 16, 12]
        img_token_feat = patch_tokens.permute(0, 2, 1).contiguous().view(B, C, H, W)

        # 文本特征 = [B, 77, 512]，尝试 reshape 为最接近的 2D 格式
        text_tokens = text_feats  # [B, 77, 512]
        B, N_t, D = text_tokens.shape

        # 找到最接近 77 的因子组合（推荐 H=11, W=7）
        H_t, W_t = 11, 7
        assert H_t * W_t == N_t, "Text token数量无法reshape为 H×W"

        # 文本特征 reshape 成 2D 格式 [B, 512, 11, 7]
        text_token_feat = text_tokens.permute(0, 2, 1).contiguous().view(B, D, H_t, W_t)
        text_feat_fused = self.cross_modal_att_text(text_token_feat, img_token_feat)  # 图像引导文本
        # print(f"[DEBUG] fused_feat.shape: {fused_feat_img.shape}")
        # print(f"[DEBUG] text_feat_fused.shape: {text_feat_fused.shape}")
        # 融合后的图像特征
        fused_feat_pooled = F.adaptive_avg_pool2d(fused_feat_img, 1).squeeze(-1).squeeze(-1)  # [B, 512]

        # 融合后的文本特征
        text_feat_pooled = F.adaptive_avg_pool2d(text_feat_fused, 1).squeeze(-1).squeeze(-1)  # [B, 512]

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})
        if 'IMAL' in self.current_task:
            ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})
            ret.update({'sdm_loss':objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})
            fused_feat_pooled = self.DiffAttention(fused_feat_pooled, text_feat_pooled)
            text_feat_pooled = self.DiffAttention(text_feat_pooled).cuda()
            ret.update({'imal_loss':objectives.IMAL(fused_feat_pooled, text_feat_pooled, batch['pids'], logit_scale)})
            ret.update({'cmpm_loss':objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})

            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})

            mlm_ids = batch['mlm_ids']

            mlm_feats = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})
        return ret


def build_model(args, num_classes=11003):
    model = CCMA(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
