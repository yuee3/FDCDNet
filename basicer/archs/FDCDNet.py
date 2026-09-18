import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_
from pdb import set_trace as stx
import numbers
import math
from timm.layers import DropPath, to_2tuple, trunc_normal_
from typing import Optional, Callable
from einops import rearrange, repeat
from .expertFN4 import FCFFN

NEG_INF = -1000000
device_id0 = 'cuda:0'
device_id1 = 'cuda:1'

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        original_shape = hidden_states.shape

        if len(original_shape) == 4:  
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(-1, original_shape[1])

        elif len(original_shape) == 3:
            pass

        else:
            raise ValueError(f"Unsupported input shape: {original_shape}")

        # 计算 RMS
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # 应用缩放
        hidden_states = hidden_states * self.weight

        # 恢复原始形状
        if len(original_shape) == 4:
            hidden_states = hidden_states.reshape(
                original_shape[0], original_shape[2], original_shape[3], original_shape[1]
            ).permute(0, 3, 1, 2)

        return hidden_states


class FDEM(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=8, sr_ratio=2, qkv_bias=False, qk_scale=None, LCM_x_sr_ratio=2,
                 **kwargs):  # 添加 **kwargs 接收多余参数
        super().__init__()

        print("\n" + "=" * 60)
        print(f"!!! [DIAGNOSTIC PROBE] NEW FDEM (Frequency Separation) is being initialized with:")
        print(f"    - Received total dim: {dim}")

        if 'alpha' in kwargs:
            print(
                f"    - Note: Parameter 'alpha'={kwargs['alpha']} is received but no longer used in this new version.")

        if dim % 2 != 0:
            raise ValueError(f"The dimension ({dim}) must be an even number for splitting into two branches.")
        self.proj_dim = dim // 2

        print(f"    - ==> Calculated projection dim for each branch: {self.proj_dim}")
        # ===============================================================

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.sr_ratio = sr_ratio  # Lo-fi K,V downsample ratio
        self.LCM_x_sr_ratio = LCM_x_sr_ratio  # Hi-fi feature downsample ratio

        # 1. 全局初始降维层
        self.initial_proj = nn.Linear(dim, self.proj_dim)

        # 2. Lo-Fi 分支的参数和层 (操作维度为 proj_dim)
        head_dim = self.proj_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.l_q = nn.Linear(self.proj_dim, self.proj_dim, bias=qkv_bias)
        self.l_kv = nn.Linear(self.proj_dim, self.proj_dim * 2, bias=qkv_bias)
        self.l_proj = nn.Linear(self.proj_dim, self.proj_dim)

        # 3. Hi-Fi 分支的参数和层 (操作维度为 proj_dim)
        self.h_conv_branch = LCM(dim=self.proj_dim)

        # 4. 最终融合层
        self.final_proj = nn.Linear(dim, dim)

        # ======================= 添加的诊断代码 =======================
        print(f"    - Lo-Fi branch will operate with {num_heads} heads on dim {self.proj_dim}")
        print(f"    - Hi-Fi branch will operate with convolutions on dim {self.proj_dim}")
        print("=" * 60 + "\n")
        # ===============================================================

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape

        # 1. 全局初始降维
        x_proj = self.initial_proj(x)  # [B, N, C/2]

        # 2. Lo-Fi 分支计算
        CDA_out = self.CDA(x_proj, H, W)  # [B, N, C/2]

        # 3. Hi-Fi 分支计算
        LCM_x_out = self.LCM_x(x_proj, H, W)  # [B, N, C/2]

        # 4. 最终拼接与融合
        # 注意：在新设计中，LCM_x_out 和 CDA_out 的顺序可以调整，可能会影响性能
        fused = torch.cat((LCM_x_out, CDA_out), dim=-1)  # [B, N, C]
        output = self.final_proj(fused)

        return output

    def CDA(self, x, H, W):

        B, N, C = x.shape  # Here C is proj_dim
        ws = self.window_size

        assert H % ws == 0 and W % ws == 0, f"H({H}) and W({W}) must be divisible by window_size({ws})"

        x_reshaped = x.reshape(B, H, W, C)
        num_windows_h = H // ws
        num_windows_w = W // ws
        num_windows = num_windows_h * num_windows_w
        x_windows = x_reshaped.reshape(B, num_windows_h, ws, num_windows_w, ws, C).permute(0, 1, 3, 2, 4,
                                                                                           5).contiguous()
        x_windows = x_windows.reshape(B * num_windows, ws * ws, C)


        q = self.l_q(x_windows).reshape(B * num_windows, ws * ws, self.num_heads, C // self.num_heads).permute(0, 2, 1,3)


        x_windows_2d = x_windows.reshape(B * num_windows, ws, ws, C).permute(0, 3, 1, 2)
        x_pooled = F.avg_pool2d(x_windows_2d, kernel_size=self.sr_ratio, stride=self.sr_ratio)
        x_pooled = x_pooled.reshape(B * num_windows, C, -1).permute(0, 2, 1)
        kv = self.l_kv(x_pooled).reshape(B * num_windows, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3,1, 4)
        k, v = kv[0], kv[1]


        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        CDA_x = (attn @ v).transpose(1, 2).reshape(B * num_windows, ws * ws, C)


        CDA_x = self.l_proj(CDA_x)


        CDA_x = CDA_x.reshape(B, num_windows_h, num_windows_w, ws, ws, C).permute(0, 1, 3, 2, 4, 5).contiguous()
        CDA_x = CDA_x.reshape(B, H * W, C)

        return CDA_x

    def LCM_x(self, x, H, W):
        """
        基于频率分离的 Hi-Fi 卷积逻辑。
        输入 x 的通道维度是 self.proj_dim (C/2)。
        """
        # 1. 生成低频近似 (模糊版本)
        # 将输入从 [B, L, C] 格式转换为 [B, C, H, W]
        x_2d = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W).contiguous()

        # 下采样
        x_low_2d = F.avg_pool2d(x_2d, kernel_size=self.LCM_x_sr_ratio, stride=self.LCM_x_sr_ratio)

        # 上采样
        x_blurred_2d = F.interpolate(x_low_2d, size=(H, W), mode='bilinear', align_corners=False)

        # 将模糊版本转换回 [B, L, C] 格式
        x_blurred = rearrange(x_blurred_2d, 'b c h w -> b (h w) c').contiguous()

        # 2. 提取高频细节
        x_high = x - x_blurred

        # 3. 使用卷积模块处理高频细节
        LCM_x_out = self.h_conv_branch(x_high, H, W)

        return LCM_x_out


class LCM(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # 确保输入通道为偶数，以便能平均拆分
        if dim % 2 != 0:
            raise ValueError(f"The dimension ({dim}) must be an even number for splitting.")
        self.half_dim = dim // 2

        # 分支 1: 3×3 DConv → GeLU → 1×1 Conv (处理 C/2 通道)
        self.branch1_dconv = nn.Conv2d(
            self.half_dim, self.half_dim,
            kernel_size=3,
            padding=1,
            groups=self.half_dim,  # Depthwise
            bias=False
        )
        self.branch1_act = nn.GELU()
        self.branch1_pconv = nn.Conv2d(
            self.half_dim, self.half_dim,
            kernel_size=1,  # Pointwise
            bias=False
        )

        # 分支 2: 5×5 DConv → GeLU → 1×1 Conv (处理 C/2 通道)
        self.branch2_dconv = nn.Conv2d(
            self.half_dim, self.half_dim,
            kernel_size=5,
            padding=2,
            groups=self.half_dim,  # Depthwise
            bias=False
        )
        self.branch2_act = nn.GELU()
        self.branch2_pconv = nn.Conv2d(
            self.half_dim, self.half_dim,
            kernel_size=1,  # Pointwise
            bias=False
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:

        # 将输入从 [B, L, C] 格式转换为 [B, C, H, W]
        x_2d = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W).contiguous()

        # 1. Split: [B, C, H, W] → 两个 [B, C/2, H, W]
        x1, x2 = torch.chunk(x_2d, 2, dim=1)

        # 2. 分支 1: 3×3 DConv → GeLU → 1×1 Conv
        branch1 = self.branch1_dconv(x1)      # [B, C/2, H, W]
        branch1 = self.branch1_act(branch1)   # [B, C/2, H, W]
        branch1 = self.branch1_pconv(branch1) # [B, C/2, H, W]

        # 3. 分支 2: 5×5 DConv → GeLU → 1×1 Conv
        branch2 = self.branch2_dconv(x2)      # [B, C/2, H, W]
        branch2 = self.branch2_act(branch2)   # [B, C/2, H, W]
        branch2 = self.branch2_pconv(branch2) # [B, C/2, H, W]

        # 4. Concat: 拼接两个分支
        output_2d = torch.cat([branch1, branch2], dim=1)  # [B, C, H, W]

        # 将输出转换回 [B, L, C] 格式
        output = rearrange(output_2d, 'b c h w -> b (h w) c').contiguous()

        return output


class FDCBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            drop_path: float = 0.,
            hilo_window_size: int = 2,
            hilo_alpha: float = 0.5,
            bias: bool = False,
            wt_levels: int = 1, 
            wt_type: str = 'db1',
            **kwargs,
    ):
        super().__init__()
        self.dim = hidden_dim
        self.norm1 = LlamaRMSNorm(self.dim)

        self.mixer = FDEM(
            dim=self.dim,
            num_heads=num_heads,
            alpha=hilo_alpha
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = LlamaRMSNorm(self.dim)
        self.mlp = FCFFN(
            dim=hidden_dim,
            mlp_ratio=2.66,
            bias=bias,
            wt_levels=wt_levels,
            wt_type=wt_type,
        )

    def forward(self, x: torch.Tensor, x_size):
        H, W = x_size

        shortcut = x
        x_norm1 = self.norm1(x)
        mixed_x = self.mixer(x_norm1, H, W)
        x = shortcut + self.drop_path(mixed_x)

        shortcut2 = x
        x_norm2 = self.norm2(x)
        mlp_x = self.mlp(x_norm2, H, W)
        x = shortcut2 + self.drop_path(mlp_x)

        return x


##########################################################################
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x


##########################################################################
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x, H, W):
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        x = self.body(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x



class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x, H, W):
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        x = self.body(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        return x


class FDCDNet(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[2, 3, 4],
                 num_heads=[2, 4, 6],  # 不同层的头数
                 window_size=8,  # 添加参数
                 mlp_ratio=4.,
                 num_refinement_blocks=4,
                 use_shift_window=True,  
                 qkv_bias=True,
                 use_checkpoint=False,
                 drop_path_rate=0.,
                 bias=False,
                 dual_pixel_task=False,
                 shift_window=True, 
                 hilo_window_size=2,  # HiLo窗口大小
                 hilo_alpha=0.5,
                 **kwargs  # 添加 **kwargs 以接受任何额外参数
                 ):

        super(FDCDNet, self).__init__()
        self.mlp_ratio = mlp_ratio
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        norm_layer = LlamaRMSNorm

        # 计算不同层的drop path率
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_blocks))]

        self.encoder_level1 = nn.ModuleList([
            FDCBlock(
                hidden_dim=dim,
                drop_path=dpr[i],
                attn_drop_rate=0,
                num_heads=num_heads[0],
                mlp_ratio=self.mlp_ratio,
                window_size=window_size,  
                shift_size=window_size // 2 if shift_window and i % 2 == 1 else 0,
                hilo_alpha=hilo_alpha,
            )
            for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)

        # 编码器第2层
        self.encoder_level2 = nn.ModuleList([
            FDCBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=dpr[i + num_blocks[0]],
                attn_drop_rate=0,
                num_heads=num_heads[1],
                mlp_ratio=self.mlp_ratio,
                window_size=window_size, 
                shift_size=window_size // 2 if shift_window and i % 2 == 1 else 0,
                hilo_alpha=hilo_alpha,
            )
            for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))

        self.latent = nn.ModuleList([
            FDCBlock(
                hidden_dim=int(dim * 2 ** 2),
                drop_path=dpr[i + num_blocks[0] + num_blocks[1]],
                attn_drop_rate=0,
                num_heads=num_heads[2],
                mlp_ratio=self.mlp_ratio,
                window_size=window_size, 
                shift_size=window_size // 2 if shift_window and i % 2 == 1 else 0,
                hilo_alpha=hilo_alpha,
            )
            for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)

        self.decoder_level2 = nn.ModuleList([
            FDCBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=dpr[i],
                attn_drop_rate=0,
                num_heads=num_heads[1],
                mlp_ratio=self.mlp_ratio,
                window_size=window_size, 
                shift_size=window_size // 2 if shift_window and i % 2 == 1 else 0,
                hilo_alpha=hilo_alpha,
            )
            for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.ModuleList([
            FDCBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=dpr[i],
                attn_drop_rate=0,
                num_heads=num_heads[0],
                mlp_ratio=self.mlp_ratio,
                window_size=window_size, 
                shift_size=window_size // 2 if shift_window and i % 2 == 1 else 0,
                hilo_alpha=hilo_alpha,
            )
            for i in range(num_blocks[0])])

        self.refinement = nn.ModuleList([
            FDCBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=dpr[i],
                attn_drop_rate=0,
                num_heads=num_heads[0],
                mlp_ratio=self.mlp_ratio,
                window_size=window_size, 
                shift_size=window_size // 2 if shift_window and i % 2 == 1 else 0,
                hilo_alpha=hilo_alpha,
            )
            for i in range(num_refinement_blocks)])

        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)

  
        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)


    def forward(self, inp_img):
        _, _, H, W = inp_img.shape
        inp_enc_level1 = self.patch_embed(inp_img)  
        out_enc_level1 = inp_enc_level1

        for layer in self.encoder_level1:
            out_enc_level1 = layer(out_enc_level1, [H, W])

        inp_enc_level2 = self.down1_2(out_enc_level1, H, W)  


        out_enc_level2 = inp_enc_level2

        for layer in self.encoder_level2:
            out_enc_level2 = layer(out_enc_level2, [H // 2, W // 2])

        inp_enc_level3 = self.down2_3(out_enc_level2, H // 2, W // 2)  

        latent = inp_enc_level3

        for layer in self.latent:
            latent = layer(latent, [H // 4, W // 4])

        inp_dec_level2 = self.up3_2(latent, H // 4, W // 4)  # b, hw//4, 2c
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 2)
        inp_dec_level2 = rearrange(inp_dec_level2, "b (h w) c -> b c h w", h=H // 2, w=W // 2).contiguous()
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        inp_dec_level2 = rearrange(inp_dec_level2, "b c h w -> b (h w) c").contiguous()  # b, hw//4, 2c

        out_dec_level2 = inp_dec_level2
        for layer in self.decoder_level2:
            out_dec_level2 = layer(out_dec_level2, [H // 2, W // 2])

        inp_dec_level1 = self.up2_1(out_dec_level2, H // 2, W // 2)  # b, hw, c
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 2)
        out_dec_level1 = inp_dec_level1

        for layer in self.decoder_level1:
            out_dec_level1 = layer(out_dec_level1, [H, W])

        for layer in self.refinement:
            out_dec_level1 = layer(out_dec_level1, [H, W])

        out_dec_level1 = rearrange(out_dec_level1, "b (h w) c -> b c h w", h=H, w=W).contiguous()

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1



