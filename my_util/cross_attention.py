import torch
import torch.nn as nn
import einops


if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
print(f'attention mode is {ATTENTION_MODE}')



class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,patch_size=4,in_chans =6,embed_dim = 768 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv1 = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.qkv2 = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.patch_dim = patch_size ** 2 * in_chans
        self.decoder_pred = nn.Linear(embed_dim, self.patch_dim, bias=True)

    def unpatchify(self, x, channels=48):
        patch_size = int((x.shape[2] // channels) ** 0.5)
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]
        x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)', h=h, p1=patch_size, p2=patch_size)
        return x

    def forward(self, x1, x2):

        B1, L1, C1 = x1.shape
        B2, L2, C2 = x2.shape

        qkv1 = self.qkv1(x1)
        qkv2 = self.qkv2(x2)

        if ATTENTION_MODE == 'flash':
            # qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
            # q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            # x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            # x = einops.rearrange(x, 'B H L D -> B L (H D)')
            pass
        elif ATTENTION_MODE == 'xformers':
            qkv = einops.rearrange(qkv1, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        elif ATTENTION_MODE == 'math':
            qkv1 = einops.rearrange(qkv1, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]  # B H L D
            qkv2 = einops.rearrange(qkv2, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]  # B H L D
            attn = (q1 @ k2.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v2).transpose(1, 2).reshape(B1, L1, C1)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        x = self.unpatchify(x)
        return x

# 输入通道应一致
img1 = torch.randn((64, 6, 32, 32))
img2 = torch.randn((64, 3, 32, 32))
# img1_flatten =img1.reshape(64, 32*32*6, 1)
# img2_flatten =img2.reshape(64, 32*32*3, 1)
patch_emb1 = PatchEmbed(4,6,768)
patch_emb2 = PatchEmbed(4,3,768)
img1_flatten = patch_emb1(img1)
img2_flatten = patch_emb2(img2)

cross_attn = Attention(dim=768)
output = cross_attn(img1_flatten,img2_flatten)

print(".")

