import torch.nn as nn
import torch
from einops import rearrange
        
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=dilation, groups=groups, bias=False, dilation=dilation)
    
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    
class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            dilation = 1
            # raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(torch.nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, strides=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.strides = strides
        if self.strides is None:
            self.strides = [2, 2, 2, 2, 2]

        self.inplanes = 256
        self.dilation = 1
        
        self.groups = groups
        self.base_width = width_per_group
        

        self.layer1 = self._make_layer(block, 256, layers[0],dilate=replace_stride_with_dilation[0])
        self.layer2 = self._make_layer(block, 512, layers[1], stride=self.strides[2],dilate=replace_stride_with_dilation[1])
        self.layer3 = self._make_layer(block, 1024, layers[2], stride=self.strides[3],dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # print("x : ",x.shape)
        x1 = self.layer1(x)
        # print("x1 : ",x1.shape)
        x2 = self.layer2(x1)
        # print("x2 : ",x2.shape)
        x3 = self.layer3(x2)
        # print("x3 : ",x3.shape)

        return x1,x2,x3

    def forward(self, x):
        return self._forward_impl(x)
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x
class Residual2(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(x, x2, **kwargs) + x
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, x2, **kwargs):
        return self.fn(self.norm(x), self.norm(x2), **kwargs)
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
class Cross_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., softmax=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.softmax = softmax
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, m, mask = None):

        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x)
        k = self.to_k(m)
        v = self.to_v(m)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), [q,k,v])

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.softmax:
            attn = dots.softmax(dim=-1)
        else:
            attn = dots
        # attn = dots
        # vis_tmp(dots)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        # vis_tmp2(out)

        return out
class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, softmax=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual2(PreNorm2(dim, Cross_Attention(dim, heads = heads,
                                                        dim_head = dim_head, dropout = dropout,
                                                        softmax=softmax))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, m, mask = None):
        """target(query), memory"""
        for attn, ff in self.layers:
            x = attn(x, m, mask = mask)
            x = ff(x)
        return x


class CrossAttention_new(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention_new, self).__init__()
        # Use multi-head attention to capture the differences between two sets of embeddings
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        # Linear layer to project the output of attention
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.gate = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.Sigmoid()  # Gate between 0 and 1
        )
        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize Linear and Multi-head Attention weights using Xavier initialization
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.multihead_attn.in_proj_weight)
        nn.init.xavier_uniform_(self.multihead_attn.out_proj.weight)

        # Initialize biases
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.multihead_attn.in_proj_bias)
        nn.init.zeros_(self.multihead_attn.out_proj.bias)

        # Initialize convolutional layer weights for gating mechanism using Kaiming initialization
        for layer in self.gate:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='sigmoid')  # Kaiming init for Conv2d
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Set bias to 0 for Conv2d

    def forward(self, image_embeddings1, image_embeddings2):
        # Reshape the input to (batch_size, seq_len, embed_dim)
        b, c, h, w = image_embeddings1.shape
        t_image_embeddings1 = image_embeddings1.view(b, c, -1).permute(2, 0, 1)  # (seq_len, batch, embed_dim)
        t_image_embeddings2 = image_embeddings2.view(b, c, -1).permute(2, 0, 1)

        # Perform cross-attention: image_embeddings1 attends to image_embeddings2
        attn_output, _ = self.multihead_attn(t_image_embeddings1, t_image_embeddings2, t_image_embeddings2)

        # Project the attention output and reshape back
        attn_output = self.proj(attn_output)
        attn_output = attn_output.permute(1, 2, 0).view(b, c, h, w)  # Back to (batch, channels, height, width)

        # output = image_embeddings2 + attn_output

        gate = self.gate(image_embeddings2)
        output = gate * attn_output + (1 - gate) * image_embeddings2

        return output

class cross_attention(torch.nn.Module):
    def __init__(self):
        super(cross_attention, self).__init__()
        kwargs = {'replace_stride_with_dilation': [False, False, False]}
        self.resnet = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
        self.token_len = 4

        self.enc_depth = 1
        self.dec_depth = 8

        self.dim_head = 64
        self.decoder_dim_head = 64

        self.with_pos = 'learned'
        self.with_decoder_pos = 'learned'

        dim_1, dim_2, dim_3 = 128, 128, 128
        self.conv_squeeze_1 = nn.Sequential(nn.Conv2d(256, dim_1, kernel_size=1, padding=0, bias=False), nn.ReLU())
        self.conv_squeeze_2 = nn.Sequential(nn.Conv2d(512, dim_2, kernel_size=1, padding=0, bias=False), nn.ReLU())
        self.conv_squeeze_3 = nn.Sequential(nn.Conv2d(1024, dim_3, kernel_size=1, padding=0, bias=False), nn.ReLU())
        self.conv_squeeze_layers = [self.conv_squeeze_1, self.conv_squeeze_2, self.conv_squeeze_3]

        self.conv_token_1 = nn.Conv2d(dim_1, self.token_len, kernel_size=1, padding=0, bias=False)
        self.conv_token_2 = nn.Conv2d(dim_2, self.token_len, kernel_size=1, padding=0, bias=False)
        self.conv_token_3 = nn.Conv2d(dim_3, self.token_len, kernel_size=1, padding=0, bias=False)
        self.conv_tokens_layers = [self.conv_token_1, self.conv_token_2, self.conv_token_3]

        self.pos_embedding_1 = nn.Parameter(torch.randn(1, self.token_len * 2, dim_1))
        self.pos_embedding_2 = nn.Parameter(torch.randn(1, self.token_len * 2, dim_2))
        self.pos_embedding_3 = nn.Parameter(torch.randn(1, self.token_len * 2, dim_3))
        # self.pos_embedding_layers = [self.pos_embedding_1, self.pos_embedding_2, self.pos_embedding_3]

        self.transformer_1 = Transformer(dim=dim_1, depth=self.enc_depth, heads=4, dim_head=self.dim_head,
                                         mlp_dim=dim_1, dropout=0)
        self.transformer_decoder_1 = TransformerDecoder(dim=dim_1, depth=4, heads=4, dim_head=self.decoder_dim_head,
                                                        mlp_dim=dim_1, dropout=0, softmax=True)
        self.transformer_2 = Transformer(dim=dim_2, depth=self.enc_depth, heads=4, dim_head=self.dim_head,
                                         mlp_dim=dim_2, dropout=0)
        self.transformer_decoder_2 = TransformerDecoder(dim=dim_2, depth=4, heads=4, dim_head=self.decoder_dim_head,
                                                        mlp_dim=dim_2, dropout=0, softmax=True)
        self.transformer_3 = Transformer(dim=dim_3, depth=self.enc_depth, heads=8, dim_head=self.dim_head,
                                         mlp_dim=dim_3, dropout=0)
        self.transformer_decoder_3 = TransformerDecoder(dim=dim_3, depth=8, heads=8, dim_head=self.decoder_dim_head,
                                                        mlp_dim=dim_3, dropout=0, softmax=True)

        self.transformer_layers = [self.transformer_1, self.transformer_2, self.transformer_3]
        self.transformer_decoder_layers = [self.transformer_decoder_1, self.transformer_decoder_2,
                                           self.transformer_decoder_3]

        self.pos_embedding_decoder_1 = nn.Parameter(torch.randn(1, dim_1, 64, 64))
        self.pos_embedding_decoder_2 = nn.Parameter(torch.randn(1, dim_2, 32, 32))
        self.pos_embedding_decoder_3 = nn.Parameter(torch.randn(1, dim_3, 16, 16))

        # self.pos_embedding_decoder_layers = [self.pos_embedding_decoder_1, self.pos_embedding_decoder_2,
        #                                      self.pos_embedding_decoder_3]

        self.conv_decode_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False)
        self.conv_decode_2 = nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False)
        self.conv_decode_3 = nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False)
        self.conv_decode_layers = [self.conv_decode_1, self.conv_decode_2, self.conv_decode_3]

        # self.upsamplex2 = nn.Upsample(scale_factor=2)
        # self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0, bias=False)
        self.upsamplex4 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=4, padding=0, bias=False)

        self.conv_layer2_0 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=3 // 2, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 128, kernel_size=3, padding=3 // 2, stride=1)
        )

        self.channel_scaler = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=3 // 2, stride=1, bias=False),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
        )

        self.final_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
        )
        
        # self.bn_x1 = nn.BatchNorm2d(256)
        # self.bn_x2= nn.BatchNorm2d(256)

    def _forward_semantic_tokens(self, x, layer=None):
        b, c, h, w = x.shape
        spatial_attention = self.conv_tokens_layers[layer](x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)
        return tokens

    def _forward_transformer(self, x, layer):
        if self.with_pos:
            if layer == 0:
                x = x + self.pos_embedding_1
            elif layer == 1:
                x = x + self.pos_embedding_2
            elif layer == 2:
                x = x + self.pos_embedding_3

            # x += self.pos_embedding_layers[layer]
        x = self.transformer_layers[layer](x)
        return x

    def _forward_transformer_decoder(self, x, m, layer):
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'learned':
            if layer == 0:
                x = x + self.pos_embedding_decoder_1
            elif layer == 1:
                x = x + self.pos_embedding_decoder_2
            elif layer == 2:
                x = x + self.pos_embedding_decoder_3
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder_layers[layer](x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_trans_module(self, x1, x2, layer):
        
        x1 = self.conv_squeeze_layers[layer](x1)
        x2 = self.conv_squeeze_layers[layer](x2)
        
        token1 = self._forward_semantic_tokens(x1, layer)
        token2 = self._forward_semantic_tokens(x2, layer)
        
        self.tokens_ = torch.cat([token1, token2], dim=1)
        self.tokens = self._forward_transformer(self.tokens_, layer)
        token1, token2 = self.tokens.chunk(2, dim=1)
        
        
        x1 = self._forward_transformer_decoder(x1, token1, layer)
        x2 = self._forward_transformer_decoder(x2, token2, layer)
        
        
        diff_token = torch.abs(token2 - token1)
        
        diff_x = self.conv_decode_layers[layer](torch.cat([x1, x2], axis=1))
        x = self._forward_transformer_decoder(diff_x, diff_token, layer)
        return x
        
    # def forward(self, x1, x2):
    #     out_3 = self._forward_trans_module(x1, x2, layer=0)
    #     return out_3
        

    def forward(self, x1, x2):

        # x1 = self.bn_x1(x1)
        # x2 = self.bn_x2(x2)

        x1_256_64, x1_512_32, x1_1024_16 = self.resnet(x1)
        x2_256_64, x2_512_32, x2_1024_16 = self.resnet(x2)

        # print("---------------")
        # print(x1_1024_16.shape)
        # print(x1_512_32.shape)
        # print(x1_256_64.shape)
        # print("---------------")

        x1 = x1_1024_16
        x2 = x2_1024_16
        out_2 = self._forward_trans_module(x1, x2, 2)
        out_2 = self.upsamplex2(out_2)
        # out_2 = out_2.to(dtype=torch.bfloat16)
        


        # print("out_2 : ",out_2.shape)

        x1 = x1_512_32
        x2 = x2_512_32
        out_1 = self._forward_trans_module(x1, x2, 1)
        out_1 = out_1 + out_2
        out_1 = self.upsamplex2(out_1)
        # out_1 = out_1.to(dtype=torch.bfloat16)
        # print("out_1 : ",out_1.shape)

        x1 = x1_256_64
        x2 = x2_256_64

        # out_0 = self.conv_layer2_0(torch.cat([x1, x2], 1))
        out_0 = self._forward_trans_module(x1, x2, 0)
        out_0 = out_1 + out_0
        # print("out_0 : ",out_0.shape)

        out = self.channel_scaler(out_0)

        out = torch.cat([x2,out], 1)
        out = self.final_layer(out)

        # print("out : ",out.shape)
        return out
