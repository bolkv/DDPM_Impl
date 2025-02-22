import torch
import torch.nn as nn
import math
import numpy as np

def get_timestep_embedding(timesteps, embedding_dim:int):
    '''
    Build sinusoidal embeddings
    positional embedding
    몇번쨰 timestep에 대한 timestep embedding이 얼마냐?
    '''
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)*-emb)
    emb = timesteps.type(torch.float32)[:,None] * emb[None, :]
    emb = torch.concat([torch.sin(emb),torch.cos(emb)],axis=1)

    return emb

class DownSampling(nn.Module):
    #C(channel) 수는 그대로 하되, 이미지 크기를 줄여나감
    def __init__(self, C):
        super().__init__()
        self.conv = nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1)
        # ((input + 2*padding - kernel_size )/stride) + 1
    
    def forward(self, x):
        B, C, H, W  = x.shape
        x = self.conv(x)
        if(x.shape != (B, C, H//2, W//2 )):
            print(x.shape)
            print(B,C,H//2,W//2)
        assert x.shape == (B, C, H//2, W//2 )
        return x


class UpSampling(nn.Module):
    #C(channel) 수는 그대로 하되, 이미지 크기를 키움
    def __init__(self, C):
        super().__init__()
        self.conv = nn.Conv2d(C, C, kernel_size=3, stride=1, padding = 1)

    def forward(self, x):
        B,C,H,W = x.shape
        x = nn.functional.interpolate(x, size = None, scale_factor=2, mode='nearest')
        x = self.conv(x)
        assert x.shape == (B, C, H*2, W*2 )
        return x

class Nin(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        scale = 1e-10
        n = (in_dim + out_dim) / 2
        limit = np.sqrt(3*scale/n)
        self.W = nn.Parameter(torch.zeros((in_dim, out_dim), dtype=torch.float32).uniform_(-limit, limit ))
        self.b = nn.Parameter(torch.zeros((1,out_dim,1,1), dtype=torch.float32))

    def forward(self, x):
        return torch.einsum('bchw, co ->bowh' , x , self.W) + self.b

class ResNetBlock(nn.Module):

    def __init__(self, in_ch, out_ch, dropout_rate=0.1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding =1)
        self.dense = nn.Linear(512, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding =1)

        if not (in_ch == out_ch):
            self.nin = Nin(in_ch, out_ch)

        self.dropout_rate = dropout_rate
        self.nonlinearity = nn.SiLU()

    def forward(self, x, temb): #temb: Batch, dim
        '''
        param x: (B, C, H, W)
        param temb: (B, dim)
        '''
        h = self.nonlinearity(nn.functional.group_norm(x,num_groups=32))
        h = self.conv1(h)

        #add timestep embedding
        h += self.dense(self.nonlinearity(temb))[:, :, None, None]

        h = self.nonlinearity(nn.functional.group_norm(h,num_groups=32))
        h = nn.functional.dropout(h, p=self.dropout_rate)
        h = self.conv2(h)

        if not (x.shape[1]==h.shape[1]):
            x = self.nin(x)

        return x + h
    
class AttentionBlock(nn.Module):

    def __init__(self, ch):
        super().__init__()
        self.Q = Nin(ch, ch)
        self.K = Nin(ch, ch)
        self.V = Nin(ch, ch)
        self.ch = ch
        self.nin = Nin(ch, ch)

    def forward(self, x):
        B, C ,H, W = x.shape

        assert C == self.ch

        h = nn.functional.group_norm(x, num_groups=32)
        q=self.Q(h)
        k=self.K(h)
        v=self.V(h)

        w = torch.einsum('bcwh, bcWH -> bwWhH', q , k) * (int(C)**(-0.5))
        w = torch.reshape(w, [B,H,W,H*W])
        w = torch.nn.functional.softmax(w, dim =-1)
        w = torch.reshape(w, [B,H,W,H,W])

        h = torch.einsum('bHWHW, bcWH -> bcWH', w , v)
        h= self.nin(h)

        assert x.shape == h.shape
        return x + h
    
class UNet(nn.Module):

    def __init__(self, ch=128, in_ch=1): #black white
        super().__init__()
        self.ch = ch
        self.linear1 = nn.Linear(ch, ch*4)
        self.linear2 = nn.Linear(ch*4, ch*4)

        self.conv1 = nn.Conv2d(in_ch, ch, 3, stride=1, padding=1)

        self.down = nn.ModuleList([
            ResNetBlock(1 * ch, 1* ch),
            ResNetBlock(1 * ch, 1 * ch),
            DownSampling(ch),
            ResNetBlock(1 * ch, 2 * ch),
            AttentionBlock(2 * ch),
            ResNetBlock(2 * ch, 2 * ch),
            AttentionBlock(2 * ch),
            DownSampling(2 * ch),
            ResNetBlock(2 * ch, 2 * ch),
            ResNetBlock(2 * ch, 2 * ch),
            DownSampling(2 * ch),
            ResNetBlock(2 * ch, 2 * ch),
            ResNetBlock(2 * ch, 2 * ch),
        ])

        self.middle = nn.ModuleList([
            ResNetBlock(2 * ch, 2 * ch),
            AttentionBlock(2 * ch),
            ResNetBlock(2 * ch, 2 * ch),
        ])

        self.up = nn.ModuleList([
            ResNetBlock(4 * ch, 2 * ch),
            ResNetBlock(4 * ch, 2 * ch),
            ResNetBlock(4 * ch, 2 * ch),
            UpSampling(2 * ch),
            ResNetBlock(4 * ch, 2 * ch),
            ResNetBlock(4 * ch, 2 * ch),
            ResNetBlock(4 * ch, 2 * ch),
            UpSampling(2 * ch),
            ResNetBlock(4 * ch, 2 * ch),
            AttentionBlock(2 * ch),
            ResNetBlock(4 * ch, 2 * ch),
            AttentionBlock(2 * ch),
            ResNetBlock(3 * ch, 2 * ch),
            AttentionBlock(2 * ch),
            UpSampling(2 * ch),
            ResNetBlock(3 * ch, ch),
            ResNetBlock(2 * ch, ch),
            ResNetBlock(2 * ch, ch),
          ])

        self.final_conv = nn.Conv2d(ch, in_ch, kernel_size=3, stride=1, padding=1 )

    def forward(self, x, t):
        '''
        param x (torch.Tensor): batch of images (B,C,H,W)
        param t (torch.Tensor): tensor of time steps (torch.long) [B]
        '''

        #timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = torch.nn.functional.silu(self.linear1(temb))
        temb = self.linear2(temb)
        assert temb.shape == (t.shape[0], self.ch*4)

        #DownSampling
        x1 = self.conv1(x) #ch
        x2 = self.down[0](x1,temb) #ch
        x3 = self.down[1](x2,temb) #ch
        x4 = self.down[2](x3) #DownSampling, ch
        x5 = self.down[3](x4,temb) #2 * ch
        x6 = self.down[4](x5) #Attention , 2 * ch
        x7 = self.down[5](x6,temb) #2 * ch
        x8 = self.down[6](x7) #Attention, 2 * ch
        x9 = self.down[7](x8) #DownSampling, 2 * ch
        x10 = self.down[8](x9, temb) #2 * ch
        x11 = self.down[9](x10,temb) #2 * ch
        x12 = self.down[10](x11) #DownSampling, 2 * ch
        x13 = self.down[11](x12,temb) #2 * ch
        x14 = self.down[12](x13,temb) #2 * ch

        #Middle - Bottleneck
        x = self.middle[0](x14, temb)
        x = self.middle[1](x)
        x = self.middle[2](x,temb)

        #UpSampling
        x = self.up[0](torch.concat((x, x14), dim = 1), temb)
        x = self.up[1](torch.concat((x,x13), dim = 1), temb)
        x = self.up[2](torch.concat((x, x12), dim = 1), temb)
        x = self.up[3](x)
        x = self.up[4](torch.concat((x, x11), dim = 1), temb)
        x = self.up[5](torch.concat((x, x10), dim = 1), temb)
        x = self.up[6](torch.concat((x, x9), dim = 1), temb)
        x = self.up[7](x)
        x = self.up[8](torch.concat((x, x8), dim = 1), temb)
        x = self.up[9](x)
        x = self.up[10](torch.concat((x, x6), dim = 1), temb)
        x = self.up[11](x)
        x = self.up[12](torch.concat((x, x4), dim = 1), temb)
        x = self.up[13](x)
        x = self.up[14](x)
        x = self.up[15](torch.concat((x, x3), dim = 1), temb)
        x = self.up[16](torch.concat((x,x2), dim = 1), temb)
        x = self.up[17](torch.concat((x, x1), dim = 1), temb)

        x = torch.nn.functional.silu(nn.functional.group_norm(x, num_groups=32))
        x = self.final_conv(x)
        return x
