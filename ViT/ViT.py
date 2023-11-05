import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t,tuple) else (t,t)

class FeedForward(nn.Module):
    def __init__(self,dim,hidden_dim,dropout=0.):
        super().__init__()
        self.net=nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,dim),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.net(x)
class Attention(nn.Module):
    def __init__(self,dim,heads=8,dim_head=64,dropout=0.):
        super().__init__()
        inner_dim=heads*dim_head
        project_out=not (heads==1 and dim_head==dim)
        self.heads=heads
        self.scale=dim_head** -0.5
        self.norm=nn.LayerNorm(dim)
        self.attend=nn.Softmax(dim=-1)
        self.dropout=nn.Dropout(dropout)
        self.to_qkv=nn.Linear(dim,inner_dim*3,bias=False)
        self.to_out=nn.Sequential(
            nn.Linear(inner_dim,dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    def forward(self,x):
        x=self.norm(x)
        #qkv=([4,3,512],[4,3,512],[4,3,512])
        qkv=self.to_qkv(x).chunk(3,dim=-1)
        #q:[4,8,3,64],k:[4,8,3,64],v:[4,8,3,64]
        q,k,v=map(lambda t:rearrange(t,'b n (h d) -> b h n d', h=self.heads),qkv)
        #dots:[4,8,3,3]
        dots=torch.matmul(q,k.transpose(-1,-2)) * self.scale
        attn=self.attend(dots)
        attn=self.dropout(attn)
        #out:[4,8,3,64]
        out=torch.matmul(attn,v)
        #out:[4,3,512]
        out=rearrange(out,'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self,dim,depth,heads,dim_head,mlp_dim,dropout=0.):
        super().__init__()
        self.norm=nn.LayerNorm(dim)
        self.layers=nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim,heads=heads,dim_head=dim_head,dropout=dropout),
                FeedForward(dim,mlp_dim,dropout=dropout)
            ]))
    def forward(self,x):
        for attn,ff in self.layers:
            x=attn(x)+x #[4,3,80]
            x=ff(x)+x#[4,3,80]

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self,*,image_size,patch_size,num_classes,dim,depth,heads,mlp_dim,pool='cls',channels=3,dim_head=64,dropout=0.,emb_dropout=0.):
        super().__init__()
        image_height, image_width=pair(image_size)
        patch_height,patch_width=pair(patch_size)

        assert image_height % patch_height==0 and image_width % patch_width==0, 'Image dimensions must be divisible by the patch size.'

        num_patches=(image_height//patch_height)*(image_width//patch_width)
        patch_dim=channels * patch_height *patch_width
        assert pool in {'cls','mean'},'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding=nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height,p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim,dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding=nn.Parameter(torch.randn(1,num_patches+1,dim))
        self.cls_token=nn.Parameter(torch.randn(1,1,dim))
        self.dropout=nn.Dropout(emb_dropout)

        self.transformer=Transformer(dim,depth,heads,dim_head,mlp_dim,dropout)
        self.pool=pool
        self.to_latent=nn.Identity()
        self.mlp_head=nn.Linear(dim,num_classes)
    def forward(self,img):
        #img:[4,3,224,224]
        x=self.to_patch_embedding(img)#[4,3136,512]
        b,n,_=x.shape#b:4,n:3136
        cls_tokens=repeat(self.cls_token,'1 1 d -> b 1 d',b=b)#[4,1,512]
        x=torch.cat((cls_tokens,x),dim=1)#[4,3137,512]
        x+=self.pos_embedding[:,:(n+1)]#[4,3137,512]
        x=self.dropout(x)#[4,3137,512]
        x=self.transformer(x)#[4,3137,512]
        x=x.mean(dim=1) if self.pool=='mean' else x[:,0]
        x=self.to_latent(x)
        return self.mlp_head(x)

if __name__ == '__main__':
    X=torch.randn(4,3,224,224)
    ViT=ViT(image_size=224,patch_size=4,num_classes=10,
        dim=512,depth=4,heads=8,
        mlp_dim=256)
    Transformer=Transformer(80,4,8,64,40,0.5)
    ViT(X)
