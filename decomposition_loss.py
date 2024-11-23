import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **
                                                                      2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()

def decomp(feature_V_B, feature_V_D,feature_I_B, feature_I_D):
    cc_loss_B = cc(feature_V_B, feature_I_B)
    cc_loss_D = cc(feature_V_D, feature_I_D)
    loss_decomp =  (cc_loss_D) ** 2/ (1.01 + cc_loss_B) 
    return loss_decomp 

a = torch.rand(10, 3, 256, 256).cuda()
b = torch.rand(10, 3, 256, 256).cuda()

c = torch.rand(10, 3, 256, 256).cuda()
d = torch.rand(10, 3, 256, 256).cuda()
loss = decomp(a, b,c,d).cuda()
print(loss)