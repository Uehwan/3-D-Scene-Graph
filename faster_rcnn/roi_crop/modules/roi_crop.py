from torch.nn.modules.module import Module
from ..functions.roi_crop import RoICropFunction
import torch
from torch.autograd import Variable
import torch.nn.functional as F

def _affine_grid_gen(rois, input_size, grid_size):

    rois = rois.detach()
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = input_size[0]
    width = input_size[1]

    zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    theta = torch.cat([\
      (x2 - x1) / (width - 1),
      zero,
      (x1 + x2 - width + 1) / (width - 1),
      zero,
      (y2 - y1) / (height - 1),
      (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, grid_size, grid_size)))

    return grid

class _RoICrop(Module):
    def __init__(self, layout = 'BHWD'):
        super(_RoICrop, self).__init__()
    def forward(self, input1, input2):
        return RoICropFunction()(input1, input2)

class RoICrop(Module):
    def __init__(self, grid_size,max_pool=True, layout = 'BHWD'):
        super(RoICrop,self).__init__()
        self.grid_size=grid_size
        self.max_pool = max_pool
    def forward(self, base_feat, rois):
        grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
        grid_yx = torch.stack([grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
        pooled_feat = RoICropFunction()(base_feat, Variable(grid_yx).detach())
        if self.max_pool:
            pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        return pooled_feat