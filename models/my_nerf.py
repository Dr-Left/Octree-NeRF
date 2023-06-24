import logging
from typing import Optional

import numpy as np
import svox
import torch
from pydantic import BaseModel

RS = 256


class CheatNeRF():
    def __init__(self, nerf):
        super(CheatNeRF, self).__init__()
        self.nerf = nerf

    def query(self, pts_xyz):
        return self.nerf(pts_xyz, torch.zeros_like(pts_xyz))


class MyModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True


class MyNeRF(MyModel):
    volume_sigma: Optional[torch.Tensor] = None
    volume_color: Optional[torch.Tensor] = None

    def __init__(self):
        super(MyNeRF, self).__init__()

    def save(self, pts_xyz, sigma, color):
        self.volume_sigma = torch.zeros((RS, RS, RS))
        self.volume_color = torch.zeros((RS, RS, RS, 3))
        x_index = ((pts_xyz[:, 0] + 0.125) * 4 * RS).clamp(0, RS - 1).long()
        y_index = ((pts_xyz[:, 1] - 0.75) * 4 * RS).clamp(0, RS - 1).long()
        z_index = ((pts_xyz[:, 2] + 0.125) * 4 * RS).clamp(0, RS - 1).long()
        self.volume_sigma[x_index, y_index, z_index] = sigma[:, 0]
        self.volume_color[x_index, y_index, z_index] = color[:, :]
        # visualize
        array_3d = self.volume_sigma.detach().cpu().numpy()
        import rebuild_3D_graph
        # with open('volume_sigma.txt', 'w') as f:
        #     array_2d = array_3d.reshape(-1, array_3d.shape[-1])
        #     np.savetxt(f, array_2d, fmt='%f')
        rebuild_3D_graph.rebuild_3D(array_3d)  # convert to numpy ndarray

    def query(self, pts_xyz):
        N, _ = pts_xyz.shape
        sigma = torch.zeros(N, 1, device=pts_xyz.device)
        color = torch.zeros(N, 3, device=pts_xyz.device)
        x_index = ((pts_xyz[:, 0] + 0.125) * 4 * RS).clamp(0, RS - 1).long()
        y_index = ((pts_xyz[:, 1] - 0.75) * 4 * RS).clamp(0, RS - 1).long()
        z_index = ((pts_xyz[:, 2] + 0.125) * 4 * RS).clamp(0, RS - 1).long()
        sigma[:, 0] = self.volume_sigma[x_index, y_index, z_index]
        color[:, :] = self.volume_color[x_index, y_index, z_index]
        return sigma, color


class HighPerformanceNeRF(MyNeRF):
    """Use svox to accelerate the query process
    """
    t: Optional[svox.N3Tree]
    r: Optional[svox.VolumeRenderer]

    def __init__(self):
        super(HighPerformanceNeRF, self).__init__()

    def save(self, pts_xyz, sigma, color, init_refine=6, refine_depth=5):
        # self.debug_printing(pts_xyz, sigma, color)
        self.t = svox.N3Tree(data_dim=4,
                             data_format='RGBA',
                             init_refine=init_refine,
                             center=[0, 0.875, 0],
                             radius=0.125,
                             depth_limit=35,
                             )
        print(f'pts_xyz.shape: {pts_xyz.shape}')
        k = pts_xyz.shape[0] // 4  # TODO: later change to 128 * 128 * 64
        # find the top k points in sigma
        _, indices = torch.topk(sigma, k, dim=0)
        indices = indices.squeeze()
        logging.info(f"selected top-k:{pts_xyz[indices].shape}")
        self.t[pts_xyz[indices]].refine(refine_depth)
        print("self.t.shape:[N, 4]: {}".format(self.t.shape))
        self.t[pts_xyz] = torch.cat([sigma, color], dim=-1)
        # self.debug_printing(pts_xyz, sigma, color)
        # self.t[pts_xyz[indices]] = torch.cat([sigma[indices], color[indices]], dim=-1)
        # super(HighPerformanceNeRF, self).save(pts_xyz, sigma, color)

    def query(self, pts_xyz):
        return self.t[pts_xyz][:, :1], self.t[pts_xyz][:, 1:]

    def debug_printing(self, pts_xyz, sigma, color):
        # dump data
        with open('pts_xyz.txt', 'w') as f:
            array_3d = pts_xyz.detach().cpu().numpy()
            array_2d = array_3d.reshape(-1, array_3d.shape[-1])
            np.savetxt(f, array_2d, fmt='%6.2f')

        with open('sigma.txt', 'w') as f:
            array = sigma.detach().cpu().numpy()
            np.savetxt(f, array, fmt='%6.2f')

        with open('color.txt', 'w') as f:
            array = color.detach().cpu().numpy()
            np.savetxt(f, array, fmt='%6.2f')

        rang = torch.arange(143953, 143963, 1, dtype=torch.long)

        print(f'pts_xyz {pts_xyz[rang]}')
        print(f't[pts_xyz] {self.t[pts_xyz[rang]]}')
        print(f'sigma {sigma[rang]}')
        print(f'color {color[rang]}')
