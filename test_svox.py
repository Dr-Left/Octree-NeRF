import matplotlib.pyplot as plt
import numpy as np
import svox
import torch

t0 = svox.N3Tree(data_dim=4, data_format="RGBA",
                 center=[0.5, 0.5, 0.5], radius=0.5,
                 N=2, device="cpu",
                 init_refine=0, depth_limit=10,
                 extra_data=None)

device = 'cpu'

data_array = np.load("./dataset/test/cameras_sphere.npz", allow_pickle=True)
print(data_array)
with open("./dataset/test/cameras_sphere.txt", 'w') as f:
    np.savetxt(f, data_array['camera_mat_0'], fmt='%8.2f')

t = svox.N3Tree.load("./dataset/test/cameras_sphere.npz", device=device)
r = svox.VolumeRenderer(t)

c2w = torch.tensor([[-0.9999999403953552, 0.0, 0.0, 0.0],
                    [0.0, -0.7341099977493286, 0.6790305972099304, 2.737260103225708],
                    [0.0, 0.6790306568145752, 0.7341098785400391, 2.959291696548462],
                    [0.0, 0.0, 0.0, 1.0],
                    ], device=device)

with torch.no_grad():
    im = r.render_persp(c2w, height=800, width=800, fx=1111.111).clamp_(0.0, 1.0)
plt.imshow(im.cpu())
plt.show()
