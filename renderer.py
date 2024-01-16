import os
import numpy as np
import matplotlib.pyplot as plt
import ffmpeg
import cv2
from utils import *

import torch
from pytorch3d.renderer import (
    look_at_view_transform, PerspectiveCameras,
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    HardPhongShader, PointLights
)
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
import ffmpeg
import numpy as np
import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, 
    PerspectiveCameras, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer, 
    HardPhongShader,
    TexturesVertex
)
from pytorch3d.ops import interpolate_face_attributes
import glob
import argparse
from PIL import Image

class Renderer:
    def __init__(self, img_size, R, T, K):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.camera = PerspectiveCameras(
            device=self.device,
            R=R.transpose(-1, -2),               # Rotation matrix
            T=T,               # Translation vector
            K=K,               # Intrinsic matrix (4x4)
            image_size=([img_size]),  # Image size (height, width)
            in_ndc=False,
        )

        self.raster_settings = RasterizationSettings(
            image_size=img_size,
            blur_radius=0.0, 
            faces_per_pixel=1, 
            bin_size=None,
        )

        self.rasterizer = MeshRasterizer(
            cameras=self.camera, 
            raster_settings=self.raster_settings
        )

    def get_cameras(self):
        return self.cameras

    def get_fragments(self, mesh, znear, zfar):
        fragments = self.rasterizer(mesh, znear=znear, zfar=zfar)
        return fragments
    
    def get_data(self, mesh, fragments):
        verts_normal = mesh.verts_normals_packed()
        faces_normal = verts_normal[mesh.faces_packed()]
        pixel_normal = -interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, faces_normal)
        pixel_depth = fragments.zbuf
        pixel_mask = (fragments.zbuf > 0).float()

        return pixel_normal, pixel_depth, pixel_mask

    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Renderer')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--gpu_id', default='3', type=str)
    parser.add_argument('--root_dir', default='data', type=str)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '4'

    vertex_list = glob.glob(os.path.join(args.root_dir, '*.obj'))

    V, F, _ = load_obj(vertex_list[0])
    V = V.to(args.device)
    F = F.verts_idx.to(args.device)

    # V = normalize_vertex(V)

    # rot_mat = build_rotation_matrix(x_angle_max=30, y_angle_max=180, x_interv=30, y_interv=30)
    # rot_mat = rot_mat.to(args.device)
    # n_rot = len(rot_mat)

    # V_mv = torch.matmul(V[None], rot_mat)
    # F_mv = F[None].repeat(n_rot, 1, 1)

    # render = Renderer(img_size=256, dist=15)
    # D_mv, N_mv, M_mv = render.get_data(V_mv, F_mv)

    # out = (N_mv[0].detach().cpu().numpy()+1) * 127.5
    # im = Image.fromarray(out.astype(np.uint8))
    # im.save("test.png")