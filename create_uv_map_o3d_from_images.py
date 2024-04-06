import os
import argparse

from renderer import Renderer
import numpy as np
import cv2
from pytorch3d.structures import Meshes
import torch
from glob import glob
import time
from tqdm import tqdm
from numba import njit, prange
from concurrent.futures import ProcessPoolExecutor
from numba.np.extensions import cross2d
import trimesh
import xatlas
import pyfqmr
import fast_simplification
import open3d as o3d
from scipy.signal import convolve2d
from scipy import ndimage

def read_imageset(filedir, filenames, resize_ratio=0.25, read_rgb=True):
    pix_normal = []
    for i in range(len(filenames)):
        # data = cv2.imread('%s/M_Normal_CAM%02d.png' % (filedir, i+1))
        data = cv2.imread('%s/M_Normal_CAM%02d.png' % (filedir, i))
        if read_rgb:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        H, W = int(data.shape[0] * resize_ratio), int(data.shape[1] * resize_ratio)
        data = cv2.resize(data, [H, W])
        pix_normal.append(data.reshape(H, W, 3))

    return np.array(pix_normal)

def read_calib(filename, resize_ratio=0.25):
    calib_list = sorted(glob('%s/*.txt' % filename))
    R, T, K, names = [], [], [], []
    for calib_ in calib_list:
        with open(calib_, 'r') as f:
            lines = f.readlines()
        extrinsic = np.array([i.rstrip().split(' ') for i in lines[1:5]]).astype(np.float32)
        intrinsic = np.array([i.rstrip().split(' ') for i in lines[7:10]]).astype(np.float32)

        rotation = extrinsic[:3, :3]
        translation = extrinsic[:3, 3]

        intrinsic = intrinsic * resize_ratio
        intrinsic = np.concatenate([intrinsic, np.zeros([3, 1])], axis=1)
        intrinsic = np.concatenate([intrinsic, np.zeros([1, 4])], axis=0)
        intrinsic[2, 2] = 0.0
        intrinsic[2, 3] = 1.0
        intrinsic[3, 2] = 1.0

        R.append(rotation)
        T.append(translation)
        K.append(intrinsic)
        names.append(os.path.basename(calib_))

    R, T, K = np.array(R), np.array(T), np.array(K)
    K[:, 0, 0] = -K[:, 0, 0]
    K[:, 1, 1] = -K[:, 1, 1]

    return R, T, K, np.array(names)

def read_obj(obj_path, vt_color=True):
    vs = []
    fvs = []
    fvts = []
    vts = []
    with open(obj_path, 'r') as obj_f:
        lines = obj_f.readlines()
        for line in lines:
            line_ = line.split(' ')
            if len(line_) < 3:
                line_ = line.split('\t')
            if line_[0] == 'v':
                v1 = float(line_[1])
                v2 = float(line_[2])
                v3 = float(line_[3])
                vs.append([v1, v2, v3])
            if vt_color:
                if line_[0] == 'vt':
                    vt1 = float(line_[1])
                    vt2 = float(line_[2])
                    vts.append([vt1, vt2])
            if line_[0] == 'f':
                f1 = int(line_[1].split('/')[0]) - 1
                f2 = int(line_[2].split('/')[0]) - 1
                f3 = int(line_[3].split('/')[0]) - 1
                fvs.append([f1, f2, f3])
                if vt_color:
                    fvt1 = int(line_[1].split('/')[1]) - 1
                    fvt2 = int(line_[2].split('/')[1]) - 1
                    fvt3 = int(line_[3].split('/')[1]) - 1
                    fvts.append([fvt1, fvt2, fvt3])
    vs = np.array(vs)
    fvs = np.array(fvs)
    fvts = np.array(fvts)
    vts = np.array(vts)

    if vt_color:
        return vs, fvs, fvts, vts
    else:
        return vs, fvs

@njit(parallel=True)
def compute_uv_attributes(loop_idx, uv_grid, uv_t0, uv_t1, uv_t2, vp, fv, vn, uv_size, b_threshold):
    uv_f   = np.zeros(uv_size * uv_size)
    uv_b0  = np.zeros(uv_size * uv_size)
    uv_b1  = np.zeros(uv_size * uv_size)
    uv_b2  = np.zeros(uv_size * uv_size)
    uv_fv0 = np.zeros(uv_size * uv_size)
    uv_fv1 = np.zeros(uv_size * uv_size)
    uv_fv2 = np.zeros(uv_size * uv_size)

    area_triangle = cross2d(uv_t1 - uv_t0, uv_t2 - uv_t0) + 1e-8

    for uv_idx in prange(len(loop_idx)):
        xy_pixel = uv_grid[uv_idx]

        t0p = xy_pixel - uv_t0
        t1p = xy_pixel - uv_t1
        t2p = xy_pixel - uv_t2

        b0 = cross2d(t1p, t2p) / area_triangle
        b1 = cross2d(t2p, t0p) / area_triangle
        b2 = cross2d(t0p, t1p) / area_triangle

        is_inside0 = b0 > b_threshold
        is_inside1 = b1 > b_threshold
        is_inside2 = b2 > b_threshold

        is_inside = is_inside0 * is_inside1 * is_inside2
        
        f_inside = np.where(is_inside == True)[0]
        if len(f_inside) > 0:
            b0_from_center = (b0[f_inside] - 1/3)**2
            b1_from_center = (b1[f_inside] - 1/3)**2
            b2_from_center = (b2[f_inside] - 1/3)**2

            f_closest = np.argmin(np.sqrt(b0_from_center + b1_from_center + b2_from_center))
            f_inside = f_inside[f_closest]
            # f_inside = f_inside[0]

            uv_f[uv_idx] = f_inside # face index per uv pixel
            uv_fv0[uv_idx] = fv[f_inside, 0]
            uv_fv1[uv_idx] = fv[f_inside, 1]
            uv_fv2[uv_idx] = fv[f_inside, 2]
            uv_b0[uv_idx] = b0[f_inside] # barycentric per uv pixel
            uv_b1[uv_idx] = b1[f_inside] # barycentric per uv pixel
            uv_b2[uv_idx] = b2[f_inside] # barycentric per uv pixel

    return uv_f, uv_fv0, uv_fv1, uv_fv2, uv_b0, uv_b1, uv_b2

def create_uv_atlas(args, data_list):
    for i, data_path in enumerate(tqdm(data_list)):
        legacy = o3d.io.read_triangle_mesh(data_path)

        if False:
            legacy.compute_vertex_normals()
            pcd = o3d.geometry.PointCloud(legacy.vertices)
            pcd.normals = legacy.vertex_normals
            legacy, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=7)

        if i == 0:
            mesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy)
            # mesh.cuda()
            mesh.compute_uvatlas(gutter=20)
            uv_legacy = o3d.t.geometry.TriangleMesh.to_legacy(mesh)
        uv_legacy.vertices = legacy.vertices

        os.makedirs(args.save_datadir, exist_ok=True)
        save_name = os.path.join(args.save_datadir, os.path.basename(data_path))
        o3d.io.write_triangle_mesh(save_name, uv_legacy, write_vertex_colors=False)

def save_uv_info(args, data_list):
    for i, data_path in enumerate(tqdm(data_list)):
        uv_path = data_path.replace('.obj', '_uv.npz')
        uv_vn_path = data_path.replace('.obj', '_vn.png')
        mtl_path = data_path.replace('.obj', '.mtl')

        with open(mtl_path, 'r') as file:
            lines = file.readlines()

        lines.append('map_Kd %s' % os.path.basename(uv_vn_path))

        os.remove(mtl_path)
        with open(mtl_path, 'w') as file:
            file.writelines(lines)
            
        # Read mesh data
        vp, fv, fvt, vt = read_obj(data_path)
        vt[:, 1] = 1- vt[:, 1]

        mesh = Meshes(verts=torch.tensor(vp[None], device=args.device).float(), 
                    faces=torch.tensor(fv[None], device=args.device))
        vn = mesh.verts_normals_packed()
        vn = vn.detach().cpu().numpy()

        if i == 0:
            uv_t = vt[fvt] # uv coordinate per face
            uv_t[:, :, 0] *= (args.uv_size -1)
            uv_t[:, :, 1] *= (args.uv_size -1)

            u = np.arange(0, args.uv_size)
            v = np.arange(0, args.uv_size)
            u_grid, v_grid = np.meshgrid(u, v)
            uv_grid = np.stack([u_grid, v_grid], axis=-1).reshape(-1, 2)

            uv_t0 = uv_t[:, 0] # face -> vertex -> uv
            uv_t1 = uv_t[:, 1]
            uv_t2 = uv_t[:, 2]

            loop_idx = np.arange(len(uv_grid))
            uv_f, uv_fv0, uv_fv1, uv_fv2, uv_b0, uv_b1, uv_b2 = compute_uv_attributes(loop_idx, uv_grid, uv_t0, uv_t1, uv_t2, vp, fv, vn, args.uv_size, args.b_threshold)

            uv_b0 = uv_b0[..., None] # (uv_size x uv_size) x 1
            uv_b1 = uv_b1[..., None] # (uv_size x uv_size) x 1
            uv_b2 = uv_b2[..., None] # (uv_size x uv_size) x 1

        uv_vp0 = vp[uv_fv0.astype(np.int64)]
        uv_vp1 = vp[uv_fv1.astype(np.int64)]
        uv_vp2 = vp[uv_fv2.astype(np.int64)]

        uv_vn0 = vn[uv_fv0.astype(np.int64)]
        uv_vn1 = vn[uv_fv1.astype(np.int64)]
        uv_vn2 = vn[uv_fv2.astype(np.int64)]

        uv_vp_bary = uv_vp0 * uv_b0 + uv_vp1 * uv_b1 + uv_vp2 * uv_b2
        uv_vp_bary = torch.tensor(uv_vp_bary, device=args.device).float() # uv_size x 3
        uv_vp_bary = uv_vp_bary.reshape(args.uv_size, args.uv_size, 3).detach().cpu().numpy()

        uv_vn_bary = uv_vn0 * uv_b0 + uv_vn1 * uv_b1 + uv_vn2 * uv_b2
        uv_vn_bary = torch.tensor(uv_vn_bary, device=args.device).float() # uv_size x 3
        uv_vn_bary = uv_vn_bary.reshape(args.uv_size, args.uv_size, 3).detach().cpu().numpy()

        np.savez(uv_path, uv_vn_bary=uv_vn_bary, uv_vp_bary=uv_vp_bary)
        cv2.imwrite(uv_vn_path, (uv_vn_bary.reshape(args.uv_size, args.uv_size, 3)+1)*127.5)

def compute_uv_texture(args, data_list):
    for i, data_path in enumerate(tqdm(data_list)):
        uv_path = data_path.replace('.obj', '_uv.npz')
        uv_vn_path = data_path.replace('.obj', '_vn.png')
        mtl_path = data_path.replace('.obj', '.mtl')

        # Read calib and normal data
        R, T, K, names = read_calib(args.cablidir, args.ratio)
        imgs = read_imageset(args.normaldir, names, args.ratio, read_rgb=True)
        imgs = torch.tensor(np.array(imgs), device=args.device).float()
        imgs = imgs / 127.5 - 1
        img_size = [imgs.shape[1], imgs.shape[2]]

        # Sampling data using interval
        imgs = imgs[::args.interval]
        R = torch.tensor(R).float()
        T = torch.tensor(T).float()
        K = torch.tensor(K).float()

        vp, fv, fvt, vt = read_obj(data_path)
        vp = torch.tensor(vp, device=args.device).float()
        fv = torch.tensor(fv, device=args.device)

        # Create renderer for fragments
        renderer = Renderer(img_size, R, T, K)
        cameras = renderer.camera

        n_view = len(R)
        mesh = Meshes(verts=vp[None].repeat(n_view, 1, 1), faces=fv[None].repeat(n_view, 1, 1))
        znear = vp[:, 2].min().item()
        zfar = vp[:, 2].max().item()
        fragments = renderer.get_fragments(mesh, znear=znear, zfar=zfar)
        pix_to_face = fragments.pix_to_face
        depths = fragments.zbuf

        # load uv info 
        npz = np.load(uv_path)
        uv_vp_bary = npz['uv_vp_bary'].reshape(-1, 3)
        uv_vn_bary = npz['uv_vn_bary'].reshape(-1, 3)

        uv_vp_bary = torch.tensor(uv_vp_bary, device=args.device).float() # uv_size x 3

        uv_vn_bary = torch.tensor(uv_vn_bary, device=args.device).float() # uv_size x 3
        uv_vn_bary = torch.matmul(uv_vn_bary, R.to(args.device))
        v_valid = uv_vn_bary[:, :, -1] > 0.0

        uv_v_color = torch.zeros_like(uv_vp_bary)[None].repeat(len(imgs), 1, 1)
        uv_v_depth = torch.zeros_like(uv_vp_bary[:, [0]])[None].repeat(len(imgs), 1, 1)

        uv_v_bary_xy = cameras.transform_points_screen(uv_vp_bary)[:, :, :2] # n_view x uv_size x 3
        uv_v_bary_z = cameras.get_world_to_view_transform().transform_points(uv_vp_bary)[:, :, [-1]]

        uv_imgs = []
        for i in range(len(imgs)):
            x = uv_v_bary_xy[i, :, 0]
            y = uv_v_bary_xy[i, :, 1]

            x1 = uv_v_bary_xy[i, :, 0].floor().long().clip(0, img_size[1] -1)
            x2 = uv_v_bary_xy[i, :, 0].ceil().long().clip(0, img_size[1] -1)
            y1 = uv_v_bary_xy[i, :, 1].floor().long().clip(0, img_size[0] -1)
            y2 = uv_v_bary_xy[i, :, 1].ceil().long().clip(0, img_size[0] -1)

            x1y1 = imgs[i, y1, x1]# * v_valid[i].reshape(-1, 1)
            x1y2 = imgs[i, y2, x1]# * v_valid[i].reshape(-1, 1)
            x2y1 = imgs[i, y1, x2]# * v_valid[i].reshape(-1, 1)
            x2y2 = imgs[i, y2, x2]# * v_valid[i].reshape(-1, 1)

            w11 = ((x2 - x) * (y2 - y) / ((x2 - x1) * (y2 - y1) + 1e-8)).reshape(-1, 1)
            w12 = ((x2 - x) * (y - y1) / ((x2 - x1) * (y2 - y1) + 1e-8)).reshape(-1, 1)
            w21 = ((x - x1) * (y2 - y) / ((x2 - x1) * (y2 - y1) + 1e-8)).reshape(-1, 1)
            w22 = ((x - x1) * (y - y1) / ((x2 - x1) * (y2 - y1) + 1e-8)).reshape(-1, 1)

            uv_v_color[i] = x1y1 * w11 + x1y2 * w12 + x2y1 * w21 + x2y2 * w22
            uv_v_color_i = uv_v_color[i].detach().cpu().numpy().reshape(args.uv_size, args.uv_size, 3)

            x1y1 = depths[i, y1, x1]# * v_valid[i].reshape(-1, 1)
            x1y2 = depths[i, y2, x1]# * v_valid[i].reshape(-1, 1)
            x2y1 = depths[i, y1, x2]# * v_valid[i].reshape(-1, 1)
            x2y2 = depths[i, y2, x2]# * v_valid[i].reshape(-1, 1)

            w11 = ((x2 - x) * (y2 - y) / ((x2 - x1) * (y2 - y1) + 1e-8)).reshape(-1, 1)
            w12 = ((x2 - x) * (y - y1) / ((x2 - x1) * (y2 - y1) + 1e-8)).reshape(-1, 1)
            w21 = ((x - x1) * (y2 - y) / ((x2 - x1) * (y2 - y1) + 1e-8)).reshape(-1, 1)
            w22 = ((x - x1) * (y - y1) / ((x2 - x1) * (y2 - y1) + 1e-8)).reshape(-1, 1)

            uv_v_depth[i] = x1y1 * w11 + x1y2 * w12 + x2y1 * w21 + x2y2 * w22
            zbuf_mask = uv_v_depth[i] - uv_v_bary_z[i]
            zbuf_mask = zbuf_mask.abs() < args.z_threshold
            mask_uv_i = zbuf_mask.float().detach().cpu().numpy().reshape(args.uv_size, args.uv_size, 1)
            
            cv2.imwrite('results/image_%03d.png' % i, (uv_v_color_i * mask_uv_i + 1) * 127.5)
            cv2.imwrite('results/mask_%03d.png' % i, mask_uv_i.astype(np.uint8) * 255)

            uv_imgs.append(uv_v_color_i * mask_uv_i)
        uv_imgs = np.stack(uv_imgs, axis=0)
        uv_imgs = np.ma.masked_equal(uv_imgs, 0)
        uv_imgs = np.ma.median(uv_imgs, axis=0).filled(0)  # .filled(0) fills masked values with 0 in the output
        uv_imgs = uv_imgs.reshape(args.uv_size, args.uv_size, 3)

        kernel_size = 3
        kernel = np.ones([kernel_size, kernel_size]) / kernel_size**2
        
        uv_zeros = uv_imgs == 0
        for i in range(10):
            # -1 indicates that the depth of the output image is the same as the input
            uv_imgs_filtered = cv2.filter2D(uv_imgs, -1, kernel)
            uv_imgs[uv_zeros] = uv_imgs_filtered[uv_zeros]
        
        cv2.imwrite(uv_vn_path, (uv_imgs+1)*127.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='2', help='gpu id')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--uv_size', type=int, default=256, help='calib file dir')
    parser.add_argument('--b_threshold', type=float, default=-0.5, help='threshold for barycentric')
    # parser.add_argument('--datadir', type=str, default='../../DB/BYroad/240320_registration/raw', help='data dir')
    # parser.add_argument('--save_datadir', type=str, default='../../DB/BYroad/240320_registration/uv_map', help='save dir')
    # parser.add_argument('--datadir', type=str, default='../../DB/BYroad/240318_body_atlas', help='data dir')
    # parser.add_argument('--save_datadir', type=str, default='../../DB/BYroad/240318_body_atlas/uv_map', help='save dir')
    parser.add_argument('--datadir', type=str, default='../../DB/BYroad/240405_Light/20240403_LSM_light_static_1', help='data dir')
    parser.add_argument('--save_datadir', type=str, default='../../DB/BYroad/240405_Light/20240403_LSM_light_static_1/uv_map2', help='save dir')

    parser.add_argument('--ratio', type=float, default=0.5, help='resize ratio')
    parser.add_argument('--n_iter', type=int, default=20, help='number of iteration')
    parser.add_argument('--interval', type=int, default=1, help='view interval')
    parser.add_argument('--normaldir', type=str, default='../../DB/BYroad/240405_Light/20240403_LSM_light_static_1/undistort', help='normal image dir')
    parser.add_argument('--cablidir', type=str, default='../../DB/BYroad/240405_Light/20240403_LSM_light_static_1/cams', help='calib file dir')
    parser.add_argument('--z_threshold', type=float, default=50, help='threshold for z buffer')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    tt = time.time()

    data_list = sorted(glob(os.path.join(args.datadir, '*.obj')))

    # registered data --> uv atlas data
    print('Creating UV atlas...')
    create_uv_atlas(args, data_list)

    # # uv atlas data --> uv mapping
    print('Computing UV mapping...')
    data_list = sorted(glob(os.path.join(args.save_datadir, '*.obj')))
    save_uv_info(args, data_list)

    compute_uv_texture(args, data_list)
    # data_list = sorted(glob(os.path.join('../../DB/BYroad/240320_registration/uv_map_lq', '*.obj')))
    # save_uv_info(args, data_list)

    # # uv_info_name = 'uv_info_b%d_%d.npz' % (args.b_threshold, args.uv_size)
    # for data_path in tqdm(data_list):
    #     uv_path = data_path.replace('.obj', '_uv.npz')
    #     save_uv_info(args, data_path, uv_path)

    print(time.time() - tt)
