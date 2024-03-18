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

def save_points(filename, vertices):
    """
    Save a 3D model as an OBJ file.

    Parameters:
    - filename: Path to the output OBJ file.
    - vertices: List of vertices, each defined as a tuple (x, y, z).
    - faces: List of faces, each defined as a list of indices into the vertices list.
    - texture_coords: (Optional) List of texture coordinates, each defined as a tuple (u, v).
    - vertex_normals: (Optional) List of vertex normals, each defined as a tuple (nx, ny, nz).
    """
    with open(filename, 'w') as file:
        # Write vertices
        for v in vertices:
            file.write(f"v {v[0]} {v[1]} {v[2]}\n")


def read_imageset(filedir, filenames, resize_ratio=0.25, read_rgb=True):
    pix_normal = []
    for i in range(len(filenames)):
        data = cv2.imread('%s/%02d_000005.png' % (filedir, i))
        if read_rgb:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        H, W = int(data.shape[0] * resize_ratio), int(data.shape[1] * resize_ratio)
        data = cv2.resize(data, [H, W])
        pix_normal.append(data.reshape(H, W, 3))

    return np.array(pix_normal)

def read_calib(filename, resize_ratio=1.0):
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

def save_uv_info(args):
    # Read mesh data
    vp, fv, fvt, vt = read_obj(args.filename)
    vt[:, 1] = 1- vt[:, 1]

    mesh = Meshes(verts=torch.tensor(vp[None], device=args.device).float(), 
                  faces=torch.tensor(fv[None], device=args.device))
    vn = mesh.verts_normals_packed()
    vn = vn.detach().cpu().numpy()

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

    uv_f = np.zeros([args.uv_size, args.uv_size, 1])
    uv_v0 = np.zeros([args.uv_size, args.uv_size, 3])
    uv_v1 = np.zeros([args.uv_size, args.uv_size, 3])
    uv_v2 = np.zeros([args.uv_size, args.uv_size, 3])
    uv_b = np.zeros([args.uv_size, args.uv_size, 3])
    uv_vn0 = np.zeros([args.uv_size, args.uv_size, 3])
    uv_vn1 = np.zeros([args.uv_size, args.uv_size, 3])
    uv_vn2 = np.zeros([args.uv_size, args.uv_size, 3])

    area_triangle = np.cross(uv_t1 - uv_t0, uv_t2 - uv_t0) + 1e-8

    for uv_idx, xy_pixel in tqdm(enumerate(uv_grid)):

        t0p = xy_pixel - uv_t0
        t1p = xy_pixel - uv_t1
        t2p = xy_pixel - uv_t2

        b0 = np.cross(t1p, t2p) / area_triangle
        b1 = np.cross(t2p, t0p) / area_triangle
        b2 = np.cross(t0p, t1p) / area_triangle

        barycentric = np.stack([b0, b1, b2], axis=-1)
        is_inside = np.all(barycentric > 0, axis=1)
        
        f_inside = np.where(is_inside == True)[0]
        if len(f_inside) > 1:
            f_inside = f_inside[0]
        if f_inside:
            uv_f[int(xy_pixel[1]), int(xy_pixel[0])] = f_inside # face index per uv pixel
            uv_v0[int(xy_pixel[1]), int(xy_pixel[0])] = vp[fv[f_inside, 0]] # vertex position per uv pixel
            uv_v1[int(xy_pixel[1]), int(xy_pixel[0])] = vp[fv[f_inside, 1]] # vertex position per uv pixel
            uv_v2[int(xy_pixel[1]), int(xy_pixel[0])] = vp[fv[f_inside, 2]] # vertex position per uv pixel
            uv_b[int(xy_pixel[1]), int(xy_pixel[0])] = barycentric[f_inside] # barycentric per uv pixel
            uv_vn0[int(xy_pixel[1]), int(xy_pixel[0])] = vn[fv[f_inside, 0]] # vertex normal per uv pixel
            uv_vn1[int(xy_pixel[1]), int(xy_pixel[0])] = vn[fv[f_inside, 1]] # vertex normal per uv pixel
            uv_vn2[int(xy_pixel[1]), int(xy_pixel[0])] = vn[fv[f_inside, 2]] # vertex normal per uv pixel
    
    uv_v = np.stack([uv_v0, uv_v1, uv_v2], axis=0)
    uv_vn = np.stack([uv_vn0, uv_vn1, uv_vn2], axis=0)
    np.savez('uv_info_%d.npz' % args.uv_size, uv_f=uv_f, uv_v=uv_v, uv_b=uv_b, uv_vn=uv_vn)

def compute_uv_texture(args):
    # Read calib and normal data
    R, T, K, names = read_calib(args.cablidir, args.ratio)
    imgs = read_imageset(args.normaldir, names, args.ratio, read_rgb=False)
    imgs = torch.tensor(np.array(imgs), device=args.device).float()
    img_size = [imgs.shape[1], imgs.shape[2]]

    # Sampling data using interval
    imgs = imgs[::args.interval]
    R = torch.tensor(R).float()
    T = torch.tensor(T).float()
    K = torch.tensor(K).float()

    vp, fv, fvt, vt = read_obj(args.filename)
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
    npz = np.load('uv_info_%d.npz' % args.uv_size)
    uv_v = npz['uv_v']
    uv_f = npz['uv_f']
    uv_b = npz['uv_b']
    uv_vn = npz['uv_vn']

    uv_b = uv_b.reshape(-1, 3)
    uv_v0, uv_v1, uv_v2 = np.split(uv_v, 3, axis=0)
    uv_v0 = uv_v0.reshape(-1, 3)
    uv_v1 = uv_v1.reshape(-1, 3)
    uv_v2 = uv_v2.reshape(-1, 3)
    uv_vn0, uv_vn1, uv_vn2 = np.split(uv_vn, 3, axis=0)
    uv_vn0 = uv_vn0.reshape(-1, 3)
    uv_vn1 = uv_vn1.reshape(-1, 3)
    uv_vn2 = uv_vn2.reshape(-1, 3)

    uv_v_bary = uv_v0 * uv_b[:, [0]] + uv_v1 * uv_b[:, [1]] + uv_v2 * uv_b[:, [2]]
    uv_v_bary = torch.tensor(uv_v_bary, device=args.device).float() # uv_size x 3

    uv_vn_bary = uv_vn0 * uv_b[:, [0]] + uv_vn1 * uv_b[:, [1]] + uv_vn2 * uv_b[:, [2]]
    uv_vn_bary = torch.tensor(uv_vn_bary, device=args.device).float() # uv_size x 3
    uv_vn_bary = torch.matmul(uv_vn_bary, R.to(args.device))
    v_valid = uv_vn_bary[:, :, -1] < -0.65

    uv_v_color = torch.zeros_like(uv_v_bary)[None].repeat(len(imgs), 1, 1)
    uv_v_depth = torch.zeros_like(uv_v_bary)[None].repeat(len(imgs), 1, 1)

    for i in range(len(imgs)):
        uv_v_bary_xy = cameras.transform_points_screen(uv_v_bary)[:, :, :2] # n_view x uv_size x 3
        uv_v_bary_z = cameras.get_world_to_view_transform().transform_points(uv_v_bary)[:, :, [-1]]

        x = uv_v_bary_xy[i, :, 0]
        y = uv_v_bary_xy[i, :, 1]

        x1 = uv_v_bary_xy[i, :, 0].floor().long()
        x2 = uv_v_bary_xy[i, :, 0].ceil().long()
        y1 = uv_v_bary_xy[i, :, 1].floor().long()
        y2 = uv_v_bary_xy[i, :, 1].ceil().long()

        x1y1 = imgs[i, y1, x1]# * v_valid[i].reshape(-1, 1)
        x1y2 = imgs[i, y2, x1]# * v_valid[i].reshape(-1, 1)
        x2y1 = imgs[i, y1, x2]# * v_valid[i].reshape(-1, 1)
        x2y2 = imgs[i, y2, x2]# * v_valid[i].reshape(-1, 1)

        w11 = ((x2 - x) * (y2 - y) / ((x2 - x1) * (y2 - y1))).reshape(-1, 1)
        w12 = ((x2 - x) * (y - y1) / ((x2 - x1) * (y2 - y1))).reshape(-1, 1)
        w21 = ((x - x1) * (y2 - y) / ((x2 - x1) * (y2 - y1))).reshape(-1, 1)
        w22 = ((x - x1) * (y - y1) / ((x2 - x1) * (y2 - y1))).reshape(-1, 1)

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
        # uv_v_depth_i = uv_v_depth[i].detach().cpu().numpy().reshape(args.uv_size, args.uv_size, 3)
        mask_uv_i = zbuf_mask.detach().cpu().numpy().reshape(args.uv_size, args.uv_size, 3)

        cv2.imwrite('results/image_%03d.png' % i, (uv_v_color_i * mask_uv_i))
        cv2.imwrite('results/mask_%03d.png' % i, mask_uv_i.astype(np.uint8) * 255)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='2', help='gpu id')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--ratio', type=float, default=0.25, help='resize ratio')
    parser.add_argument('--n_iter', type=int, default=20, help='number of iteration')
    parser.add_argument('--interval', type=int, default=1, help='view interval')
    parser.add_argument('--filename', type=str, default='../../DB/BYroad/240115/sending/output_face_4.obj', help='obj name')
    parser.add_argument('--normaldir', type=str, default='../../DB/BYroad/240115/sending/frame0005', help='normal image dir')
    parser.add_argument('--cablidir', type=str, default='../../DB/BYroad/240115/sending/cams', help='calib file dir')
    parser.add_argument('--uv_size', type=int, default=256, help='calib file dir')
    parser.add_argument('--z_threshold', type=float, default=0.5, help='threshold for z buffer')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    tt = time.time()
    # save_uv_info(args)
    # print('1')
    compute_uv_texture(args)
    print(time.time() - tt)

    # om.write_mesh('refine_hjw_81.obj', om.TriMesh(v, f))
