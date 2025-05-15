# adapted from https://github.com/jzhangbs/DTUeval-python

import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
from tqdm import tqdm
from scipy.io import loadmat
import multiprocessing as mp
import trimesh
from plyfile import PlyData, PlyElement

def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[:n1 + 1, :n2 + 1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1, 2, 0))
    k = c[c.sum(axis=-1) < 1]  # m2
    q = v1 * k[:, :1] + v2 * k[:, 1:] + tri_vert
    return q

def eval(estimated_mesh_path, gt_pcd_path):
    in_file = estimated_mesh_path
    
    data_mesh = o3d.io.read_triangle_mesh(str(in_file))

    data_mesh.remove_unreferenced_vertices()

    mp.freeze_support()

    # default dtu values
    max_dist = 20
    patch = 60
    thresh = 0.2  # downsample density

    pbar = tqdm(total=9)
    pbar.set_description('read data mesh')

    vertices = np.asarray(data_mesh.vertices)
    triangles = np.asarray(data_mesh.triangles)
    tri_vert = vertices[triangles]

    pbar.update(1)
    pbar.set_description('sample pcd from mesh')
    v1 = tri_vert[:, 1] - tri_vert[:, 0]
    v2 = tri_vert[:, 2] - tri_vert[:, 0]
    l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
    non_zero_area = (area2 > 0)[:, 0]
    l1, l2, area2, v1, v2, tri_vert = [
        arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert]
    ]
    thr = thresh * np.sqrt(l1 * l2 / area2)
    n1 = np.floor(l1 / thr)
    n2 = np.floor(l2 / thr)

    with mp.Pool() as mp_pool:
        new_pts = mp_pool.map(sample_single_tri,
                              ((n1[i, 0], n2[i, 0], v1[i:i + 1], v2[i:i + 1], tri_vert[i:i + 1, 0]) for i in
                               range(len(n1))), chunksize=1024)

    new_pts = np.concatenate(new_pts, axis=0)
    data_pcd = np.concatenate([vertices, new_pts], axis=0)
    
    pbar.update(1)
    pbar.set_description('random shuffle pcd index')
    shuffle_rng = np.random.default_rng()
    shuffle_rng.shuffle(data_pcd, axis=0)

    pbar.update(1)
    pbar.set_description('downsample pcd')
    
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1)
    nn_engine.fit(data_pcd)
    data_down = data_pcd
    trimesh.PointCloud(data_down).export("tmp.ply", "ply")


    data_in = data_down
    data_in_obs = data_down

    pbar.update(1)
    pbar.set_description('read STL pcd')
    point_cloud = o3d.io.read_point_cloud(gt_pcd_path)
    stl = np.asarray(point_cloud.points)
    
    color_data_pcd = np.full((data_pcd.shape[0], 3), [255, 0, 0], dtype=np.uint8)  # Red color for A
    color_stl = np.full((stl.shape[0], 3), [0, 0, 255], dtype=np.uint8)  # Blue color for B
    points = np.vstack((data_pcd, stl))
    colors = np.vstack((color_data_pcd, color_stl))
 
    pbar.update(1)
    pbar.set_description('compute data2stl')
    nn_engine.fit(stl)
    dist_d2s, idx_d2s = nn_engine.kneighbors(data_in_obs, n_neighbors=1, return_distance=True)

    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()

    pbar.update(1)
    pbar.set_description('compute stl2data')

    stl_above = stl
    nn_engine.fit(data_in)
    dist_s2d, idx_s2d = nn_engine.kneighbors(stl_above, n_neighbors=1, return_distance=True)
    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

    
    
    pbar.update(1)
    pbar.close()
    over_all = (mean_d2s + mean_s2d) / 2

    print("mean_d2s, mean_s2d, over_all", mean_d2s, mean_s2d, over_all)