from functools import partial
import numpy as np
import torch
import torch.nn.functional as torch_F
import torch_scatter
from collections import defaultdict

from imaginaire.models.base import Model as BaseModel
from project.nerf.utils import nerf_util, camera, render
from project.splatsdf.utils import misc
from project.splatsdf.utils.modules import NeuralSDF, NeuralRGB, BackgroundNeRF

from project.gaussians.scene.gaussian_model import GaussianModel
from project.gaussians.scene.gaussian_dataset_readers import readNerfSyntheticInfo, readDTUInfo
from project.gaussians.utils.camera_utils import cameraList_from_camInfos
from project.gaussians.gaussian_renderer import render as gaussian_render
from project.gaussians.utils.loss_utils import ssim
from project.gaussians.utils.sh_utils import SH2RGB


from pytorch3d.transforms import quaternion_apply, quaternion_invert, matrix_to_quaternion, quaternion_to_matrix

from torch.utils.cpp_extension import load as load_cuda
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from plyfile import PlyData, PlyElement


parent_dir = os.path.dirname(os.path.abspath(__file__))
query_worldcoords_cuda = load_cuda(
    name='query_worldcoords_cuda',
    sources=[
        os.path.join(parent_dir, path)
        for path in ['cuda/query_worldcoords.cpp', 'cuda/query_worldcoords.cu']],
    verbose=True)

def construct_vox_points_closest(xyz_val, vox_res, partition_xyz=None, space_min=None, space_max=None):
    # xyz, N, 3
    xyz = xyz_val if partition_xyz is None else partition_xyz
    if space_min is None:
        xyz_min, xyz_max = torch.min(xyz, dim=-2)[0], torch.max(xyz, dim=-2)[0]
        space_edge = torch.max(xyz_max - xyz_min) * 1.05
        xyz_mid = (xyz_max + xyz_min) / 2
        space_min = xyz_mid - space_edge / 2
    else:
        space_edge = space_max - space_min
        mask = (xyz_val - space_min[None,...])
        mask *= (space_max[None,...] - xyz_val)
        mask = torch.prod(mask, dim=-1) > 0
        xyz_val = xyz_val[mask, :]
    construct_vox_sz = space_edge / vox_res
    xyz_shift = xyz - space_min[None, ...]
    sparse_grid_idx, inv_idx, point_counts = torch.unique(torch.floor(xyz_shift / construct_vox_sz[None, ...]).to(torch.int32), dim=0, return_inverse=True, return_counts=True)
    haha = torch.nonzero(torch.gt(point_counts,1))
    
    indices_list = []
    for b in haha:
        indices = torch.nonzero(torch.eq(inv_idx, b)).squeeze().tolist()
        if isinstance(indices, list):
            for i in indices:
                indices_list.append(i)
        else:
            indices_list.append(indices)
    indices_list = torch.tensor(indices_list).to('cuda')
    
    xyz_sub = xyz[indices_list, :]

    return indices_list

class Model(BaseModel):

    def __init__(self, cfg_model, cfg_data):
        super().__init__(cfg_model, cfg_data)
        self.cfg_render = cfg_model.render
        self.white_background = cfg_model.background.white
        self.with_background = cfg_model.background.enabled
        self.with_appear_embed = cfg_model.appear_embed.enabled
        self.anneal_end = cfg_model.object.s_var.anneal_end
        self.outside_val = 1000. * (-1 if cfg_model.object.sdf.mlp.inside_out else 1)
        self.image_size_train = cfg_data.train.image_size
        self.image_size_val = cfg_data.val.image_size
        self.rendered_depth_path = cfg_data.pt_rendered_depth_path
        self.dataset_name = cfg_data.dataset_name
        # Define models.
        self.build_model(cfg_model, cfg_data)
        # Define functions.
        self.ray_generator = partial(nerf_util.ray_generator,
                                     camera_ndc=False,
                                     num_rays=cfg_model.render.rand_rays)
        self.sample_dists_from_pdf = partial(nerf_util.sample_dists_from_pdf,
                                             intvs_fine=cfg_model.render.num_samples.fine)
        
        # init gaussians
        with torch.no_grad():
            self.gaussians = GaussianModel(3)
            
        if self.dataset_name == "nerf":
            gaussian_path = cfg_data.root
            self.scene_info_gauss = readNerfSyntheticInfo(gaussian_path, 'images', False)
        elif self.dataset_name == "dtu":
            gaussian_path = cfg_data.gaussian_path
            self.scene_info_gauss = readDTUInfo(gaussian_path, 'images', False)
        with torch.no_grad():
            gaussian_ply_path = cfg_data.gaussian_ply_path
            self.gaussians.load_ply(gaussian_ply_path)
            
        if self.dataset_name == "nerf":
            with torch.no_grad():
                self.preprocess_gaussian()
        
        if cfg_model.aggregator_ablation:
            from project.splatsdf.aggregator_ablation_pts import Aggregator
        else:
            from project.splatsdf.aggregator import Aggregator
        self.aggregator = Aggregator()
        self.viewpoint_stack = cameraList_from_camInfos(self.scene_info_gauss.train_cameras).copy()

    def preprocess_gaussian(self):
        # voxlization pruning guassian noise
        indices_list = construct_vox_points_closest(self.gaussians._xyz, 256)
        indices_mask = torch.zeros((self.gaussians._xyz.shape[0])).bool().to(self.gaussians._xyz.device)
        indices_mask[indices_list] = True
        indices_mask = ~indices_mask
        
        self.gaussians._xyz = self.gaussians._xyz[~indices_mask]
        self.gaussians._features_dc = self.gaussians._features_dc[~indices_mask]
        self.gaussians._features_rest = self.gaussians._features_rest[~indices_mask]
        self.gaussians._opacity = self.gaussians._opacity[~indices_mask]
        self.gaussians._scaling = self.gaussians._scaling[~indices_mask]
        self.gaussians._rotation = self.gaussians._rotation[~indices_mask]

    def build_model(self, cfg_model, cfg_data):
        # appearance encoding
        if cfg_model.appear_embed.enabled:
            assert cfg_data.num_images is not None
            self.appear_embed = torch.nn.Embedding(cfg_data.num_images, cfg_model.appear_embed.dim)
            if cfg_model.background.enabled:
                self.appear_embed_outside = torch.nn.Embedding(cfg_data.num_images, cfg_model.appear_embed.dim)
            else:
                self.appear_embed_outside = None
        else:
            self.appear_embed = self.appear_embed_outside = None
        self.neural_sdf = NeuralSDF(cfg_model.object.sdf)
        self.neural_rgb = NeuralRGB(cfg_model.object.rgb, feat_dim=cfg_model.object.sdf.mlp.hidden_dim,
                                    appear_embed=cfg_model.appear_embed)
        if cfg_model.background.enabled:
            self.background_nerf = BackgroundNeRF(cfg_model.background, appear_embed=cfg_model.appear_embed)
        else:
            self.background_nerf = None
        self.s_var = torch.nn.Parameter(torch.tensor(cfg_model.object.s_var.init_val, dtype=torch.float32))
        self.surface_fusion = cfg_model.surface_fusion
        self.gs_render_depth = cfg_model.gs_render_depth
        

    def forward(self, data):
        #gaussian forward
        viewpoint_cam = self.viewpoint_stack[data['idx']]
        self.viewpoint_cam = viewpoint_cam
        
        bg = torch.zeros((3), device="cuda")
        mode='rgb'
        render_pkg = gaussian_render(mode, viewpoint_cam, self.gaussians, bg)
        est_img_gauss, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        gt_img_gauss = viewpoint_cam.original_image.cuda()
        gaussian_l1 = torch.abs((est_img_gauss - gt_img_gauss)).mean()
        mse = (((est_img_gauss - gt_img_gauss)) ** 2).view(est_img_gauss.shape[0], -1).mean(1, keepdim=True)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        psnr = psnr.mean()
        
        self.sphere_center= torch.tensor([0.0, 0.0, 0.0]).cuda()
        
        if self.dataset_name == "nerf":
            self.sphere_radius = 1.5
        elif self.dataset_name == "dtu":
            self.sphere_radius = 1.0
        
        if self.gs_render_depth:
            temp = torch.matmul(viewpoint_cam.world_view_transform.T, torch.cat((self.gaussians._xyz, torch.ones(self.gaussians._xyz.shape[0],1).cuda()), 1).T)
            temp = temp/(temp[-1:, :].repeat(4, 1))
            gaussian_pts_view = temp.T[:, :-1]
            gaussian_depth_view = gaussian_pts_view[:, -1][:, None].repeat(1,3)
            max_depth = gaussian_depth_view.max()
            bg = torch.zeros((3), device="cuda") + max_depth
            mode='depth'
            render_pkg = gaussian_render(mode, viewpoint_cam, self.gaussians, bg, gaussian_depth=gaussian_depth_view)
            self.rendered_depth = render_pkg["render"][0, :] #(800,800)
        else: # render depth from gaussian center (pointcloud) directly
            rendered_depth_path = os.path.join(self.rendered_depth_path, '{:02}.npy'.format(int(data['idx'])))
            if os.path.exists(rendered_depth_path):
                #load rendered depth directly from pointcloud
                depth_map = np.load(rendered_depth_path)
                self.rendered_depth = torch.from_numpy(depth_map).cuda()
            else: #save depth_map in first epoch, only slow in the first epoch
                if not os.path.exists(self.rendered_depth_path):
                    os.makedirs(self.rendered_depth_path, exist_ok=True)
                # render point-depth with o3d
                import open3d as o3d
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector((self.gaussians._xyz/1.5).detach().cpu().numpy())
                fx = data['intr'][0,0,0]
                fy = data['intr'][0,1,1]
                cx = data['intr'][0,0,2]
                cy = data['intr'][0,1,2]
                width = data['width']
                height = data['height']
                intrinsic = o3d.camera.PinholeCameraIntrinsic()
                intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
        
                # Create an offscreen renderer
                renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

                # Set background color (optional)
                renderer.scene.set_background([1, 1, 1, 1])  # RGBA (white background)
        
                # Define material for point cloud
                material = o3d.visualization.rendering.MaterialRecord()
                material.shader = "defaultUnlit"

                # Optionally adjust point size
                material.point_size = 2.0  # Adjust as needed
        
                # Add the point cloud geometry to the scene
                renderer.scene.add_geometry("pointcloud", pcd, material)
        
                # Create a PinholeCameraParameters object
                camera_params = o3d.camera.PinholeCameraParameters()
                camera_params.intrinsic = intrinsic
                extrinsic = np.eye(4)
                extrinsic[:3, :] = data['pose'].detach().cpu().numpy()
                camera_params.extrinsic = extrinsic

                # Apply camera parameters to the renderer's camera
                renderer.setup_camera(camera_params.intrinsic, camera_params.extrinsic)
                
                # Render the scene and capture the depth image
                depth_image = renderer.render_to_depth_image(z_in_view_space=True)

                # Convert to NumPy array for further processing
                depth_map = np.asarray(depth_image)*self.sphere_radius
                if self.dataset_name == "nerf":
                    depth_map[depth_map>6.5] = -1.
                elif self.dataset_name == "dtu":
                    depth_map[depth_map>4.5] = -1.
                np.save(rendered_depth_path, depth_map)
                self.rendered_depth = torch.from_numpy(depth_map).cuda()


        
        self.gaussian_pts_scaled = (self.gaussians._xyz - self.sphere_center) /self.sphere_radius
        self.gaussian_feat = self.neural_sdf.encode(self.gaussian_pts_scaled, cat_pts=False).to(torch.float32)
        self.gaussians._scaling_scaled = self.gaussians._scaling /self.sphere_radius
        self.gaussians._rotation_scaled = quaternion_to_matrix(self.gaussians._rotation)
        temp = self.gaussians._rotation_scaled * self.gaussians._scaling_scaled[:, None]
        cov_matrix = temp @ temp.transpose(-1, -2)
        self.gaussians.cov3D = torch.zeros((cov_matrix.shape[0], 6), dtype=torch.float, device=cov_matrix.device)
        self.gaussians.cov3D[:, 0] = cov_matrix[:, 0, 0]
        self.gaussians.cov3D[:, 1] = cov_matrix[:, 0, 1]
        self.gaussians.cov3D[:, 2] = cov_matrix[:, 0, 2]
        self.gaussians.cov3D[:, 3] = cov_matrix[:, 1, 1]
        self.gaussians.cov3D[:, 4] = cov_matrix[:, 1, 2]
        self.gaussians.cov3D[:, 5] = cov_matrix[:, 2, 2]
        self.gaussians.cov_matrix = cov_matrix

        self.rendered_depth_scale = (self.rendered_depth/self.sphere_radius).flatten()[data['ray_idx']]
        output = self.render_pixels(data["pose"], data["intr"], image_size=self.image_size_train,
                                    stratified=self.cfg_render.stratified, sample_idx=data["idx"],
                                    ray_idx=data["ray_idx"])
       
        
        return output


    def render_pixels(self, pose, intr, image_size, stratified=False, sample_idx=None, ray_idx=None):
        center, ray = camera.get_center_and_ray(pose, intr, image_size)  # [B,HW,3] all in world coord
        center = nerf_util.slice_by_ray_idx(center, ray_idx)  # [B,R,3]
        ray = nerf_util.slice_by_ray_idx(ray, ray_idx)  # [B,R,3]
        ray_unit = torch_F.normalize(ray, dim=-1)  # [B,R,3]
        output = self.render_rays(pose, center, ray_unit, ray_idx, sample_idx=sample_idx, stratified=stratified)
        return output

    def render_rays(self, pose, center, ray_unit, ray_idx, sample_idx=None, stratified=False):
        with torch.no_grad():
            near, far, outside = self.get_dist_bounds(center, ray_unit)
        app, app_outside = self.get_appearance_embedding(sample_idx, ray_unit.shape[1])
        
        output_object = self.render_rays_object(pose, center, ray_unit, ray_idx, near, far, outside, app, stratified=stratified)
        if self.with_background:
            output_background = self.render_rays_background(pose, center, ray_unit, far, app_outside, stratified=stratified)
            # Concatenate object and background samples.
            rgbs = torch.cat([output_object["rgbs"], output_background["rgbs"]], dim=2) 
            dists = torch.cat([output_object["dists"], output_background["dists"]], dim=2) 
            alphas = torch.cat([output_object["alphas"], output_background["alphas"]], dim=2)
            normals = torch.cat([output_object["normals"], output_background["normals"]], dim=2)
        else:
            rgbs = output_object["rgbs"]  # [B,R,No,3]
            dists = output_object["dists"]  # [B,R,No,1]
            alphas = output_object["alphas"]  # [B,R,No]
            normals = output_object["normals"]
        
        weights = render.alpha_compositing_weights(alphas)  # [B,R,No+Nb,1]
        rgb = render.composite(rgbs, weights)  # [B,R,3]
        
        if self.white_background:
            opacity_all = render.composite(1., weights)  # [B,R,1]
            rgb = rgb + (1 - opacity_all)
        # Collect output.
        output = dict(
            rgb=rgb,  # [B,R,3]
            opacity=output_object["opacity"],  # [B,R,1]/None
            outside=outside,  # [B,R,1]
            dists=dists,  # [B,R,No+Nb,1]
            weights=weights,  # [B,R,No+Nb,1]
            gradient=output_object["gradient"],  # [B,R,3]/None
            gradients=output_object["gradients"],  # [B,R,No,3]
            hessians=output_object["hessians"],  # [B,R,No,3]/None
            ray_idx=ray_idx # Only for objects case when sampling rays from valid mask
        )
        return output

    def render_rays_object(self, pose, center, ray_unit, ray_idx, near, far,  outside, app, stratified=False):
        with torch.no_grad():
            dists = self.sample_dists_all(center, ray_unit, near, far, stratified=stratified)  # [B,R,N,3]
            
        if self.surface_fusion:
            points_surface = camera.get_3D_points_from_dist(center, ray_unit, self.rendered_depth_scale[..., None, None])
            surface_diff_min, surface_idx = torch.min(torch.abs(dists.squeeze(0).squeeze(-1) - self.rendered_depth_scale[..., None].squeeze(0)), dim=-1)
        else:
            with torch.no_grad():
                surface_diff_min, surface_idx = torch.topk(torch.abs(dists.squeeze(0).squeeze(-1) - self.rendered_depth_scale[..., None].squeeze(0)), k=5, dim=-1, largest=False)
            dists_surface = torch.gather(dists.squeeze(-1), dim=2, index=surface_idx.unsqueeze(0))
            points_surface = camera.get_3D_points_from_dist(center, ray_unit, dists_surface[..., None])
        
        # knn
        actual_numpoints_tensor = torch.tensor([self.gaussian_pts_scaled.shape[0]], dtype=torch.int32).cuda() 
        kernel_size_tensor = torch.tensor([3,3,3], dtype=torch.int32).cuda()
        query_size_tensor = torch.tensor([3,3,3], dtype=torch.int32).cuda()
        
        if self.dataset_name == "nerf":
            K = 4
        elif self.dataset_name == "dtu":
            K = 1
        R = 512
        max_o = 610000
        P = 26
        if self.surface_fusion:
            SR = 1
            D = 1
        else:
            SR = 5
            D = 5
        scaled_vsize_tensor = torch.tensor([0.0050, 0.0050, 0.0050]).cuda()
        radius_limit_np = np.array(2 * scaled_vsize_tensor[0].cpu().numpy()).astype(np.float32) 
        gpu_maxthr = 1024
        NN = 2
        
        min_xyz, max_xyz = torch.min(self.gaussian_pts_scaled.unsqueeze(0), dim=-2)[0][0], torch.max(self.gaussian_pts_scaled.unsqueeze(0), dim=-2)[0][0]
        ranges = [-0.8, -0.8, -0.8, 0.8, 0.8, 0.8]
        ranges_min = torch.as_tensor(ranges[:3], dtype=torch.float32, device=min_xyz.device)
        ranges_max = torch.as_tensor(ranges[3:], dtype=torch.float32, device=min_xyz.device)
        if ranges is not None:
            min_xyz, max_xyz = torch.max(torch.stack([min_xyz, ranges_min], dim=0), dim=0)[0], torch.min(torch.stack([max_xyz, ranges_max], dim=0), dim=0)[0]
        min_xyz = min_xyz - scaled_vsize_tensor * kernel_size_tensor / 2
        max_xyz = max_xyz + scaled_vsize_tensor * kernel_size_tensor / 2

        ranges_tensor = torch.cat([min_xyz, max_xyz], dim=-1)
        scaled_vdim_tensor = torch.ceil((max_xyz - min_xyz)/scaled_vsize_tensor).to(torch.int32)

        sample_pidx_tensor, sample_loc_w_tensor, ray_mask_tensor = \
            query_worldcoords_cuda.woord_query_grid_point_index(points_surface, self.gaussian_pts_scaled.unsqueeze(0), actual_numpoints_tensor, kernel_size_tensor,
                                                                query_size_tensor, SR, K, R, D,
                                                                scaled_vdim_tensor,
                                                                max_o, P, radius_limit_np,
                                                                ranges_tensor,
                                                                scaled_vsize_tensor,
                                                                gpu_maxthr, NN)

        mask_temp = sample_pidx_tensor == -1
        sample_pidx_tensor_equal_to_minus_1 = torch.sum(mask_temp).item()
        ray_mask_true_indices = torch.nonzero(ray_mask_tensor.squeeze(), as_tuple=False).squeeze(1)
        #print("ray_mask_true_indices", ray_mask_true_indices, ray_mask_true_indices.shape) #pixel indices of valid rays
        
        sample_pidx_tensor_holder = -torch.ones(sample_pidx_tensor.shape[0], R, sample_pidx_tensor.shape[2], sample_pidx_tensor.shape[3]).to(torch.int32).cuda()
        sample_pidx_tensor_holder[:, ray_mask_true_indices.long(), :, :] = sample_pidx_tensor
        sample_pidx_tensor = sample_pidx_tensor_holder
        
        sample_pnt_mask = sample_pidx_tensor>=0
        sample_pidx_tensor = torch.clamp(sample_pidx_tensor, min=0).view(-1).long()
        sample_pnt_mask = sample_pnt_mask.view(-1)
        
        gaussian_sh = torch.index_select(self.gaussians._features_dc, 0, sample_pidx_tensor)
        gaussian_sh = gaussian_sh.view(-1, gaussian_sh.shape[-1])
        gaussian_sh = gaussian_sh[sample_pnt_mask]
        
        gaussian_rgb = SH2RGB(self.gaussians._features_dc)
        gaussian_rgb = torch.index_select(gaussian_rgb, 0, sample_pidx_tensor)
        gaussian_rgb = gaussian_rgb.view(-1, gaussian_rgb.shape[-1])
        gaussian_rgb = gaussian_rgb[sample_pnt_mask]
        
        gaussian_feat = torch.index_select(self.gaussian_feat, 0, sample_pidx_tensor)
        gaussian_feat = gaussian_feat.view(-1, gaussian_feat.shape[-1])
        gaussian_feat = gaussian_feat[sample_pnt_mask]
        
        gaussian_cov3D = torch.index_select(self.gaussians.cov3D, 0, sample_pidx_tensor)
        gaussian_cov3D = gaussian_cov3D.view(-1, 6)
        gaussian_cov3D = gaussian_cov3D[sample_pnt_mask]
        
        gaussian_opacity = torch.index_select(self.gaussians._opacity, 0, sample_pidx_tensor)
        gaussian_opacity = gaussian_opacity.view(-1, gaussian_opacity.shape[-1])
        gaussian_opacity = gaussian_opacity[sample_pnt_mask]
        
        gaussian_agg_feats = self.aggregator(gaussian_rgb, gaussian_sh, gaussian_feat, gaussian_cov3D)
        gaussian_agg_feats_holder = torch.zeros((sample_pnt_mask.shape[0], gaussian_agg_feats.shape[1])).cuda()
        gaussian_agg_feats_holder[sample_pnt_mask] = gaussian_agg_feats
        gaussian_agg_feats = gaussian_agg_feats_holder.view(-1, K, gaussian_agg_feats_holder.shape[-1])
        
        gaussian_opacity_holder = torch.zeros((sample_pnt_mask.shape[0], gaussian_opacity.shape[1])).cuda()
        gaussian_opacity_holder[sample_pnt_mask] = gaussian_opacity
        gaussian_opacity = gaussian_opacity_holder.view(-1, K, gaussian_opacity_holder.shape[-1])
        
        sampled_pts_coord = points_surface.view(-1, 3)
        neighbor_gaussian_coord = torch.index_select(self.gaussian_pts_scaled, 0, sample_pidx_tensor)
        neighbor_gaussian_coord = neighbor_gaussian_coord.view(-1, neighbor_gaussian_coord.shape[-1])
        neighbor_gaussian_coord = neighbor_gaussian_coord[sample_pnt_mask]
        neighbor_gaussian_coord_holder = torch.zeros((sample_pnt_mask.shape[0], neighbor_gaussian_coord.shape[1])).cuda()
        neighbor_gaussian_coord_holder[sample_pnt_mask] = neighbor_gaussian_coord
        neighbor_gaussian_coord = neighbor_gaussian_coord_holder.view(-1, K, neighbor_gaussian_coord_holder.shape[-1])
        
        neighbor_gaussian_cov = torch.index_select(self.gaussians.cov_matrix, 0, sample_pidx_tensor)
        neighbor_gaussian_cov = neighbor_gaussian_cov.view(-1, neighbor_gaussian_cov.shape[-2], neighbor_gaussian_cov.shape[-1])
        neighbor_gaussian_cov = neighbor_gaussian_cov[sample_pnt_mask] 
        neighbor_gaussian_cov = torch.linalg.inv(neighbor_gaussian_cov)
        neighbor_gaussian_cov_holder = torch.zeros((sample_pnt_mask.shape[0], neighbor_gaussian_cov.shape[1], neighbor_gaussian_cov.shape[2])).cuda()
        neighbor_gaussian_cov_holder[sample_pnt_mask] = neighbor_gaussian_cov
        neighbor_gaussian_cov = neighbor_gaussian_cov_holder.view(-1, K, neighbor_gaussian_cov_holder.shape[-2], neighbor_gaussian_cov_holder.shape[-1])
        
        diff = (sampled_pts_coord[:,None,:] - neighbor_gaussian_coord).unsqueeze(-1)
        
        sample_pnt_mask = sample_pnt_mask.view(-1, K)
        
        agg_weights = torch.exp(-0.5*(diff).transpose(-2,-1) @ neighbor_gaussian_cov @ diff)
        agg_weights = agg_weights * sample_pnt_mask[...,None, None]
        sample_pnt_mask_sum = sample_pnt_mask.bool().sum(-1)
        sample_pnt_mask_sum[sample_pnt_mask_sum==0] = 1
        gaussian_agg_feats = (gaussian_agg_feats * agg_weights.squeeze(-1) * gaussian_opacity).sum(1)/sample_pnt_mask_sum[:,None] 
        gaussian_agg_feats = gaussian_agg_feats.view(points_surface.shape[0], points_surface.shape[1], points_surface.shape[2], -1)

        '''
        # For debugging dense/surface fusion: Get the indices of the surface point among all key points along the ray
        #print("dists", dists.shape, dists)
        #print("self.rendered_depth_scale", self.rendered_depth_scale.shape)
        print("torch.abs(dists.squeeze(0).squeeze(-1) - self.rendered_depth_scale[..., None].squeeze(0))", torch.abs(dists.squeeze(0).squeeze(-1) - self.rendered_depth_scale[..., None].squeeze(0)).shape)
        #surface_diff_min, surface_idx = torch.min(torch.abs(dists.squeeze(0).squeeze(-1) - self.rendered_depth_scale[..., None].squeeze(0)), dim=-1)
        surface_diff_min, surface_idx = torch.topk(torch.abs(dists.squeeze(0).squeeze(-1) - self.rendered_depth_scale[..., None].squeeze(0)), k=5, dim=-1, largest=False)
        #print("surface_diff_min, surface_idx", surface_diff_min.shape, surface_idx.shape, surface_diff_min, surface_idx)
        #print("surface_idx", surface_idx.shape, surface_idx)
        surface_idx = surface_idx[ray_mask_true_indices]
        surface_diff_min = surface_diff_min[ray_mask_true_indices]
        print("surface_idx", surface_idx.shape, surface_idx)'''
    
        points = camera.get_3D_points_from_dist(center, ray_unit, dists)
        if self.surface_fusion:
            surface_idx = surface_idx[ray_mask_true_indices]
        sdfs, feats = self.neural_sdf.forward(points, gaussian_agg_feats, mode='replace', ray_idx=ray_mask_true_indices, surface_idx=surface_idx, surface_points=points_surface, surface_fusion=self.surface_fusion)  # [B,R,N,1],[B,R,N,K]
        
        '''
        # For debugging dense/surface fusion
        mask_tmp = (sdfs[:,:,:-1,:]*sdfs[:,:,1:,:])<0
        mask_tmp_sum = torch.sum(mask_tmp, -2)
        print("mask_tmp", mask_tmp.shape, torch.unique(mask_tmp, return_counts=True))
        print("mask_tmp_sum", mask_tmp_sum.shape, torch.unique(mask_tmp_sum, return_counts=True))
        '''
        
        sdfs[outside[..., None].expand_as(sdfs)] = self.outside_val
     
        # Compute 1st- and 2nd-order gradients.
        rays_unit = ray_unit[..., None, :].expand_as(points).contiguous()  # [B,R,N,3]
        gradients, hessians = self.neural_sdf.compute_gradients(points, training=self.training, sdf=sdfs)
        normals = torch_F.normalize(gradients, dim=-1)  # [B,R,N,3] (1,512,128,3)
        
        rot = pose[..., :3, :3]
        normal_cam = -normals @ rot.transpose(-1, -2) 
        rgbs = self.neural_rgb.forward(points, normals, rays_unit, feats, app=app)  # [B,R,N,3]
        alphas = self.compute_neus_alphas(ray_unit, sdfs, gradients, dists, dist_far=far[..., None],
                                          progress=self.progress)  # [B,R,N]
        '''
        # For debugging dense/surface fusion
        print("alphas", alphas.shape)
        print("alphas_range", torch.unique(alphas, return_counts=True))
        '''

        if not self.training:
            weights = render.alpha_compositing_weights(alphas)  # [B,R,N,1]
            opacity = render.composite(1., weights)  # [B,R,1]
            gradient = render.composite(gradients, weights)  # [B,R,3]
        else:
            opacity = None
            gradient = None
        # Collect output.
        output = dict(
            rgbs=rgbs,  # [B,R,N,3]
            normals = normals,
            normal_cam = normal_cam,
            sdfs=sdfs[..., 0],  # [B,R,N]
            dists=dists,  # [B,R,N,1]
            alphas=alphas,  # [B,R,N]
            opacity=opacity,  # [B,R,3]/None
            gradient=gradient,  # [B,R,3]/None
            gradients=gradients,  # [B,R,N,3]
            hessians=hessians,  # [B,R,N,3]/None
        )
        
        
        return output
    

    def render_rays_background(self, pose, center, ray_unit, far, app_outside, stratified=False):
        with torch.no_grad():
            dists = self.sample_dists_background(ray_unit, far, stratified=stratified)
        points = camera.get_3D_points_from_dist(center, ray_unit, dists)  # [B,R,N,3]

        sdfs, feats = self.neural_sdf.forward(points)
        gradients, hessians = self.neural_sdf.compute_gradients(points, training=self.training, sdf=sdfs)
        normals = torch_F.normalize(gradients, dim=-1)
        rot = pose[..., :3, :3]
        normal_cam = -normals @ rot.transpose(-1, -2) 

        rays_unit = ray_unit[..., None, :].expand_as(points)  # [B,R,N,3]
        rgbs, densities = self.background_nerf.forward(points, rays_unit, app_outside)  # [B,R,N,3]
        alphas = render.volume_rendering_alphas_dist(densities, dists)  # [B,R,N]
        
        
        points_view = points*self.sphere_radius + self.sphere_center
        points_view = torch.matmul(torch.cat((points_view, torch.ones(points_view.shape[0],points_view.shape[1],points_view.shape[2],1).cuda()), -1), self.viewpoint_cam.world_view_transform)
        points_view = points_view/points_view[:,:,:,-1:].repeat(1,1,1,4)
        sample_depth_view = points_view[:,:,:,2]

        # Collect output.
        output = dict(
            rgbs=rgbs,  # [B,R,3]
            dists=dists,  # [B,R,N,1]
            alphas=alphas,  # [B,R,N]
            depths=sample_depth_view,
            normals=normals,
            normal_cam = normal_cam,
            gradients=gradients
        )
        return output

    @torch.no_grad()
    def get_dist_bounds(self, center, ray_unit):
        dist_near, dist_far = nerf_util.intersect_with_sphere(center, ray_unit, radius=1.)
        dist_near.relu_()  # Distance (and thus depth) should be non-negative.
        outside = dist_near.isnan()
        dist_near[outside], dist_far[outside] = 1, 1.2  # Dummy distances. Density will be set to 0.
        return dist_near, dist_far, outside

    def get_appearance_embedding(self, sample_idx, num_rays):
        if self.with_appear_embed:
            # Object appearance embedding.
            num_samples_all = self.cfg_render.num_samples.coarse + \
                self.cfg_render.num_samples.fine * self.cfg_render.num_sample_hierarchy
            app = self.appear_embed(sample_idx)[:, None, None]  # [B,1,1,C]
            app = app.expand(-1, num_rays, num_samples_all, -1)  # [B,R,N,C]
            # Background appearance embedding.
            if self.with_background:
                app_outside = self.appear_embed_outside(sample_idx)[:, None, None]  # [B,1,1,C]
                app_outside = app_outside.expand(-1, num_rays, self.cfg_render.num_samples.background, -1)  # [B,R,N,C]
            else:
                app_outside = None
        else:
            app = app_outside = None
        return app, app_outside

    @torch.no_grad()
    def sample_dists_all(self, center, ray_unit, near, far, stratified=False):
        dists = nerf_util.sample_dists(ray_unit.shape[:2], dist_range=(near[..., None], far[..., None]),
                                       intvs=self.cfg_render.num_samples.coarse, stratified=stratified)
        
        if self.cfg_render.num_sample_hierarchy > 0:
            points = camera.get_3D_points_from_dist(center, ray_unit, dists)  # [B,R,N,3]
            sdfs = self.neural_sdf.sdf(points)
        for h in range(self.cfg_render.num_sample_hierarchy):
            dists_fine = self.sample_dists_hierarchical(dists, sdfs, inv_s=(64 * 2 ** h))  # [B,R,Nf,1]
            dists = torch.cat([dists, dists_fine], dim=2)  # [B,R,N+Nf,1]
            dists, sort_idx = dists.sort(dim=2)
            if h != self.cfg_render.num_sample_hierarchy - 1:
                points_fine = camera.get_3D_points_from_dist(center, ray_unit, dists_fine)  # [B,R,Nf,3]
                sdfs_fine = self.neural_sdf.sdf(points_fine)  # [B,R,Nf]
                sdfs = torch.cat([sdfs, sdfs_fine], dim=2)  # [B,R,N+Nf]
                sdfs = sdfs.gather(dim=2, index=sort_idx.expand_as(sdfs))  # [B,R,N+Nf,1]
        return dists

    def sample_dists_hierarchical(self, dists, sdfs, inv_s, robust=True, eps=1e-5):
        sdfs = sdfs[..., 0]  # [B,R,N]
        prev_sdfs, next_sdfs = sdfs[..., :-1], sdfs[..., 1:]  # [B,R,N-1]
        prev_dists, next_dists = dists[..., :-1, 0], dists[..., 1:, 0]  # [B,R,N-1]
        mid_sdfs = (prev_sdfs + next_sdfs) * 0.5  # [B,R,N-1]
        cos_val = (next_sdfs - prev_sdfs) / (next_dists - prev_dists + 1e-5)  # [B,R,N-1]
        if robust:
            prev_cos_val = torch.cat([torch.zeros_like(cos_val)[..., :1], cos_val[..., :-1]], dim=-1)  # [B,R,N-1]
            cos_val = torch.stack([prev_cos_val, cos_val], dim=-1).min(dim=-1).values  # [B,R,N-1]
        dist_intvs = dists[..., 1:, 0] - dists[..., :-1, 0]  # [B,R,N-1]
        est_prev_sdf = mid_sdfs - cos_val * dist_intvs * 0.5  # [B,R,N-1]
        est_next_sdf = mid_sdfs + cos_val * dist_intvs * 0.5  # [B,R,N-1]
        prev_cdf = (est_prev_sdf * inv_s).sigmoid()  # [B,R,N-1]
        next_cdf = (est_next_sdf * inv_s).sigmoid()  # [B,R,N-1]
        alphas = ((prev_cdf - next_cdf) / (prev_cdf + eps)).clip_(0.0, 1.0)  # [B,R,N-1]
        weights = render.alpha_compositing_weights(alphas)  # [B,R,N-1,1]
        dists_fine = self.sample_dists_from_pdf(dists, weights=weights[..., 0])  # [B,R,Nf,1]
        return dists_fine

    def sample_dists_background(self, ray_unit, far, stratified=False, eps=1e-5):
        inv_dists = nerf_util.sample_dists(ray_unit.shape[:2], dist_range=(1, 0),
                                           intvs=self.cfg_render.num_samples.background, stratified=stratified)
        dists = far[..., None] / (inv_dists + eps)  # [B,R,N,1]
        return dists

    def compute_neus_alphas(self, ray_unit, sdfs, gradients, dists, dist_far=None, progress=1., eps=1e-5):
        sdfs = sdfs[..., 0]  # [B,R,N]
        # SDF volume rendering in NeuS.
        print("self.s_var", self.s_var) # monitor the training
        inv_s = self.s_var.exp()
        true_cos = (ray_unit[..., None, :] * gradients).sum(dim=-1, keepdim=False)  # [B,R,N]
        iter_cos = self._get_iter_cos(true_cos, progress=progress)  # [B,R,N]
        # Estimate signed distances at section points
        if dist_far is None:
            dist_far = torch.empty_like(dists[..., :1, :]).fill_(1e10)  # [B,R,1,1]
        dists = torch.cat([dists, dist_far], dim=2)  # [B,R,N+1,1]
        dist_intvs = dists[..., 1:, 0] - dists[..., :-1, 0]  # [B,R,N]
        dist_most = dists[..., -1, 0] - dists[..., 0, 0]
        est_prev_sdf = sdfs - iter_cos * dist_intvs * 0.5  # [B,R,N]
        est_next_sdf = sdfs + iter_cos * dist_intvs * 0.5  # [B,R,N]
        prev_cdf = (est_prev_sdf * inv_s).sigmoid()  # [B,R,N]
        next_cdf = (est_next_sdf * inv_s).sigmoid()  # [B,R,N]
        haha = (prev_cdf-0.5)*(next_cdf-0.5)
        haha = (haha<0).to(torch.int32).sum(-1)
        alphas = ((prev_cdf - next_cdf) / (prev_cdf + eps)).clip_(0.0, 1.0)  # [B,R,N]
        return alphas

    def _get_iter_cos(self, true_cos, progress=1.):
        anneal_ratio = min(progress / self.anneal_end, 1.)
        # The anneal strategy below keeps the cos value alive at the beginning of training iterations.
        return -((-true_cos * 0.5 + 0.5).relu() * (1.0 - anneal_ratio) +
                 (-true_cos).relu() * anneal_ratio)  # always non-positive
