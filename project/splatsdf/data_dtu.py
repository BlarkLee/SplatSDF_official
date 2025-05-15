import json
import numpy as np
import torch
import torchvision.transforms.functional as torchvision_F
from PIL import Image, ImageFile
import cv2
import os

from project.nerf.datasets import base
from project.nerf.utils import camera

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset(base.Dataset):

    def __init__(self, cfg, is_inference=False):
        super().__init__(cfg, is_inference=is_inference, is_test=False)
        cfg_data = cfg.data
        self.root = cfg_data.root
        self.preload = cfg_data.preload
        self.H, self.W = cfg_data.val.image_size if is_inference else cfg_data.train.image_size
        meta_fname = f"{cfg_data.root}/transforms.json"
        with open(meta_fname) as file:
            self.meta = json.load(file)
        self.list = self.meta["frames"]
        if cfg_data[self.split].subset:
            subset = cfg_data[self.split].subset
            subset_idx = np.linspace(0, len(self.list), subset+1)[:-1].astype(int)
            self.list = [self.list[i] for i in subset_idx]
        
        camera_dict = np.load(os.path.join(cfg_data.gaussian_path, 'cameras.npz'))
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(len(self.list))]
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(len(self.list))]

        self.intrinsics_all = []
        self.pose_all = []
        for world_mat, scale_mat in zip(world_mats, scale_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = self.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.intrinsics_all = torch.stack(self.intrinsics_all)
        self.inv_intrinsics_all = torch.inverse(self.intrinsics_all)
        self.pose_all = torch.stack(self.pose_all)
        self.inv_pose_all = torch.inverse(self.pose_all)
        
        
        self.num_rays = cfg.model.render.rand_rays
        self.readjust = getattr(cfg_data, "readjust", None)
        # Preload dataset if possible.
        if cfg_data.preload:
            self.images = self.preload_threading(self.get_image, cfg_data.num_workers)
            self.cameras = self.preload_threading(self.get_camera, cfg_data.num_workers, data_str="cameras")
    
    def load_K_Rt_from_P(self, filename, P=None):
        if P is None:
            lines = open(filename).read().splitlines()
            if len(lines) == 4:
                lines = lines[1:]
            lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
            P = np.asarray(lines).astype(np.float32).squeeze()
    
        out = cv2.decomposeProjectionMatrix(P)
        K = out[0]
        R = out[1]
        t = out[2]

        K = K/K[2,2]
        intrinsics = np.eye(4)
        intrinsics[:3, :3] = K
    
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R.transpose()
        pose[:3,3] = (t[:3] / t[3])[:,0]
    
        return intrinsics, pose

    def __getitem__(self, idx):
        """Process raw data and return processed data in a dictionary.

        Args:
            idx: The index of the sample of the dataset.
        Returns: A dictionary containing the data.
                 idx (scalar): The index of the sample of the dataset.
                 image (R tensor): Image idx for per-image embedding.
                 image (Rx3 tensor): Image with pixel values in [0,1] for supervision.
                 intr (3x3 tensor): The camera intrinsics of `image`.
                 pose (3x4 tensor): The camera extrinsics [R,t] of `image`.
        """
        # Keep track of sample index for convenience.
        sample = dict(idx=idx)
        # Get the images.
        image, image_size_raw = self.images[idx] if self.preload else self.get_image(idx)
        image = self.preprocess_image(image)
        # Get the cameras (intrinsics and pose).
        intr = self.intrinsics_all[idx]
        pose = self.inv_pose_all[idx]
        intr = self.preprocess_camera(intr, image_size_raw)
        pose = pose[:3, :]
        intr = intr[:3, :3]
        # Pre-sample ray indices.
        if self.split == "train":
            ray_idx = torch.randperm(self.H * self.W)[:self.num_rays]  # [R]
            sample.update(
                ray_idx=ray_idx,
                image_sampled=image,
                intr=intr,
                pose=pose,
            )
        else:  # keep image during inference
            sample.update(
                image=image,
                intr=intr,
                pose=pose,
            )
        return sample

    def get_image(self, idx):
        fpath = self.list[idx]["file_path"]
        image_fname = f"{self.root}/{fpath}"
        image = Image.open(image_fname)
        image.load()
        image_size_raw = image.size
        return image, image_size_raw

    def preprocess_image(self, image):
        # Resize the image.
        image = image.resize((self.W, self.H))
        image = torchvision_F.to_tensor(image)
        rgb = image[:3]
        return rgb

    def get_camera(self, idx):
        # Camera intrinsics.
        intr = torch.tensor([[self.meta["fl_x"], self.meta["sk_x"], self.meta["cx"]],
                             [self.meta["sk_y"], self.meta["fl_y"], self.meta["cy"]],
                             [0, 0, 1]]).float()
        # Camera pose.
        c2w_gl = torch.tensor(self.list[idx]["transform_matrix"], dtype=torch.float32)
        c2w = self._gl_to_cv(c2w_gl)
        # center scene
        center = np.array(self.meta["sphere_center"])
        center += np.array(getattr(self.readjust, "center", [0])) if self.readjust else 0.
        c2w[:3, -1] -= center
        # scale scene
        scale = np.array(self.meta["sphere_radius"])
        scale *= getattr(self.readjust, "scale", 1.) if self.readjust else 1.
        c2w[:3, -1] /= scale
        w2c = camera.Pose().invert(c2w[:3])
        return intr, w2c

    def preprocess_camera(self, intr, image_size_raw):
        # Adjust the intrinsics according to the resized image.
        intr = intr.clone()
        raw_W, raw_H = image_size_raw
        intr[0] *= self.W / raw_W
        intr[1] *= self.H / raw_H
        return intr

    def _gl_to_cv(self, gl):
        # convert to CV convention used in Imaginaire
        cv = gl * torch.tensor([1, -1, -1, 1])
        return cv
