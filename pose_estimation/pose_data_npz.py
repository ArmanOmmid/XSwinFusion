import os
import shutil
import pickle
import numpy as np
from PIL import Image
import trimesh
import cv2

import torch

from .utils import back_project, crop_and_resize_multiple, enumerate_symmetries
from .pose_data import PoseData

class PoseDataNPZ():

    _npz_handlers_cache = {}

    def __init__(self, npz_data_path, data_path=None, models_path=None, levels=None, split=None, object_caching=False) -> None:
        
        self.npz_data_path = npz_data_path
        if data_path is not None and models_path is not None:
            self.pose_data = PoseData(data_path, models_path, object_caching=object_caching)
        else:
            assert os.path.exists(self.npz_data_path), "Must Provide NPZ Path if not providing data_path and model_path"
            print(f"Presumed Preloaded NPZ Dataset: {npz_data_path}")
            self.pose_data = None
            # NOTE : You cannot INTERNALLY do levels or splits this way

        self.npz(npz_data_path)

        if isinstance(object_caching, str):
            self.objects_npz_path = object_caching
        else:
            self.objects_npz_path = os.path.join(npz_data_path, "objects.npz")
        if os.path.exists(self.objects_npz_path):
            # Cache Handlers
            if  self.objects_npz_path not in self._npz_handlers_cache:
                self._npz_handlers_cache[self.objects_npz_path] = np.load(self.objects_npz_path, allow_pickle=True, mmap_mode="r")
            self.objects = self._npz_handlers_cache[self.objects_npz_path]
            self.info = self.objects["info"] # objects.csv
        else:
            self.info = self.pose_data.objects
            self.objects = None # Will have to get it manually from PoseData.get_mesh()

        self.object_RAM_cache = [None] * len(self.info)

        if levels is not None:
            levels = [levels] if isinstance(levels, int) else levels

        scenes_path = os.path.join(npz_data_path, "scenes")
        self.data = {}
        for file in os.listdir(scenes_path):
            key = tuple(int(i) for i in file.split(".")[0].split("-"))
            l, s, v = key
            if levels is not None and l not in levels:
                continue
            scene_path = os.path.join(npz_data_path, "scenes", f"{l}-{s}-{v}.npz")

            # Cache Handlers
            if scene_path not in self._npz_handlers_cache:
                self._npz_handlers_cache[scene_path] = np.load(scene_path, allow_pickle=True, mmap_mode="r") # NPZ Generator object 
            self.data[key] = self._npz_handlers_cache[scene_path]

            # color, depth, label, meta

        self.keylist = list(self.data.keys())

    def npz(self, npz_data_path):
        if self.pose_data is None:
            return
        self.npz_data_path = npz_data_path
        if os.path.exists(self.npz_data_path):
            print(f"NPZ Path Already Exists: {self.npz_data_path}")
            return
        self.pose_data.npz(self.npz_data_path)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def __len__(self):
        return len(self.keylist)

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.data[self.keylist[i]] # if you give an int
        else:
            return self.data[i] # if you give a key tuple (l, s, v)
        
    def get_mesh(self, obj_id):
        if self.object_RAM_cache[obj_id] is not None:
            return self.object_RAM_cache[obj_id]
        elif isinstance(self.objects, np.lib.npyio.NpzFile):
            mesh = self.objects[f"{obj_id}"].item()
        elif self.objects is None:
            mesh = self.pose_data.get_mesh(obj_id)
        
        self.object_RAM_cache[obj_id] = mesh # Cache the mesh!
        return mesh

    def get_info(self, obj_id):
        if self.info is None:
            return self.pose_data.get_info(obj_id)
        return self.info[obj_id]
    
    def sample_mesh(self, obj_id, n):
        return trimesh.sample.sample_surface(self.get_mesh(obj_id), n)[0] # samples, faces
    
    def meta(self, key):
        "Easily get meta info"
        return self[key]["meta"][()]

    def label(self, key):
        "Saftely Get Label"
        label = self[key]["label"]
        if type(label.dtype).__name__ == "dtype[object_]":
            return None
        else:
            return label
    
    # def __del__(self):
    #     try:
    #         self.objects.close()
    #     except:
    #         pass
    #     for loader in self.data.values():
    #         loader.close()

class PoseDataNPZTorch(torch.utils.data.Dataset):
    def __init__(self, npz_data_path, data_path=None, models_path=None, object_caching=False,
                 levels=None, split=None, samples=8_000,
                 resize=(144, 256), aspect_ratio=True, margin=12,
                 symmetry_pad=64, 
                 ): 

        assert samples is not None, "No Longer Supporting Variable Samples"

        self.data = PoseDataNPZ(npz_data_path, data_path, models_path, object_caching=object_caching, levels=levels, split=split)
        self.num_classes = len(self.data.info)
        self.samples = samples

        self.resize = resize
        self.aspect_ratio = aspect_ratio
        self.margin = margin

        self.source_pcd_cache = [None] * self.num_classes
        self.symmetry_cache = [None] * self.num_classes
        self.symmetry_pad = symmetry_pad

        self._data = []

        for i, key in enumerate(self.data.keylist):
            for obj_id in self.data.meta(key)["objects"]:
                self._data.append((key, obj_id))

    def __len__(self):
        return len(self.data)
    
    def sample_source_pcd(self, obj_id):
        if self.source_pcd_cache[obj_id] is None:
            self.source_pcd_cache[obj_id] = \
                self.data.sample_mesh(obj_id, self.samples).astype(np.float32)
            
        return self.source_pcd_cache[obj_id]
    
    def get_symmetry(self, obj_id):
        if self.symmetry_cache[obj_id] is None:
            sym_pad = torch.eye(3).unsqueeze(0).repeat(self.symmetry_pad, 1, 1)
            sym = enumerate_symmetries(self.data.get_info(obj_id)["geometric_symmetry"])
            sym_pad[:len(sym), :, :] = torch.cat(sym) # convert from list
            self.symmetry_cache[obj_id] = sym_pad.float()

        return self.symmetry_cache[obj_id]
        
    def __getitem__(self, i):
        key, obj_id = self._data[i]

        scene = self.data[key]

        color = scene["color"]
        depth = scene["depth"]
        meta = self.data.meta(key)
        label = self.data.label(key)
        if label is None:
            # Can't Do Segmentation here
            raise NotImplementedError
            # label = segmentation_model(color)
        
        mask = scene["label"] == obj_id


        (color, depth, mask), scale, translate = crop_and_resize_multiple(
            (color, depth, mask), 
            mask, target_size=self.resize, margin=self.margin, aspect_ratio=self.aspect_ratio)
        
        color = color.astype(np.float32) / 255
        depth = depth.astype(np.float32) / 1000

        color = np.transpose(color, (2, 0, 1)) # H W C -> C H W # NOTE : Do this after crop/resize

        target_pcd, mask_indices = back_project(depth, meta, mask, (scale, translate), samples=self.samples)
        target_pcd = target_pcd.astype(np.float32)

        t_samples = len(target_pcd)

        if t_samples < self.samples:
            repeats = np.ceil(self.samples / t_samples).astype(int)
            mask_indices = np.repeat(mask_indices.T, repeats, axis=0)[:self.samples].T # REMEMBER, these correspond to the 2D MASK!
            target_pcd = np.repeat(target_pcd, repeats, axis=0)[:self.samples]

        try:
            pose = meta["poses_world"][obj_id][:3, :] # 4x4 -> 3x4
        except Exception:
            pose = 0

        sym = self.data.info[obj_id]["geometric_symmetry"]

        source_pcd = self.sample_source_pcd(obj_id) * meta["scales"][obj_id]

        sym = self.get_symmetry(obj_id)

        return source_pcd, target_pcd, color, mask_indices, pose, sym, np.array(key), obj_id


class PoseDataNPZSegmentationTorch(torch.utils.data.Dataset):
    def __init__(self, npz_data_path, data_path=None, models_path=None, object_caching=False,
                 levels=None, split=None, resize=None): 

        self.data = PoseDataNPZ(npz_data_path, data_path, models_path, object_caching=object_caching, levels=levels, split=split)
        self.num_classes = len(self.data.info)

        self.resize = resize

        self._data = []

        for i, key in enumerate(self.data.keylist):
            self._data.append((key))

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, i):
        key = self._data[i]

        scene = self.data[key]

        color = scene["color"]
        # meta = self.data.meta(key)
        label = self.data.label(key)

        if self.resize:
            
            H, W = self.resize
            color = cv2.resize(color, (W, H), interpolation=cv2.INTER_NEAREST)
            if label is not None:
                label = cv2.resize(label, (W, H), interpolation=cv2.INTER_NEAREST)

        if label is None:
            label = 0 # Ignore label if testing

        color = np.transpose(color, (2, 0, 1))

        return color.astype(np.float32)/255, label, np.array(key)
