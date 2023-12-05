import os
import shutil
import pickle
import numpy as np
from PIL import Image
import trimesh
import torch

class PoseData:

    """
    scene : 
        - color : RGB image
        - depth : from camera view (convert from mm to m)
        - label : a segmentation image capture form a camera containing the target objects. The segmentations id for objects are from 0 to 78
        - meta : camera parameters, object names, ground_truth poses (comparison labels)
        - key : CUSTOM : level - scene - variant
    meta : 
        - poses_world (list) : 
            The length is NUM_OBJECTS
            A pose is a 4x4 transformation matrix (rotation and translation) for each object in the world frame, or None for non-existing objects.
        - extents (list) : 
            The length is NUM_OBJECTS
            An extent is a(3,) array, representing the size of each object in its canonical frame (without scaling), or None for non-existing objects. The order is xyz.
        - scales (list) : 
            The length is NUM_OBJECTS
            A scale is a (3,) array, representing the scale of each object, or None for non-existing objects
        - object_names(list):
            the object names of interest.
        - extrinsic:
            4x4 transformation matrix, world -> viewer(opencv)
        - intrinsic :
            3x3 matrix, viewer(opencv) -> image
        - object_ids (list) : 
            the object ids of interest.

        - unique : unique labels 
        - objects : CORRECT object ids list
    """

    INFO_HEADERS = ['object', 'class', 'source', 'location', 'metric', 'min_x', 'max_x', 'min_y', 'max_y', 'min_z', 'max_z', 'width', 'length', 'height', 'visual_symmetry', 'geometric_symmetry']
    CSV_FILENAME = "objects_v1.csv"
    DATA_FOLDERNAME = "v2.2"

    _npz_handlers_cache = {}

    def __init__(self, data_path, models_path, object_caching=False, split_processed_data=None) -> None:

        self.data_path = data_path
        self.models_path = models_path

        if split_processed_data is not None:
            self.objects, self.data, self.nested_data, self.object_cache = split_processed_data
        else:
            self.objects = self.organize_objects(os.path.join(data_path, self.CSV_FILENAME))
            self.data, self.nested_data = self.organize_data(os.path.join(data_path, self.DATA_FOLDERNAME))

        self.keylist = list(self.data.keys())

        self.object_cache = object_caching
        if isinstance(object_caching, str):
            self.object_cache_path = object_caching
            self.object_cache = True
        else:
            self.object_cache_path = os.path.join(self.models_path, "objects.npz")

        if os.path.exists(self.object_cache_path):
            self.object_cache = True # Why Not
        if self.object_cache not in [False, None]:
            if self.object_cache is True: # We may have gotten it from a split
                self.make_object_cache()
        else:
            self.object_cache = None

    def organize_objects(self, objects_path):
        object_lists = np.genfromtxt(objects_path, delimiter=',', skip_header=1, dtype=None, encoding="utf-8")
        objects = []
        for object_list in object_lists:
            objects_dict = {header : value for header, value in zip(self.INFO_HEADERS, object_list)}
            objects.append(objects_dict)
        return objects

    def get_mesh(self, object_id):
        if isinstance(self.object_cache, np.lib.npyio.NpzFile):
            return self.object_cache[f"{object_id}"].item()
        object_info = self.objects[object_id]
        location = object_info["location"].split("/")[-1]
        visual_dae_path = os.path.join(self.models_path, location, "visual_meshes", "visual.dae")
        mesh = trimesh.load(visual_dae_path, force="mesh")
        return mesh

    def get_info(self, object_id):
        return self.objects[object_id]
    
    def make_object_cache(self):
        if not os.path.exists(self.object_cache_path):
            object_infos = []
            object_meshes = {}
            for i in range(len(self.objects)):
                object_infos.append(self.get_info(i))
                object_meshes[f"{i}"] = self.get_mesh(i)
            np.savez(self.object_cache_path, **object_meshes, info=np.array(object_infos))
        
        if self.object_cache_path not in self._npz_handlers_cache:
            self._npz_handlers_cache[self.object_cache_path] = np.load(self.object_cache_path, allow_pickle=True, mmap_mode="r")
        self.object_cache = self._npz_handlers_cache[self.object_cache_path]

    def organize_data(self, data_path):
        data = {}
        nested_data = {}
        components = ["color", "depth", "label", "meta"]

        def get_component(filename):
            for component in components:
                if component in filename:
                    return component
            else:
                raise KeyError(f"{filename}")

        for filename in os.listdir(data_path):

            filepath = os.path.join(data_path, filename)
            component = get_component(filename)
            level, scene, variant = [int(idx) for idx in filename.split("_")[0].split("-")]
            key = (level, scene, variant)

            if key not in data:
                data[key] = {"key" : key}

            if level not in nested_data:
                nested_data[level] = {}
            if scene not in nested_data[level]:
                nested_data[level][scene] = {}
            if variant not in nested_data[level][scene]:
                nested_data[level][scene][variant] = {"key" : key}

            if component == "meta" :
                with open(filepath, "rb") as f:
                    entry = pickle.load(f)
            else:
                # Normlize condition
                dtype = None
                dtype = np.uint8 if component == "color" \
                        else np.int16 if component == "depth" \
                        else dtype
                # Closure over variable
                def closure(filepath, dtype=None):
                    # Generate PNG
                    def generator():
                        if dtype is not None:
                            return np.array(Image.open(filepath), dtype=dtype)
                        return np.array(Image.open(filepath))
                    return generator
                entry = closure(filepath, dtype)

            # Save memory by making these point to the same object
            data[key][component] = nested_data[level][scene][variant][component] = entry

        return data, nested_data

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def __len__(self):
        return len(self.keylist)

    def __call__(self, idx):
        # Get from global, flattened index
        return self.data[self.keylist[idx]]

    def __getitem__(self, indices):
        if isinstance(indices, int):
            indices = (1,)
        if len(indices) == 3:
            return self.data[indices]
        value = self.nested_data
        for i in indices:
            value = value[i]
        return value

    def level_split(self, selected_levels):
        
        if not isinstance(selected_levels, (tuple, list)):
            selected_levels = [selected_levels]

        # Create splits
        split_data = {}
        split_nested_data = {}
        for key in self.data.keys():
            level, scene, variant = key

            if level not in selected_levels:
                continue

            split_data[key] = self.data[key]

            if level not in split_nested_data:
                split_nested_data[level] = {}
            if scene not in split_nested_data[level]:
                split_nested_data[level][scene] = {}
            if variant not in split_nested_data[level][scene]:
                split_nested_data[level][scene][variant] = self.data[key]

        return PoseData(self.data_path, self.models_path, split_processed_data=(self.objects, split_data, split_nested_data, self.object_cache))

    def txt_split(self, split_txt_path):

        # Load split txt file
        string_indices = np.loadtxt(split_txt_path, dtype=str)

        # Convert to Tuple
        tuple_indices = []
        for string in string_indices:
            indices = [int(idx) for idx in string.split("-")]
            tuple_indices.append(tuple(indices))

        # Create splits
        split_data = {}
        split_nested_data = {}
        for key in tuple_indices:
            level, scene, variant = key

            split_data[key] = self.data[key]

            if level not in split_nested_data:
                split_nested_data[level] = {}
            if scene not in split_nested_data[level]:
                split_nested_data[level][scene] = {}
            if variant not in split_nested_data[level][scene]:
                split_nested_data[level][scene][variant] = self.data[key]

        return PoseData(self.data_path, self.models_path, split_processed_data=(self.objects, split_data, split_nested_data, self.object_cache))

    # def __del__(self):
    #     try:
    #         self.object_cache.close()
    #     except:
    #         pass

    def npz(self, npz_dataset_path, levels=None, split=None):
        
        pose_data = self
        if split is not None:
            pose_data = self.txt_split(split)
        if levels is not None:
            pose_data = self.level_split(levels)

        os.makedirs(npz_dataset_path)

        if isinstance(self.object_cache, np.lib.npyio.NpzFile):
            shutil.copy(self.object_cache_path, os.path.join(npz_dataset_path, "objects.npz"))
        else:
            print("WARNING: objects.npz is not cached")
        
        scene_path = os.path.join(npz_dataset_path, "scenes")
        os.makedirs(scene_path)
        for i, key in enumerate(self.data.keys()):

            l, s, v = key

            scene = pose_data.data[key]

            color = scene["color"]() # normalization 255
            depth = scene["depth"]() # normalization 1000
            meta = scene["meta"]
            if "label" in scene:
                label = scene["label"]()
                meta["unique"] = list(np.unique(label))
                meta["objects"] = [id for id in meta["unique"] if id < 79]
            else:
                label = None
                meta["unique"] = None
                meta["objects"] = None

            scene_path_i = os.path.join(scene_path, f"{l}-{s}-{v}")
            np.savez(scene_path_i, color=color, depth=depth, label=label, meta=meta)

        return self.data.keys()

            