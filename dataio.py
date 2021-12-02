from torch.utils.data import Dataset
import glob
import numpy as np
import torch
import random
from scipy.io import loadmat
import os.path as osp
from skimage.transform import resize
from scipy.spatial.transform import Rotation
import pickle
import geometry
import matplotlib.pyplot as plt
import util


class OccTrainDataset(Dataset):

    def __init__(self, phase='train'):
        self.shapenet_dict = pickle.load(open("shapenet.p", "rb"))
        self.keys = list(self.shapenet_dict.keys())
        self.n = len(self.keys)

        block = 128
        self.hbs = 1 / block / 2.

    def __len__(self):
        return 100000

    def get_item(self, index):
        # In case of a missing file or IO error, switch to a random sample instead
        ix = random.randint(0, self.n - 1)
        key = self.keys[ix]
        coord, voxel_bool, voxel_pos = self.shapenet_dict[key]

        rix = np.random.permutation(voxel_bool.shape[0])
        rix_pos = np.random.permutation(voxel_pos.shape[0])

        voxel_pos_select = voxel_pos[rix_pos[:1000]]
        voxel_bool_select = (voxel_bool[rix[:1000]].astype(np.float32) - 0.5) * 2
        coord_select = coord[rix[:1000]]

        offset = np.random.uniform(-self.hbs, self.hbs, coord_select.shape)
        coord_select = coord_select + offset

        offset = np.random.uniform(-self.hbs, self.hbs, coord_select.shape)
        voxel_pos_select = voxel_pos_select + offset

        res = {'depth_coords': torch.from_numpy(voxel_pos_select).float(),
               'coords': torch.from_numpy(coord_select).float(),
               }
        return res, {'occ': torch.from_numpy(voxel_bool_select).float()}


    def __getitem__(self, index):
        return self.get_item(index)


class DepthDistTrainDataset(Dataset):

    def __init__(self, sidelength, phase='train'):

        self.root = "/data/scratch/asimeonov/repos/research/PIFU_robot/data_gen/data/mug_table_upright_pose_4_cam_half_occ_full_rand_scale"
        self.files = glob.glob(self.root+"/*.npz")
        self.files = sorted(self.files)

        self.sidelength = sidelength

        block = 64
        bs = 1 / block
        hbs = bs * 0.5
        y, z, x = np.meshgrid(np.linspace(-.5+hbs, .5-hbs, block), np.linspace(-.5+hbs, .5-hbs, block), np.linspace(-.5+hbs, .5-hbs, block))
        self.bs = bs
        self.hbs = hbs
        # voxel = np.stack([z, x, -y], axis=-1)
        # self.voxel = voxel.reshape((-1, 3))

        self.shapenet_dict = pickle.load(open("shapenet_dist.p", "rb"))
        self.projection_mode = "perspective"

        n = len(self.files)
        idx = int(0.9 * n)

        if phase == 'train':
            self.files = self.files[:idx]
        else:
            self.files = self.files[idx:]

        self.cache_file = None
        self.count = 0

        print("files length ", len(self.files))

    def __len__(self):
        return len(self.files)

    def get_item(self, index):
        # In case of a missing file or IO error, switch to a random sample instead
        # try:

       #  if self.cache_file is None or self.count % 100 == 0:
       #      file = random.choice(self.files)
       #      data = np.load(file, allow_pickle=True)
       #      self.cache_file = data
        data = np.load(self.files[index], allow_pickle=True)
        posecam =  data['object_pose_cam_frame']
        pos = posecam[0, :3]
        quat = posecam[0, 3:]

        pos2 = posecam[1, :3]
        quat2 = posecam[1, 3:]

        shapenet_id = str(data['shapenet_id'].item())
        category_id = str(data['shapenet_category_id'].item())
        vertex_offset = data['vertex_offset']

        depths = []
        segs = []
        for i in range(2):
            seg = data['object_segmentation'][i, 0]
            depth = data['depth_observation'][i]
            segs.append(seg)
            depths.append(torch.from_numpy(depth))

        y, x = torch.meshgrid(torch.arange(480), torch.arange(640))

        # Compute native intrinsic matrix
        sensor_half_width = 320
        sensor_half_height = 240

        # hor_fov = 60 * np.pi / 180
        vert_fov = 60 * np.pi / 180

        vert_f = sensor_half_height / np.tan(vert_fov / 2)
        hor_f = sensor_half_width / (np.tan(vert_fov / 2) * 320 / 240)

        intrinsics = np.array(
            [[hor_f, 0., sensor_half_width, 0.],
             [0., vert_f, sensor_half_height, 0.],
             [0., 0., 1., 0.]]
        )

        # Rescale to new sidelength
        # intrinsics[:2, :3] *= np.array([128/640, 128/480])[:, None]
        intrinsics = torch.from_numpy(intrinsics)

        seg_mask = segs[0]
        dp_np_first = geometry.lift(x.flatten()[seg_mask], y.flatten()[seg_mask], depths[0].flatten(), intrinsics[None, :, :])

        seg_mask = segs[1]
        dp_np_second = geometry.lift(x.flatten()[seg_mask], y.flatten()[seg_mask], depths[1].flatten(), intrinsics[None, :, :])

        shapenet_path = "/om2/user/yilundu/occupancy_network_new/data/ShapeNet_new/data/vision/billf/object-properties/dataset/billf-6/ShapeNetCore.v2/"

        voxel_path = osp.join(shapenet_path, category_id, shapenet_id, 'models', 'model_normalized_128.mat')
        coord, voxel_bool, voxel_pos = self.shapenet_dict[voxel_path]

        rix = np.random.permutation(coord.shape[0])

        coord = coord[rix[:1500]]
        label = voxel_bool[rix[:1500]] * data['mesh_scale']

        offset = np.random.uniform(-self.hbs, self.hbs, coord.shape)
        coord = coord + offset
        coord = coord_orig = coord * data['mesh_scale']

        coord = torch.from_numpy(coord)

        quat_list = [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
        rotation_matrix = Rotation.from_quat(quat_list)
        rotation_matrix = rotation_matrix.as_matrix()

        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, -1] = pos

        quat_list = [float(quat2[0]), float(quat2[1]), float(quat2[2]), float(quat2[3])]

        rotation_matrix = Rotation.from_quat(quat_list)
        rotation_matrix = rotation_matrix.as_matrix()

        transform2 = np.eye(4)
        transform2[:3, :3] = rotation_matrix
        transform2[:3, -1] = pos2

        transform = torch.from_numpy(transform)
        transform2 = torch.from_numpy(transform2)

        dp_np_first = torch.cat([dp_np_first, torch.ones_like(dp_np_first[..., :1])], dim=-1)
        dp_np_second = torch.cat([dp_np_second, torch.ones_like(dp_np_second[..., :1])], dim=-1)

        coord = torch.cat([coord, torch.ones_like(coord[..., :1])], dim=-1)
        coord = torch.sum(transform[None, :, :] * coord[:, None, :], dim=-1)
        coord = coord[..., :3]

        point_transform = torch.matmul(transform, torch.inverse(transform2))
        dp_np_second = torch.sum(point_transform[None, :, :] * dp_np_second[:, None, :], dim=-1)

        depth_coords = torch.cat([dp_np_first[..., :3], dp_np_second[..., :3]], dim=0)

        rix = torch.randperm(depth_coords.size(0))
        depth_coords = depth_coords[rix[:1000]]

        center = (depth_coords.min(dim=0)[0] + depth_coords.max(dim=0)[0]) / 2.
        coord = coord - center[None, :]
        depth_coords = depth_coords - center[None, :]

        all_coords = coord

        if 'point_cloud' in data.files:
            gt_pointcloud = data['point_cloud']
            if gt_pointcloud.shape[0] >= 1000:
                gt_pointcloud = torch.from_numpy(gt_pointcloud)
                gt_pcd_idx = torch.randperm(gt_pointcloud.size(0))
                gt_pointcloud = gt_pointcloud[gt_pcd_idx[:1000]]
            else:
                gt_pointcloud = torch.rand(1000, 3)
        else:
            gt_pointcloud = torch.rand(1000, 3)

        res = {'depth_coords': depth_coords.float(),
               'coords': coord.float(),
               'intrinsics':intrinsics.float(),
               'pointcloud': gt_pointcloud.float(),
               }
        return res, {'occ': torch.from_numpy(label).float()}

        # except Exception as e:
        #     print(e)
        #     # print(file)
        #     return self.get_item(index=random.randint(0, self.__len__() - 1))

    def __getitem__(self, index):
        return self.get_item(index)


class RackOccTrainDataset(Dataset):

    def __init__(self, sidelength, depth_aug=False, multiview_aug=False, phase='train'):
        # Path setup
        self.root = "/data/scratch/asimeonov/shared_data/rack_rand_scale"
        self.files = glob.glob(self.root+"/*.npz")
        self.files = sorted(self.files)

        self.sidelength = sidelength

        racks_path = "/data/scratch/asimeonov/shared_data/racks/rack_occupancy/*.npz"
        racks_path = glob.glob(racks_path)
        rack_dict = {}
        self.depth_aug = depth_aug
        self.multiview_aug = multiview_aug

        for rack in racks_path:
            rack_key = rack.split("/")[-1]
            data = np.load(rack)
            points = data['points']
            occupancy = data['occupancy']

            rack_dict[rack_key] = (points, occupancy)

        self.rack_dict = rack_dict
        racks = []

        self.projection_mode = "perspective"

        n = len(self.files)
        idx = int(0.9 * n)

        if phase == 'train':
            self.files = self.files[:idx]
        else:
            self.files = self.files[idx:]


    def __len__(self):
        return len(self.files)

    def get_item(self, index):

        # try:
            data = np.load(self.files[index], allow_pickle=True)
            posecam =  data['object_pose_cam_frame']

            idxs = list(range(posecam.shape[0]))
            random.shuffle(idxs)
            select = random.randint(1, 4)

            if self.multiview_aug:
                idxs = idxs[:select]

            poses = []
            quats = []
            for i in idxs:
                pos = posecam[i, :3]
                quat = posecam[i, 3:]

                poses.append(pos)
                quats.append(quat)

            file_id = str(data['obj_file']).split("/")[-1]

            depths = []
            segs = []
            cam_poses = []

            for i in idxs:
                seg = data['object_segmentation'][i, 0]
                depth = data['depth_observation'][i]
                rix = np.random.permutation(depth.shape[0])[:1000]
                seg = seg[rix]
                depth = depth[rix]

                if self.depth_aug:
                    depth = depth + np.random.randn(*depth.shape) * 0.1

                segs.append(seg)
                depths.append(torch.from_numpy(depth))
                cam_poses.append(data['cam_pose_world'][i])

            y, x = torch.meshgrid(torch.arange(480), torch.arange(640))

            # Compute native intrinsic matrix
            sensor_half_width = 320
            sensor_half_height = 240

            # hor_fov = 60 * np.pi / 180
            vert_fov = 60 * np.pi / 180

            vert_f = sensor_half_height / np.tan(vert_fov / 2)
            hor_f = sensor_half_width / (np.tan(vert_fov / 2) * 320 / 240)

            intrinsics = np.array(
                [[hor_f, 0., sensor_half_width, 0.],
                 [0., vert_f, sensor_half_height, 0.],
                 [0., 0., 1., 0.]]
            )

            # Rescale to new sidelength
            # intrinsics[:2, :3] *= np.array([128/640, 128/480])[:, None]
            intrinsics = torch.from_numpy(intrinsics)

            dp_nps = []

            for i in range(len(segs)):
                seg_mask = segs[i]
                dp_np = geometry.lift(x.flatten()[seg_mask], y.flatten()[seg_mask], depths[i].flatten(), intrinsics[None, :, :])
                dp_np = torch.cat([dp_np, torch.ones_like(dp_np[..., :1])], dim=-1)
                dp_nps.append(dp_np)

            shapenet_path = "/data/vision/billf/object-properties/dataset/billf-6/ShapeNetCore.v2/"

            file_id =  file_id[:-4] + "_occupancy.npz"
            coord, voxel_bool = self.rack_dict[file_id]

            rix = np.random.permutation(coord.shape[0])

            coord = coord[rix[:1500]]
            label = voxel_bool[rix[:1500]]

            offset = np.random.uniform(-0.01, 0.01, coord.shape)
            coord = coord + offset
            coord = coord_orig = coord * data['mesh_scale']

            coord = torch.from_numpy(coord)

            transforms = []
            for quat, pos in zip(quats, poses):
                quat_list = [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
                rotation_matrix = Rotation.from_quat(quat_list)
                rotation_matrix = rotation_matrix.as_matrix()

                transform = np.eye(4)
                transform[:3, :3] = rotation_matrix
                transform[:3, -1] = pos
                transform = torch.from_numpy(transform)
                transforms.append(transform)


            transform = transforms[0]
            coord = torch.cat([coord, torch.ones_like(coord[..., :1])], dim=-1)
            coord = torch.sum(transform[None, :, :] * coord[:, None, :], dim=-1)
            coord = coord[..., :3]

            dp_np_extra = []

            for i, dp_np in enumerate(dp_nps):
                point_transform = torch.matmul(transform, torch.inverse(transforms[i]))
                dp_np = torch.sum(point_transform[None, :, :] * dp_np[:, None, :], dim=-1)
                dp_np_extra.append(dp_np[..., :3])

            depth_coords = torch.cat(dp_np_extra, dim=0)

            rix = torch.randperm(depth_coords.size(0))
            depth_coords = depth_coords[rix[:1000]]

            if depth_coords.size(0) != 1000:
                return self.get_item(index=random.randint(0, self.__len__() - 1))

            label = (label - 0.5) * 2.0
            # center = (depth_coords.min(dim=0)[0] + depth_coords.max(dim=0)[0]) / 2.
            center = depth_coords.mean(dim=0)
            coord = coord - center[None, :]
            depth_coords = depth_coords - center[None, :]

            all_coords = coord
            labels = label

            # if 'point_cloud' in data.files:
            #     gt_pointcloud = data['point_cloud']
            #     if gt_pointcloud.shape[0] >= 1000:
            #         gt_pointcloud = torch.from_numpy(gt_pointcloud)
            #         gt_pcd_idx = torch.randperm(gt_pointcloud.size(0))
            #         gt_pointcloud = gt_pointcloud[gt_pcd_idx[:1000]]
            #     else:
            #         gt_pointcloud = torch.rand(1000, 3)
            # else:
            gt_pointcloud = torch.rand(1000, 3)

            res = {'depth_coords': depth_coords.float(),
                   'coords': coord.float(),
                   'intrinsics':intrinsics.float(),
                   'pointcloud': gt_pointcloud.float(),
                   'cam_poses': np.zeros(1)}
            return res, {'occ': torch.from_numpy(labels).float()}

        # except Exception as e:
        #     print(e)
        #     # print(file)
        #     return self.get_item(index=random.randint(0, self.__len__() - 1))

    def __getitem__(self, index):
        return self.get_item(index)


class RepBCTrainDataset(Dataset):

    def __init__(self, sidelength, depth_aug=False, multiview_aug=False, phase='train'):
        # Path setup
        viewMatrix = (-1.9626155728532085e-16, -0.7071068286895752, 0.7071068286895752, 0.0, 1.0, 0.0, 2.7755578262606874e-16, 0.0, -1.9626155728532085e-16, 0.7071068286895752, 0.7071068286895752, 0.0, 3.434577252493115e-16, 0.1767767071723938, -1.2374370098114014, 1.0)
        self.viewMatrix = np.array(viewMatrix).reshape((4, 4)).transpose()
        self.root = "/data/scratch/asimeonov/shared_data/rack_rand_scale"

        if phase == "train":
            states = sorted(glob.glob("/data/vision/billf/scratch/yilundu/rep4bc/data/pushing_box-train/state/*.pkl"))
            depths = sorted(glob.glob("/data/vision/billf/scratch/yilundu/rep4bc/data/pushing_box-train/depth/*.pkl"))
        else:
            states = sorted(glob.glob("/data/vision/billf/scratch/yilundu/rep4bc/data/pushing_box-test/state/*.pkl"))
            depths = sorted(glob.glob("/data/vision/billf/scratch/yilundu/rep4bc/data/pushing_box-test/depth/*.pkl"))

        data = np.load("cracker_box.npz")
        self.cube_pos = data['pos']
        self.cube_neg = data['neg']

        state_arr = []
        depth_arr = []

        for state, depth in zip(states, depths):
            state = pickle.load(open(state, "rb"))
            depth = pickle.load(open(depth, "rb"))

            depth_arr.extend(depth)
            state_arr.extend(state)

        self.depth = depth_arr
        self.state = state_arr


    def __len__(self):
        return len(self.depth)

    def get_item(self, index):

        depth = self.depth[index]
        _, pos, quat = self.state[index]

        quat_list = [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
        rotation_matrix = Rotation.from_quat(quat_list)
        rotation_matrix = rotation_matrix.as_matrix()

        pos = np.array(pos)
        y, x = torch.meshgrid(torch.arange(480), torch.arange(640))

        # Compute native intrinsic matrix
        sensor_half_width = 320
        sensor_half_height = 240

        # hor_fov = 60 * np.pi / 180
        vert_fov = 60 * np.pi / 180

        vert_f = 450
        hor_f = 450

        intrinsics = np.array(
            [[hor_f, 0., sensor_half_width, 0.],
             [0., vert_f, sensor_half_height, 0.],
             [0., 0., 1., 0.]]
        )

        # Rescale to new sidelength
        # intrinsics[:2, :3] *= np.array([128/640, 128/480])[:, None]
        intrinsics = torch.from_numpy(intrinsics)

        depth_coords = geometry.lift(x.flatten(), y.flatten(), depth.flatten(), intrinsics[None, :, :])

        rix = np.random.permutation(self.cube_pos.shape[0])
        coord_pos = self.cube_pos[rix[:750]]
        coord_neg = self.cube_neg[rix[750:1500]]

        coord = np.concatenate([coord_pos, coord_neg], axis=0)
        label = np.concatenate([np.ones(750), np.zeros(750)], axis=-1)

        coord = np.sum(rotation_matrix[None, :, :] * coord[:, None, :], axis=-1)
        coord = coord + pos[None, :]
        offset = np.random.uniform(-0.01, 0.01, coord.shape)
        coord = coord + offset

        coord = np.concatenate([coord, np.ones_like(coord[..., :1])], axis=-1)
        viewMatrix = self.viewMatrix
        coord = np.sum(viewMatrix[None, :, :] * coord[:, None, :], axis=-1)

        label = (label - 0.5) * 2.0
        # center = (depth_coords.min(dim=0)[0] + depth_coords.max(dim=0)[0]) / 2.
        rix = np.random.permutation(depth_coords.shape[0])

        depth_coords = depth_coords[rix[:1024]]

        labels = label

        res = {'depth_coords': torch.Tensor(depth_coords).float()[..., :3],
                'coords': torch.Tensor(coord).float()[..., :3],
               'intrinsics':intrinsics.float(),
               'cam_poses': np.zeros(1)}
        return res, {'occ': torch.from_numpy(labels).float()}

    def __getitem__(self, index):
        return self.get_item(index)


class DepthOccTrainDataset(Dataset):

    def __init__(self, sidelength, phase='train'):

        # Path setup
        self.root = "/data/scratch/asimeonov/repos/research/PIFU_robot/data_gen/data/mug_table_upright_pose_4_cam_half_occ_full_rand_scale"
        self.files = glob.glob(self.root+"/*.npz")
        self.files = sorted(self.files)

        self.sidelength = sidelength

        block = 64
        bs = 1 / block
        hbs = bs * 0.5
        y, z, x = np.meshgrid(np.linspace(-.5+hbs, .5-hbs, block), np.linspace(-.5+hbs, .5-hbs, block), np.linspace(-.5+hbs, .5-hbs, block))
        self.bs = bs
        self.hbs = hbs
        # voxel = np.stack([z, x, -y], axis=-1)
        # self.voxel = voxel.reshape((-1, 3))

        self.shapenet_dict = pickle.load(open("shapenet_mug.p", "rb"))
        self.projection_mode = "perspective"

        n = len(self.files)
        idx = int(0.9 * n)

        if phase == 'train':
            self.files = self.files[:idx]
        else:
            self.files = self.files[idx:]

        self.cache_file = None
        self.count = 0

        print("files length ", len(self.files))

    def __len__(self):
        return len(self.files)

    def get_item(self, index):
        # In case of a missing file or IO error, switch to a random sample instead
        # try:

       #  if self.cache_file is None or self.count % 100 == 0:
       #      file = random.choice(self.files)
       #      data = np.load(file, allow_pickle=True)
       #      self.cache_file = data
        try:
            data = np.load(self.files[index], allow_pickle=True)
            posecam =  data['object_pose_cam_frame']
            pos = posecam[0, :3]
            quat = posecam[0, 3:]

            pos2 = posecam[1, :3]
            quat2 = posecam[1, 3:]

            shapenet_id = str(data['shapenet_id'].item())
            category_id = str(data['shapenet_category_id'].item())
            vertex_offset = data['vertex_offset']

            depths = []
            segs = []
            cam_poses = []
            for i in range(2):
                seg = data['object_segmentation'][i, 0]
                depth = data['depth_observation'][i]
                segs.append(seg)
                depths.append(torch.from_numpy(depth))
                cam_poses.append(data['cam_pose_world'][i])

            y, x = torch.meshgrid(torch.arange(480), torch.arange(640))

            # Compute native intrinsic matrix
            sensor_half_width = 320
            sensor_half_height = 240

            # hor_fov = 60 * np.pi / 180
            vert_fov = 60 * np.pi / 180

            vert_f = sensor_half_height / np.tan(vert_fov / 2)
            hor_f = sensor_half_width / (np.tan(vert_fov / 2) * 320 / 240)

            intrinsics = np.array(
                [[hor_f, 0., sensor_half_width, 0.],
                 [0., vert_f, sensor_half_height, 0.],
                 [0., 0., 1., 0.]]
            )

            # Rescale to new sidelength
            # intrinsics[:2, :3] *= np.array([128/640, 128/480])[:, None]
            intrinsics = torch.from_numpy(intrinsics)

            seg_mask = segs[0]
            dp_np_first = geometry.lift(x.flatten()[seg_mask], y.flatten()[seg_mask], depths[0].flatten(), intrinsics[None, :, :])

            seg_mask = segs[1]
            dp_np_second = geometry.lift(x.flatten()[seg_mask], y.flatten()[seg_mask], depths[1].flatten(), intrinsics[None, :, :])

            shapenet_path = "/data/vision/billf/object-properties/dataset/billf-6/ShapeNetCore.v2/"

            voxel_path = osp.join(shapenet_path, category_id, shapenet_id, 'models', 'model_normalized_128.mat')
            coord, voxel_bool, voxel_pos = self.shapenet_dict[voxel_path]

            rix = np.random.permutation(coord.shape[0])

            coord = coord[rix[:1500]]
            label = voxel_bool[rix[:1500]]

            offset = np.random.uniform(-self.hbs, self.hbs, coord.shape)
            coord = coord + offset
            coord = coord_orig = coord * data['mesh_scale']

            coord = torch.from_numpy(coord)

            quat_list = [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
            rotation_matrix = Rotation.from_quat(quat_list)
            rotation_matrix = rotation_matrix.as_matrix()

            transform = np.eye(4)
            transform[:3, :3] = rotation_matrix
            transform[:3, -1] = pos

            quat_list = [float(quat2[0]), float(quat2[1]), float(quat2[2]), float(quat2[3])]

            rotation_matrix = Rotation.from_quat(quat_list)
            rotation_matrix = rotation_matrix.as_matrix()

            transform2 = np.eye(4)
            transform2[:3, :3] = rotation_matrix
            transform2[:3, -1] = pos2

            transform = torch.from_numpy(transform)
            transform2 = torch.from_numpy(transform2)

            dp_np_first = torch.cat([dp_np_first, torch.ones_like(dp_np_first[..., :1])], dim=-1)
            dp_np_second = torch.cat([dp_np_second, torch.ones_like(dp_np_second[..., :1])], dim=-1)

            coord = torch.cat([coord, torch.ones_like(coord[..., :1])], dim=-1)
            coord = torch.sum(transform[None, :, :] * coord[:, None, :], dim=-1)
            coord = coord[..., :3]

            point_transform = torch.matmul(transform, torch.inverse(transform2))
            dp_np_second = torch.sum(point_transform[None, :, :] * dp_np_second[:, None, :], dim=-1)

            depth_coords = torch.cat([dp_np_first[..., :3], dp_np_second[..., :3]], dim=0)

            rix = torch.randperm(depth_coords.size(0))
            depth_coords = depth_coords[rix[:1000]]

            if depth_coords.size(0) != 1000:
                return self.get_item(index=random.randint(0, self.__len__() - 1))

            label = (label - 0.5) * 2.0
            # center = (depth_coords.min(dim=0)[0] + depth_coords.max(dim=0)[0]) / 2.
            center = depth_coords.mean(dim=0)
            coord = coord - center[None, :]
            depth_coords = depth_coords - center[None, :]

            all_coords = coord
            labels = label

            if 'point_cloud' in data.files:
                gt_pointcloud = data['point_cloud']
                if gt_pointcloud.shape[0] >= 1000:
                    gt_pointcloud = torch.from_numpy(gt_pointcloud)
                    gt_pcd_idx = torch.randperm(gt_pointcloud.size(0))
                    gt_pointcloud = gt_pointcloud[gt_pcd_idx[:1000]]
                else:
                    gt_pointcloud = torch.rand(1000, 3)
            else:
                gt_pointcloud = torch.rand(1000, 3)

            res = {'depth_coords': depth_coords.float(),
                   'coords': coord.float(),
                   'intrinsics':intrinsics.float(),
                   'pointcloud': gt_pointcloud.float(),
                   'cam_poses': np.asarray(cam_poses)}
            return res, {'occ': torch.from_numpy(labels).float()}

        except Exception as e:
            print(e)
            # print(file)
            return self.get_item(index=random.randint(0, self.__len__() - 1))

    def __getitem__(self, index):
        return self.get_item(index)

class BowlOccTrainDataset(Dataset):

    def __init__(self, sidelength, phase='train'):
        # Path setup
        self.root = "/data/scratch/asimeonov/repos/research/PIFU_robot/data_gen/data/bowl_table_upright_pose_4_cam_half_occ_full_rand_scale"
        self.files = glob.glob(self.root+"/*.npz")
        self.files = sorted(self.files)

        self.sidelength = sidelength

        block = 64
        bs = 1 / block
        hbs = bs * 0.5
        y, z, x = np.meshgrid(np.linspace(-.5+hbs, .5-hbs, block), np.linspace(-.5+hbs, .5-hbs, block), np.linspace(-.5+hbs, .5-hbs, block))
        self.bs = bs
        self.hbs = hbs
        # voxel = np.stack([z, x, -y], axis=-1)
        # self.voxel = voxel.reshape((-1, 3))

        self.shapenet_dict = pickle.load(open("shapenet_bowl.p", "rb"))
        self.projection_mode = "perspective"

        n = len(self.files)
        idx = int(0.9 * n)

        if phase == 'train':
            self.files = self.files[:idx]
        else:
            self.files = self.files[idx:]

        self.cache_file = None
        self.count = 0

        print("files length ", len(self.files))

    def __len__(self):
        return len(self.files)

    def get_item(self, index):
        try:
            data = np.load(self.files[index], allow_pickle=True)
            posecam =  data['object_pose_cam_frame']
            pos = posecam[0, :3]
            quat = posecam[0, 3:]

            pos2 = posecam[1, :3]
            quat2 = posecam[1, 3:]

            shapenet_id = str(data['shapenet_id'].item())
            category_id = str(data['shapenet_category_id'].item())
            vertex_offset = data['vertex_offset']

            depths = []
            segs = []
            cam_poses = []
            for i in range(2):
                seg = data['object_segmentation'][i, 0]
                depth = data['depth_observation'][i]
                segs.append(seg)
                depths.append(torch.from_numpy(depth))
                cam_poses.append(data['cam_pose_world'][i])

            y, x = torch.meshgrid(torch.arange(480), torch.arange(640))

            # Compute native intrinsic matrix
            sensor_half_width = 320
            sensor_half_height = 240

            # hor_fov = 60 * np.pi / 180
            vert_fov = 60 * np.pi / 180

            vert_f = sensor_half_height / np.tan(vert_fov / 2)
            hor_f = sensor_half_width / (np.tan(vert_fov / 2) * 320 / 240)

            intrinsics = np.array(
                [[hor_f, 0., sensor_half_width, 0.],
                 [0., vert_f, sensor_half_height, 0.],
                 [0., 0., 1., 0.]]
            )

            # Rescale to new sidelength
            # intrinsics[:2, :3] *= np.array([128/640, 128/480])[:, None]
            intrinsics = torch.from_numpy(intrinsics)

            seg_mask = segs[0]
            dp_np_first = geometry.lift(x.flatten()[seg_mask], y.flatten()[seg_mask], depths[0].flatten(), intrinsics[None, :, :])

            seg_mask = segs[1]
            dp_np_second = geometry.lift(x.flatten()[seg_mask], y.flatten()[seg_mask], depths[1].flatten(), intrinsics[None, :, :])

            shapenet_path = "/data/vision/billf/object-properties/dataset/billf-6/ShapeNetCore.v2/"

            voxel_path = osp.join(shapenet_path, category_id, shapenet_id, 'models', 'model_normalized_128.mat')
            coord, voxel_bool, voxel_pos = self.shapenet_dict[voxel_path]

            rix = np.random.permutation(coord.shape[0])

            coord = coord[rix[:1500]]
            label = voxel_bool[rix[:1500]]

            offset = np.random.uniform(-self.hbs, self.hbs, coord.shape)
            coord = coord + offset
            coord = coord_orig = coord * data['mesh_scale']

            coord = torch.from_numpy(coord)

            quat_list = [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
            rotation_matrix = Rotation.from_quat(quat_list)
            rotation_matrix = rotation_matrix.as_matrix()

            transform = np.eye(4)
            transform[:3, :3] = rotation_matrix
            transform[:3, -1] = pos

            quat_list = [float(quat2[0]), float(quat2[1]), float(quat2[2]), float(quat2[3])]

            rotation_matrix = Rotation.from_quat(quat_list)
            rotation_matrix = rotation_matrix.as_matrix()

            transform2 = np.eye(4)
            transform2[:3, :3] = rotation_matrix
            transform2[:3, -1] = pos2

            transform = torch.from_numpy(transform)
            transform2 = torch.from_numpy(transform2)

            dp_np_first = torch.cat([dp_np_first, torch.ones_like(dp_np_first[..., :1])], dim=-1)
            dp_np_second = torch.cat([dp_np_second, torch.ones_like(dp_np_second[..., :1])], dim=-1)

            coord = torch.cat([coord, torch.ones_like(coord[..., :1])], dim=-1)
            coord = torch.sum(transform[None, :, :] * coord[:, None, :], dim=-1)
            coord = coord[..., :3]

            point_transform = torch.matmul(transform, torch.inverse(transform2))
            dp_np_second = torch.sum(point_transform[None, :, :] * dp_np_second[:, None, :], dim=-1)

            depth_coords = torch.cat([dp_np_first[..., :3], dp_np_second[..., :3]], dim=0)

            rix = torch.randperm(depth_coords.size(0))
            depth_coords = depth_coords[rix[:1000]]

            if depth_coords.size(0) != 1000:
                return self.get_item(index=random.randint(0, self.__len__() - 1))

            label = (label - 0.5) * 2.0
            # center = (depth_coords.min(dim=0)[0] + depth_coords.max(dim=0)[0]) / 2.
            center = depth_coords.mean(dim=0)
            coord = coord - center[None, :]
            depth_coords = depth_coords - center[None, :]

            all_coords = coord
            labels = label

            if 'point_cloud' in data.files:
                gt_pointcloud = data['point_cloud']
                if gt_pointcloud.shape[0] >= 1000:
                    gt_pointcloud = torch.from_numpy(gt_pointcloud)
                    gt_pcd_idx = torch.randperm(gt_pointcloud.size(0))
                    gt_pointcloud = gt_pointcloud[gt_pcd_idx[:1000]]
                else:
                    gt_pointcloud = torch.rand(1000, 3)
            else:
                gt_pointcloud = torch.rand(1000, 3)

            res = {'depth_coords': depth_coords.float(),
                   'coords': coord.float(),
                   'intrinsics':intrinsics.float(),
                   'pointcloud': gt_pointcloud.float(),
                   'cam_poses': np.asarray(cam_poses)}
            return res, {'occ': torch.from_numpy(labels).float()}

        except Exception as e:
            print(e)
            # print(file)
            return self.get_item(index=random.randint(0, self.__len__() - 1))

    def __getitem__(self, index):
        return self.get_item(index)


class BottleOccTrainDataset(Dataset):

    def __init__(self, sidelength, phase='train'):

        # Path setup
        self.root = "/data/scratch/asimeonov/repos/research/PIFU_robot/data_gen/data/bottle_table_upright_pose_4_cam_half_occ_full_rand_scale"
        self.files = glob.glob(self.root+"/*.npz")
        self.files = sorted(self.files)

        self.sidelength = sidelength

        block = 64
        bs = 1 / block
        hbs = bs * 0.5
        y, z, x = np.meshgrid(np.linspace(-.5+hbs, .5-hbs, block), np.linspace(-.5+hbs, .5-hbs, block), np.linspace(-.5+hbs, .5-hbs, block))
        self.bs = bs
        self.hbs = hbs

        self.shapenet_dict = pickle.load(open("shapenet_bottle.p", "rb"))
        self.projection_mode = "perspective"

        n = len(self.files)
        idx = int(0.9 * n)

        if phase == 'train':
            self.files = self.files[:idx]
        else:
            self.files = self.files[idx:]

        self.cache_file = None
        self.count = 0

        print("files length ", len(self.files))

    def __len__(self):
        return len(self.files)

    def get_item(self, index):
        # try:
        data = np.load(self.files[index], allow_pickle=True)
        posecam =  data['object_pose_cam_frame']
        pos = posecam[0, :3]
        quat = posecam[0, 3:]

        pos2 = posecam[1, :3]
        quat2 = posecam[1, 3:]

        shapenet_id = str(data['shapenet_id'].item())
        category_id = str(data['shapenet_category_id'].item())
        vertex_offset = data['vertex_offset']

        depths = []
        segs = []
        cam_poses = []
        for i in range(2):
            seg = data['object_segmentation'][i, 0]
            depth = data['depth_observation'][i]
            segs.append(seg)
            depths.append(torch.from_numpy(depth))
            cam_poses.append(data['cam_pose_world'][i])

        y, x = torch.meshgrid(torch.arange(480), torch.arange(640))

        # Compute native intrinsic matrix
        sensor_half_width = 320
        sensor_half_height = 240

        # hor_fov = 60 * np.pi / 180
        vert_fov = 60 * np.pi / 180

        vert_f = sensor_half_height / np.tan(vert_fov / 2)
        hor_f = sensor_half_width / (np.tan(vert_fov / 2) * 320 / 240)

        intrinsics = np.array(
            [[hor_f, 0., sensor_half_width, 0.],
             [0., vert_f, sensor_half_height, 0.],
             [0., 0., 1., 0.]]
        )

        # Rescale to new sidelength
        # intrinsics[:2, :3] *= np.array([128/640, 128/480])[:, None]
        intrinsics = torch.from_numpy(intrinsics)

        seg_mask = segs[0]
        dp_np_first = geometry.lift(x.flatten()[seg_mask], y.flatten()[seg_mask], depths[0].flatten(), intrinsics[None, :, :])

        seg_mask = segs[1]
        dp_np_second = geometry.lift(x.flatten()[seg_mask], y.flatten()[seg_mask], depths[1].flatten(), intrinsics[None, :, :])

        shapenet_path = "/data/vision/billf/object-properties/dataset/billf-6/ShapeNetCore.v2/"

        voxel_path = osp.join(shapenet_path, category_id, shapenet_id, 'models', 'model_normalized_128.mat')
        coord, voxel_bool, voxel_pos = self.shapenet_dict[voxel_path]

        rix = np.random.permutation(coord.shape[0])

        coord = coord[rix[:1500]]
        label = voxel_bool[rix[:1500]]

        offset = np.random.uniform(-self.hbs, self.hbs, coord.shape)
        coord = coord + offset
        coord = coord_orig = coord * data['mesh_scale']

        coord = torch.from_numpy(coord)

        quat_list = [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
        rotation_matrix = Rotation.from_quat(quat_list)
        rotation_matrix = rotation_matrix.as_matrix()

        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, -1] = pos

        quat_list = [float(quat2[0]), float(quat2[1]), float(quat2[2]), float(quat2[3])]

        rotation_matrix = Rotation.from_quat(quat_list)
        rotation_matrix = rotation_matrix.as_matrix()

        transform2 = np.eye(4)
        transform2[:3, :3] = rotation_matrix
        transform2[:3, -1] = pos2

        transform = torch.from_numpy(transform)
        transform2 = torch.from_numpy(transform2)

        dp_np_first = torch.cat([dp_np_first, torch.ones_like(dp_np_first[..., :1])], dim=-1)
        dp_np_second = torch.cat([dp_np_second, torch.ones_like(dp_np_second[..., :1])], dim=-1)

        coord = torch.cat([coord, torch.ones_like(coord[..., :1])], dim=-1)
        coord = torch.sum(transform[None, :, :] * coord[:, None, :], dim=-1)
        coord = coord[..., :3]

        point_transform = torch.matmul(transform, torch.inverse(transform2))
        dp_np_second = torch.sum(point_transform[None, :, :] * dp_np_second[:, None, :], dim=-1)

        depth_coords = torch.cat([dp_np_first[..., :3], dp_np_second[..., :3]], dim=0)

        rix = torch.randperm(depth_coords.size(0))
        depth_coords = depth_coords[rix[:1000]]

        if depth_coords.size(0) != 1000:
            return self.get_item(index=random.randint(0, self.__len__() - 1))

        label = (label - 0.5) * 2.0
        # center = (depth_coords.min(dim=0)[0] + depth_coords.max(dim=0)[0]) / 2.
        center = depth_coords.mean(dim=0)
        coord = coord - center[None, :]
        depth_coords = depth_coords - center[None, :]

        all_coords = coord
        labels = label

        if 'point_cloud' in data.files:
            gt_pointcloud = data['point_cloud']
            if gt_pointcloud.shape[0] >= 1000:
                gt_pointcloud = torch.from_numpy(gt_pointcloud)
                gt_pcd_idx = torch.randperm(gt_pointcloud.size(0))
                gt_pointcloud = gt_pointcloud[gt_pcd_idx[:1000]]
            else:
                gt_pointcloud = torch.rand(1000, 3)
        else:
            gt_pointcloud = torch.rand(1000, 3)

        res = {'depth_coords': depth_coords.float(),
               'coords': coord.float(),
               'intrinsics':intrinsics.float(),
               'pointcloud': gt_pointcloud.float(),
               'cam_poses': np.asarray(cam_poses)}
        return res, {'occ': torch.from_numpy(labels).float()}

        # except Exception as e:
        #     print(e)
        #     # print(file)
        #     return self.get_item(index=random.randint(0, self.__len__() - 1))

    def __getitem__(self, index):
        return self.get_item(index)

class JointOccTrainDataset(Dataset):

    def __init__(self, sidelength, depth_aug=False, multiview_aug=False, phase='train'):

        # Path setup
        # self.root = "/data/scratch/asimeonov/repos/research/PIFU_robot/data_gen/data/mug_table_upright_pose_4_cam_half_occ_full_rand_scale"
        mug_path = "/media/jiahui/JIAHUI/obj_data/sim/train"
        # bottle_path = "/data/scratch/asimeonov/repos/research/PIFU_robot/data_gen/data/bottle_table_all_pose_4_cam_half_occ_full_rand_scale"
        # bowl_path = "/scratch/anthony/repos/research/PIFU_robot/data_gen/data/bowl_table_all_pose_4_cam_half_occ_full_rand_scale"
        paths = [mug_path]#, bottle_path, bowl_path]

        files_total = []
        for path in paths:
            files = list(sorted(glob.glob(path+"/*.npz")))
            n = len(files)
            idx = int(0.9 * n)

            if phase == 'train':
                files = files[:idx]
            else:
                files = files[idx:]

            files_total.extend(files)

        self.files = files_total

        self.sidelength = sidelength
        self.depth_aug = depth_aug
        self.multiview_aug = multiview_aug

        block = 64
        bs = 1 / block
        hbs = bs * 0.5
        y, z, x = np.meshgrid(np.linspace(-.5+hbs, .5-hbs, block), np.linspace(-.5+hbs, .5-hbs, block), np.linspace(-.5+hbs, .5-hbs, block))
        self.bs = bs
        self.hbs = hbs

        shapenet_files_path = "/home/jiahui/shape_occupancy/shapenet_files/"
        self.shapenet_mug_dict = pickle.load(open(shapenet_files_path+"shapenet_mug.p", "rb"))
        # print("shape dictionary",self.shapenet_mug_dict)
        # self.shapenet_bowl_dict = pickle.load(open("shapenet_bowl.p", "rb"))
        # self.shapenet_bottle_dict = pickle.load(open("shapenet_bottle.p", "rb"))

        self.shapenet_dict = {'03797390': self.shapenet_mug_dict}#, '02880940': self.shapenet_bowl_dict, '02876657': self.shapenet_bottle_dict}

        self.projection_mode = "perspective"

        self.cache_file = None
        self.count = 0

        print("files length ", len(self.files))

    def __len__(self):
        return len(self.files)

    def get_item(self, index):
        try:
            data = np.load(self.files[index], allow_pickle=True)
            depth = data["point_cloud"]
            # print("depth",depth.shape)
            if depth.shape[0] <1000: # choose another one
                return self.get_item(index=random.randint(0, self.__len__() - 1))

            # 1000 random sampled point clouds from object
            rix_pcd = np.random.permutation(depth.shape[0])
            depth_pointcloud = depth[rix_pcd[:1000]]
            # print("depth pointcloud",rix_pcd.max())
            if depth_pointcloud.shape[0] != 1000: # choose another one
                # print("choose another one!")
                return self.get_item(index=random.randint(0, self.__len__() - 1))

            shapenet_id = str(data['shapenet_id'].item())
            category_id = str(data['shapenet_category_id'].item())
            obj_pose = data["object_pose_world"]
            pos = obj_pose[:3]
            quat = obj_pose[3:]
            quat_list = [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
            rotation_matrix = Rotation.from_quat(quat_list)
            rotation_matrix = rotation_matrix.as_matrix()

            intrinsics = data["cam_intrinsics"]
            intrinsics = torch.from_numpy(intrinsics)

            shapenet_path = "/data/vision/billf/object-properties/dataset/billf-6/ShapeNetCore.v2/"
            voxel_path = osp.join(shapenet_path, category_id, shapenet_id, 'models', 'model_normalized_128.mat')
            coord, voxel_bool, voxel_pos = self.shapenet_dict[category_id][voxel_path]
            # print("coord complte",coord.shape,voxel_bool.shape,voxel_pos.shape)

            mask = (voxel_bool == True)[:, 0]

            coord_pos = coord[mask]
            coord_neg = coord[~mask]
            label_pos = voxel_bool[mask]
            label_neg = voxel_bool[~mask]
            rix_pos = np.random.permutation(coord_pos.shape[0])
            rix_neg = np.random.permutation(coord_neg.shape[0])
            coord_pos = coord_pos[rix_pos[:750]]
            coord_neg = coord_neg[rix_neg[:750]]
            label_pos = label_pos[rix_pos[:750]]
            label_neg = label_neg[rix_neg[:750]]
            label = np.concatenate([label_pos, label_neg], axis=0)
            coord = np.concatenate([coord_pos, coord_neg], axis=0)

            # offset = np.random.uniform(-self.hbs, self.hbs, coord.shape)
            coord = coord #+ offset
            mesh_scale = 0.5 #data['mesh_scale']
            coord = coord_orig = coord * mesh_scale
            coord = np.matmul(rotation_matrix,coord.T).T+pos.reshape((-1,3))

            label = (label - 0.5) * 2.0

            center = np.mean(depth_pointcloud,axis=0)
            # print("center:",center)
            coord = coord # - center[None, :]
            depth_pointcloud = depth_pointcloud# -center

            # #viz
            # fig = plt.figure()
            # ax = fig.add_subplot(projection='3d')
            # # For each set of style and range settings, plot n random points in the box
            # # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
            # ax.scatter(depth_pointcloud[:,0], depth_pointcloud[:,1], depth_pointcloud[:,2], marker='o')
            # ax.scatter(coord[:,0], coord[:,1], coord[:,2], marker='^')
            # ax.set_xlabel('X Label')
            # ax.set_ylabel('Y Label')
            # ax.set_zlabel('Z Label')
            #
            # plt.show()


            labels = label
            gt_pointcloud = torch.rand(1000, 3)

            res = {'depth_coords': torch.from_numpy(depth_pointcloud).float(),
                   'coords': torch.from_numpy(coord).float(),
                   'intrinsics':intrinsics.float(),
                   'pointcloud': gt_pointcloud.float(),
                   'cam_poses': np.zeros(1)}
            # print("depth+coord",depth_coords,depth_coords.shape)
            # print("coords",coord,coord.shape)
            # print("intriincs",intrinsics)
            return res, {'occ': torch.from_numpy(labels).float()}

        except Exception as e:
            # print("errorr!")
            print(e)
            # print(file)
            return self.get_item(index=random.randint(0, self.__len__() - 1))

    def __getitem__(self, index):
        return self.get_item(index)

if __name__ == "__main__":
    dataset = RepBCTrainDataset(128)
    dataset[0]
