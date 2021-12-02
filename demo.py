import vnn_occupancy_network
import torch
from glob import glob
import numpy as np
import geometry
from scipy.spatial.transform import Rotation
import open3d as o3d
from matplotlib import pyplot as plt

def load_pointcloud(data):
    posecam =  data['object_pose_cam_frame']
    idxs = list(range(posecam.shape[0]))

    segs = []
    depths = []

    poses = []
    quats = []
    # Load the camera positions of each camera
    for i in idxs:
        pos = posecam[i, :3]
        quat = posecam[i, 3:]

        poses.append(pos)
        quats.append(quat)

    # Process and obtain the depth map / segmentation of each object
    for i in idxs:
        seg = data['object_segmentation'][i, 0]
        depth = data['depth_observation'][i]

        rix = np.random.permutation(depth.shape[0])[:1000]
        seg = seg[rix]
        depth = depth[rix]

        segs.append(seg)
        depths.append(torch.from_numpy(depth))

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

    intrinsics = torch.from_numpy(intrinsics)

    y, x = torch.meshgrid(torch.arange(480), torch.arange(640))

    # Compute to camera coordinates each segmentation mask
    dp_nps = []
    for i in range(len(segs)):
        seg_mask = segs[i]
        dp_np = geometry.lift(x.flatten()[seg_mask], y.flatten()[seg_mask], depths[i].flatten(), intrinsics[None, :, :])
        dp_np = torch.cat([dp_np, torch.ones_like(dp_np[..., :1])], dim=-1)
        dp_nps.append(dp_np)

    transforms = []
    # Create transform for each camera
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

    dp_np_extra = []

    # Convert each depth camera to camera coordinates
    for i, dp_np in enumerate(dp_nps):
        point_transform = torch.matmul(transform, torch.inverse(transforms[i]))
        dp_np = torch.sum(point_transform[None, :, :] * dp_np[:, None, :], dim=-1)
        dp_np_extra.append(dp_np[..., :3])

    depth_coords = torch.cat(dp_np_extra, dim=0)

    rix = torch.randperm(depth_coords.size(0))
    depth_coords = depth_coords[rix[:1000]]

    center = (depth_coords.min(dim=0)[0] + depth_coords.max(dim=0)[0]) / 2.
    # Center depth coordinates
    depth_coords = depth_coords - center[None, :]

    return depth_coords

def load_pcd(path):
    pcd = o3d.io.read_point_cloud(path)
    points = np.array(pcd.points)
    max = np.max(points,axis=0)
    min = np.min(points,axis=0)
    # print("min,max",min.shape,max.shape)
    scale_min = np.array([-0.1148,-0.1082,-0.1485])
    scale_max = (-scale_min).copy()
    ratio = (max-min)/(scale_max-scale_min)
    points_scaled = (points-min)/ratio+scale_min
    o3d.visualization.draw_geometries([pcd])
    # print("points_scaled",points_scaled.min(axis=0),points_scaled.max(axis=0))
    return points_scaled

def viz_latent(latent):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(latent)
    o3d.visualization.draw_geometries([pcd])
    min = latent.min(axis=0)
    max = latent.max(axis=0)
    ratio = (max-min)/255.
    latent_rescaled = (latent-min)/ratio
    plt.imshow(latent_rescaled, interpolation='nearest')
    plt.show()

if __name__ == "__main__":
    model = vnn_occupancy_network.VNNOccNet(latent_dim=256)#.cuda()
    ckpt = torch.load("model_current.pth")
    model.load_state_dict(ckpt)
   # model.cuda()

    # demo_path = "data.npz"
    # data = np.load(demo_path, allow_pickle=True)

    for index in range(1,8):
        # index
        pcd_path = "/home/jiahui/OBJ_SLAM"+"/"+str(index)+".pcd"
        data = torch.from_numpy(load_pcd(pcd_path),map_location=torch.device('cpu'))
    # create pointcloud from data
    pointcloud = data[None, :, :].float()#.cuda()
    # pointcloud = load_pointcloud(data)[None, :, :].float().cuda()
    # import pdb
    # pdb.set_trace()

    # Scale pointcloud when passing into model
    pointcloud = pointcloud * 10.
    latent = model.encoder(pointcloud).cpu().detach().numpy()
    latent_shape = (latent * latent).sum(dim=-1)
    viz_latent(latent[0,:])
    print("latent",latent.shape,latent[0,:].min(axis=0),latent[0,:].max(axis=0))
    # import pdb
    # pdb.set_trace()
    # print(latent)



    # example loading of the data
