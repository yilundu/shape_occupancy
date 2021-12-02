import vnn_occupancy_network
import torch
from glob import glob
import numpy as np
import geometry
from scipy.spatial.transform import Rotation
import open3d as o3d
import matplotlib.pyplot as plt

def load_data(folder,seed):
    data = np.load(folder+"all_matrix"+str(seed)+".npz",allow_pickle=True)
    print("data name",data.files)
    obj_pose = data["obj_pose"]
    cam_traj = data["cam_traj"]
    projectionMatrix = data["projectionMatrix"]
    return obj_pose,cam_traj,projectionMatrix

def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy")
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame’s origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

def point_cloud_viz(points):
    pc = o3d.geometry.PointCloud()
    # print("in viz",points,points.shape)
    pc.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pc])

def point_cloud_scaling(points):
    max = np.amax(points, axis=0)
    min = np.amin(points,axis=0)
    # print("min,max",min.shape,max.shape)
    scale_min = np.array([-0.1149, -0.1634, 0.9826])
    scale_max = np.array([0.1139, 0.0554, 1.2800])
    ratio = (max-min)/(scale_max-scale_min)
    points_scaled = (points-min)/ratio+scale_min
    return points_scaled

def get_point_cloud(obj_pose,cam_traj,projectionMatrix):
    # return a N x 1000 x 3 point cloud for each of the N objects in the current frame
    num_frame = cam_traj.shape[0]
    num_pt=1000
    # print("projection matirx",projectionMatrix,projectionMatrix.shape)
    point_cloud_total = []
    obj_pts = [ [] for i in range(obj_pose.shape[0])]
    for idx in range(num_frame):
        print("frame id",idx)
        frame_data = np.load(folder+"frame"+str(idx)+".npz")
        im=frame_data["im"]
        depth = frame_data["depth"]
        seg = frame_data["seg"]
        # plt.imshow(seg, interpolation='nearest')
        # plt.show()
        transform = cam_traj[idx,:]
        cam2world = np.linalg.inv(transform)
        im_w = im.shape[1]
        im_h = im.shape[0]
        focal = 0.5 * im_h / np.tan(0.5 * np.pi / 3)
        rays_o, rays_d = get_rays_np(im_h, im_w, focal, cam2world)
        pointcloud_frame = depth[:, :, None] * rays_d + rays_o
        for obj_idx in range(2,seg.max()+1):
            obj = obj_idx-2
            print("obj_idx",obj_idx,seg.max(),seg.min())
            obj_pixel = np.where(seg==obj_idx)
            if (len(obj_pixel[0])):
                obj_all = pointcloud_frame[obj_pixel[0],obj_pixel[1]].reshape((-1,3))
                # point_cloud_viz(obj_all)
                obj_pts[obj].append(obj_all)
    for obj in range(obj_pose.shape[0]):
        obj_body = np.concatenate(obj_pts[obj], axis=0)
        point_cloud_viz(obj_body)
        print("obj",obj_body.shape)
        id = np.random.choice(obj_body.shape[0],num_pt,replace=True)
        obj_pt_selected  = point_cloud_scaling(obj_body[id,:])
        # zero-centered
        center = (np.amax(obj_pt_selected,axis=0)+np.amin(obj_pt_selected,axis=0))/2.
        print("small large",obj_pt_selected.min(axis=0),obj_pt_selected.max(axis=0),center)
        print("after centered",obj_pt_selected.min(axis=0)-center,obj_pt_selected.max(axis=0)-center,center)
        point_cloud_viz(obj_pt_selected-center)
        point_cloud_total.append(obj_pt_selected-center)
    return point_cloud_total

if __name__ == "__main__":
    print("startS")
    model = vnn_occupancy_network.VNNOccNet(latent_dim=256)#.cuda()
    ckpt = torch.load("model_current.pth",map_location=torch.device('cpu'))
    model.load_state_dict(ckpt)
    #model.cuda()
    folder = "pic_4/"
    seed=12
    obj_pose,cam_traj,projectionMatrix=load_data(folder,seed)

    point_cloud_total = get_point_cloud(obj_pose,cam_traj,projectionMatrix)
    # print("poitcloud total",len(point_cloud_total))
    latent_total = []
    for idx in [0,4]:#range(cam_traj.shape[0]):
        point_cloud = point_cloud_total[idx]
        # print("point cloud current frame",point_cloud.shape)
        latent_all = np.zeros((len(point_cloud),256,3))
        for i in range(len(point_cloud)):
            obj_partial_pc = point_cloud[i]
            point_cloud_tmp = obj_partial_pc*10
            input = torch.from_numpy(point_cloud_tmp[None,:,:]).float()
            # print("INt",input.shape)
            latent = model.encoder(input)
            # print("latent",latent.shape)
            latent_all[i,:] = latent.squeeze().detach().numpy()
        latent_total.append(latent_all)

    # shape comparison
    num_obj = obj_pose.shape[0]
    duration =cam_traj.shape[0]
    shape_total=[]
    for obj in range(num_obj):
        latent = []
        encoding=[]
        shape=np.zeros((num_obj,256))
        for i in [0,1]:
            shape[i,:] = np.linalg.norm(latent_total[i][obj,:],axis=1)
            latent.append(latent_total[i][obj,:])#np.matmul(latent_total[i][obj,:],latent_total[i][obj,:].T))
            pc = o3d.geometry.PointCloud()
    # print("in viz",points,points.shape)
            pc.points = o3d.utility.Vector3dVector(latent_total[i][obj,:])
            encoding.append(pc)
        encoding[0].paint_uniform_color([1,0,0])
        encoding[1].paint_uniform_color([0,0,0])
        shape_total.append(shape)

    def get_distance(vec1,vec2):
        # print("haha",np.dot(vec1,vec2),np.linalg.norm(vec1),np.linalg.norm(vec2))
        # cos = np.dot(vec1,vec2)/(np.linalg.norm(vec1))/(np.linalg.norm(vec2))
        dis  = np.linalg.norm(vec1-vec2)
        return dis

    for i in range(2):
        for j in range(2):
            print("distance",get_distance(shape_total[0][i],shape_total[1][j]))
    # x = range(shape[0].shape[0])
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    #
    # ax1.scatter(x, shape_total[0][0], s=10, c='b', marker="s", label='first')
    # ax1.scatter(x,shape_total[1][0], s=10, c='r', marker="o", label='second')
    # plt.legend(loc='upper left')
    # plt.show()






        # print("encoding shape",shape_1-shape_0)
        # o3d.visualization.draw_geometries(encoding)
        # print(np.mean(latent[0]-latent[1])) #.linalg.norm(








