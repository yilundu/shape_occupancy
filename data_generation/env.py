import pybullet as p
from pybullet_utils import set_pose, load_pybullet, Pose, Point, stable_z, connect, Euler,get_pose,get_pose_distance,get_center_extent,get_aabb, pairwise_collision, remove_body
import numpy as np
import random
import os
from default_eval_cfg import get_eval_cfg_defaults
import open3d as o3d
from imageio import imwrite
from scipy.spatial.transform import Rotation
import pickle
import os.path as osp
import matplotlib.pyplot as plt
import torch

def get_z_offset(obj):
    (_, _, z_center) , _ = get_center_extent(obj)
    # Compute geometric center of mass of object
    (_, _, z_com), _ = get_pose(obj)
    offset = z_center - z_com
    p.changeVisualShape(obj, -1, rgbaColor=tuple((0,0,1)))
    return offset

def get_random_location(table):
    # get location on the table
    aa,bb = get_aabb(table)
    xpos,ypos,_ = aa+0.1*(bb-aa)+random.random()*(bb-aa)*0.8
    return xpos,ypos
    # orn = p.getQuaternionFromEuler([0, 0, angle])
    # urdf_path = os.path.join(self._urdfRoot, urdf_name)
    # uid = p.loadURDF(urdf_path, [xpos, ypos, .15], [orn[0], orn[1], orn[2], orn[3]])

def gen_traj(obj_pose,radius_dim, roll_dim, z_dim,duration=100):
    # time = np.arange(duration)#.reshape((-1,1))
    # print("time",time.shape)
    roll = np.random.uniform(low=0,high = 2.4,size=roll_dim)*np.pi
    roll = np.sort(roll)
    # print("rooll",roll/np.pi)
    # radius_sigma = 0.02
    # radius = 2+np.random.normal(0,radius_sigma,duration)
    radius = np.random.uniform(low=0.8, high=4.8, size=radius_dim)
    radius = np.sort(radius)
    z_offset = np.random.uniform(-0.8,1.2,z_dim)
    z = obj_pose[2] + z_offset
    z = np.sort(z)
    mesh = np.vstack(np.meshgrid(radius,roll,z)).reshape(3,-1).T
    # print("base")
    # print("radius",np.sort(radius))
    # print("roll",np.sort(roll))
    # print("z",np.sort(z))
    # add offset
    # mu = 0
    # sigma_x = 0.02
    # sigma_y = 0.02
    roll_x_offset = 0#np.random.normal(mu, sigma_x, roll_dim*radius_dim*z_dim)#*2*np.pi
    roll_y_offset = 0#np.random.normal(mu, sigma_y, roll_dim*radius_dim*z_dim)#*2*np.pi
    # add noise
    mu = 0
    sigma_x = 0.025
    sigma_y = 0.025
    sigma_z = 0.025
    x_eps = np.random.normal(mu, sigma_x, z_dim*roll_dim*radius_dim)#.reshape((-1,1))
    y_eps = np.random.normal(mu, sigma_y, z_dim*roll_dim*radius_dim)#.reshape((-1,1))
    z_eps = np.random.normal(mu,sigma_z,roll_dim*radius_dim*z_dim)

    location = np.zeros((radius_dim*roll_dim*z_dim,3))
    # print("location",location.shape,roll_x_offset.shape, mesh[:,1].shape,x_eps.shape)
    location[:,0] = mesh[:,0]*np.cos(mesh[:,1]+roll_x_offset)+x_eps+obj_pose[0]
    location[:,1] = mesh[:,0]*np.sin(mesh[:,1]+roll_y_offset)+y_eps+obj_pose[1]
    # print("x_unique",np.unique(np.cos(mesh[:,1])))
    # print("y_unique",np.unique(np.sin(mesh[:,1])))
    location[:,2]= mesh[:,2]+z_eps

    # # add noise
    #
    # print("x_eps",x_eps.shape)
    # x = radius*np.cos(roll+x_offset)+x_eps
    # y = radius*np.sin(roll+y_offset)+y_eps
    #
    # # x = np.convolve(x,np.ones(4),mode = 'same').reshape((-1,1))
    # # y = np.convolve(y,np.ones(4),mode = 'same').reshape((-1,1))
    # # sigma_z=0.03
    # print("table z",table_z)
    # #1.2*np.ones(x.shape) #np.random.rand(duration,1)+1.2
    # print("z",z)
    # # x = x.reshape((-1,1))
    # # y=y.reshape((-1,1))
    # # z = z.reshape((-1,1))
    # location = np.vstack(np.meshgrid(x,y,z)).reshape(3,-1).T
    # print("location",location,x.shape,z.shape,location.shape)
    return location

def load_obj(type):
    cfg = get_eval_cfg_defaults()
    train_list = cfg.MUG.TRAIN_SHAPENET_IDS
    bad_list = cfg.MUG.AVOID_SHAPENET_IDS
    test_list = cfg.MUG.TEST_SHAPENET_IDS
    if (type=="train"):
        obj_list = [ id for id in train_list if id not in bad_list]
    else:
        obj_list = [ id for id in test_list if id not in bad_list]
    # objects_raw = "/home/jiahui/shape_occupancy/airobot/src/eof_robot/src/eof_robot/descriptions/objects/mug_centered_obj"#os.listdir(osp.join(shapenet_data_dir, obj_class, SHAPENET_ID_DICT[obj_class]))
    # objects_filtered = [fn for fn in objects_raw if fn.split('/')[-1] not in bad_shapenet_ids_list]
    # total_filtered = len(objects_filtered)
    # print("totla filtered",len(obj_list))
    return obj_list
    # print("totla filtered",len(list),len(train_list))

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
    # while 1:
    #     plane_model, inliers = pc.segment_plane(distance_threshold=0.01,
    #                                          ransac_n=3,
    #                                          num_iterations=1000)
    #     [a, b, c, d] = plane_model
    #     print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    #     inlier_cloud = pc.select_by_index(inliers)
    #     inlier_cloud.paint_uniform_color([0,1, 0])
    #     outlier_cloud = pc.select_by_index(inliers, invert=True)
    #     o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    #     pc = outlier_cloud

def get_point_cloud_per_frame(num_obj,transform,depth,im,seg):
    # return a N x 1000 x 3 point cloud for each of the N objects in the current frame
    # num_pt=1000
    # print("inside")
    cam2world = np.linalg.inv(transform)
    im_w = im.shape[1]
    im_h = im.shape[0]
    focal = 0.5 * im_h / np.tan(0.5 * np.pi / 3)
    rays_o, rays_d = get_rays_np(im_h, im_w, focal, cam2world)
    # print("ray o d",rays_o.shape,rays_d.shape)
    pointcloud_frame = depth[:, :, None] * rays_d + rays_o
    # point_cloud_viz(pointcloud_frame.reshape((-1,3)))

    max_obj_idx = seg.max()
    # print("in max",max_obj_idx)
    point_cloud_total=[]
    # obj_frame=  np.zeros((num_obj,num_pt,3))
    for obj in range(1,3):#,max_obj_idx+1):
        obj_pixel = np.where(seg==obj)
        if(len(obj_pixel[0])<=0):
            return
        # print("obj_pixel",obj_pixel,len(obj_pixel),len(obj_pixel[0]),type(obj_pixel))
        obj_all = pointcloud_frame[obj_pixel[0],obj_pixel[1]].reshape((-1,3))
        # point_cloud_viz(obj_all.reshape((-1,3)))
        # print("obj_all",obj_all.shape)
        # id = np.random.choice(obj_all.shape[0],num_pt,replace=True)
        point_cloud_total.append(obj_all)
    # print("total pcd per frame",len(point_cloud_total))
    return point_cloud_total

def apply_noise_to_depth(depth, seg, obj_id, std, rate=0.00025):
        s = depth.shape
        flat_depth = depth.flatten()
        flat_seg = seg.flatten()
        obj_inds = np.where(flat_seg == obj_id)
        obj_depth = flat_depth[obj_inds[0]]
        eps = 0.0001

        new_depth = []
        for i in range(100):
            start, end = i*int(len(obj_depth)/100), (i+1)*int(len(obj_depth)/100)
            depth_window = obj_depth[start:end]
            std_dev = max(std + rate*np.mean(depth_window)**2, eps)
            noise_sample = np.random.normal(
                loc=0,
                scale=std_dev,
                size=depth_window.shape)
            new_depth_window = depth_window + noise_sample
            new_depth.append(new_depth_window)

        new_depth = np.asarray(new_depth).flatten()
        flat_depth[obj_inds[0][:len(new_depth)]] = new_depth

        return flat_depth.reshape(s)
if __name__ == "__main__":
    # start physics server
    # connect()
    seed=12
    # folder = 4
    random.seed(seed)
    np.random.seed(seed)

    root = "home/jiahui/shape_occupancy/"
    path = root +str(seed)#"pic_"+str(folder)+"obj"+str(seed)+".txt"

    # root = "/home/jiahui/shape_occupancy/"
    # path = root +"pic_"+str(folder)+"obj"+str(seed)+".txt"

    if(os.path.exists(path)):
        os.remove(path)
    # file_obj = open(path,"w")
    pClient = p.connect(p.DIRECT)#p.GUI)#
    p.setGravity(0,0,0)
    type = "train"
    obj_list = load_obj(type)
    # Load floor
    floor = p.loadURDF("short_floor.urdf", useFixedBase=1)#,basePosition=[0,0,0])
    set_pose(floor, Pose(Point(x=0, y=0, z=0),Euler(roll=0)))
    # file_obj.writelines(str(get_pose(floor)))
    # file_obj.writelines("\n")
    # load table
    table = p.loadURDF("19203/19203.urdf", useFixedBase=1)
    set_pose(table, Pose(Point(x=0, y=0, z=stable_z(table, floor)),Euler(roll=0)))

    # Compute native intrinsic matrix
    sensor_half_width = 240
    sensor_half_height = 240

    # hor_fov = 60 * np.pi / 180
    vert_fov = 60 * np.pi / 180

    vert_f = sensor_half_height / np.tan(vert_fov / 2)
    hor_f = sensor_half_width / (np.tan(vert_fov / 2))

    cam_intrinsics = np.array(
        [[hor_f, 0., sensor_half_width, 0.],
         [0., vert_f, sensor_half_height, 0.],
         [0., 0., 1., 0.]]
        )

    len_list = len(obj_list)
    mesh_scale=0.5
    ratio = 0.1 # subset
    print("ratio",int(ratio*len_list),len_list,obj_list[0])
    for mug_id in [obj_list[0]]:
        mug_friend = np.random.randint(0,len_list,size=np.array(1))
        # print("mug",mug_id,mug_friend)
        obj_table_list = [mug_id,obj_list[mug_friend[0]]]#,"1ea9ea99ac8ed233bf355ac8109b9988_model_128_df"]#,"3c0467f96e26b8c6a93445a1757adf6_model_128_df_dec","6faf1f04bde838e477f883dde7397db2_model_128_df_dec"]
        mugs = []
        obj_pose = np.zeros((len(obj_table_list),7))
        for k in range(len(obj_table_list)):
            obj = obj_table_list[k]
            while True:
                mug = load_pybullet("/home/jiahui/shape_occupancy/model/mug_centered_obj/"+obj+"_model_128_df.obj", scale=mesh_scale)
                x,y = get_random_location(table)
                yaw = np.pi / 2 + np.pi * random.random()
                set_pose(mug, Pose(Point(x=x, y=y, z=stable_z(mug,table)), Euler(roll=np.pi/2,yaw=yaw)))  #critical, can't be removed
                offset = get_z_offset(mug)
                set_pose(mug, Pose(Point(x=x, y=y, z=(stable_z(mug,table)-offset)),Euler(roll=np.pi/2,yaw=yaw)))
                collision = False
                for mug_i in mugs:
                    if pairwise_collision(mug_i, mug):
                        collision = True
                        break
                if collision:
                    remove_body(mug)
                else:
                    mugs.append(mug)
                    break
            obj_pose[k,:3] = np.array(get_pose(mugs[k])[0])
            obj_pose[k,3:] = np.array(get_pose(mugs[k])[1])
        # print("Mugs",mugs)
        # Create camera to view the scene
        near = 0.2
        far = 5
        projectionMatrix = p.computeProjectionMatrixFOV(60., 1.0, near, far)
        radius_dim= 8
        roll_dim=10
        z_dim= 8
        cam_location = gen_traj(obj_pose[0,0:3],radius_dim,roll_dim,z_dim)
        save_path= "/media/jiahui/JIAHUI/obj_data/sim/"+type+"/"

        for i in range(cam_location.shape[0]):
            location = cam_location[i,:]
            viewMatrix = p.computeViewMatrix(location,[0,0,1], [0, 0, 1])
            _, _, im, depth, seg = p.getCameraImage(width=480, height=480, viewMatrix=viewMatrix, projectionMatrix=projectionMatrix)
            depth = far * near / (far  - (far - near) * depth)
            # depth_noise = apply_noise_to_depth(depth)
            # print("error depth",np.mean(depth-depth_noise))
            transform = np.array(viewMatrix).reshape((4, 4)).transpose()
            # print("transform",transform)
            pcd = get_point_cloud_per_frame(len(obj_table_list),transform,depth,im,seg)
            if (pcd==None):
                # print("noting!")
                p.stepSimulation()
                continue
            p.stepSimulation()
            # pose = obj_pose[0,:]
            # pos = pose[:3]
            # quat = pose[3:]
            # quat_list = [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
            # rotation_matrix = Rotation.from_quat(quat_list)
            # rotation_matrix = rotation_matrix.as_matrix()
            #
            # transform = np.eye(4)
            # transform[:3, :3] = rotation_matrix
            # transform[:3, -1] = pos
            #
            # shapenet_files_path = "/home/jiahui/shape_occupancy/shapenet_files/"
            # shapenet_dict = pickle.load(open(shapenet_files_path+"shapenet_mug.p", "rb"))
            #
            # category_id= str('03797390')
            # shapenet_id = str(mug_id)
            # shapenet_path = "/data/vision/billf/object-properties/dataset/billf-6/ShapeNetCore.v2/"
            # voxel_path = osp.join(shapenet_path, category_id, shapenet_id, 'models', 'model_normalized_128.mat')
            # # print("shapet dict",shapenet_dict)
            # coord, voxel_bool, voxel_pos = shapenet_dict[voxel_path]
            # rix_model = np.random.permutation(coord.shape[0])
            # # print("coord complte",coord.shape,voxel_bool.shape,voxel_pos.shape)
            # coord = coord[np.where(voxel_bool==True)[0]]
            # coord = coord_orig = coord * mesh_scale
            # # print("coord shape",coord.shape)
            # coord = torch.from_numpy(coord)
            # coord = torch.cat([coord, torch.ones_like(coord[..., :1])], dim=-1)
            # # print("homo,",coord.shape)
            # coord = torch.sum(torch.from_numpy(transform)[None, :, :] * coord[:, None, :], dim=-1)
            # coord = coord[..., :3]
            # # coord = torch.matmul(torch.from_numpy(transform),torch.transpose(coord,0,1))
            # # coord = torch.transpose(coord,0,1)[..., :3]
            # coord = coord.detach().numpy()
            #
            # mug_mesh = o3d.io.read_triangle_mesh("/home/jiahui/shape_occupancy/model/mug_centered_obj/"+obj_table_list[0]+"_model_128_df.obj")
            # mug_pt = np.asarray(mug_mesh.sample_points_poisson_disk(1500).points)*mesh_scale
            # mug_trans_pt = np.matmul(rotation_matrix,mug_pt.T).T+pos.reshape((-1,3))
            #
            # fig = plt.figure()
            # ax = fig.add_subplot(projection='3d')
            #
            # # For each set of style and range settings, plot n random points in the box
            # # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
            # # ax.scatter(mug_trans_pt[:,0], mug_trans_pt[:,1], mug_trans_pt[:,2], marker='o',color="green")
            # ax.scatter(pcd[1][:,0], pcd[1][:,1], pcd[1][:,2], marker='^',color ="red")
            # ax.scatter(coord[:,0], coord[:,1],coord[:,2], marker='^',color ="blue")
            #
            # ax.set_xlabel('X Label')
            # ax.set_ylabel('Y Label')
            # ax.set_zlabel('Z Label')
            #
            # plt.show()

            np.savez(save_path+str(mug_id)+"_"+str(i),
                        cam_pose_world=transform,
                        cam_intrinsics=cam_intrinsics,
                        object_pose_world=obj_pose[0,:],
                        rgb=im,
                        depth=depth,
                        seg=seg,
                        shapenet_id= obj_table_list[0],
                        shapenet_category_id= '03797390',
                        point_cloud= pcd[1],
                        table_point_cloud= pcd[0])
        p.resetSimulation()
        # print("RESETTTT!")
        p.setGravity(0,0,0)
        # Load floor
        floor = p.loadURDF("short_floor.urdf", useFixedBase=1)#,basePosition=[0,0,0])
        set_pose(floor, Pose(Point(x=0, y=0, z=0),Euler(roll=0)))
        # load table
        table = p.loadURDF("19203/19203.urdf", useFixedBase=1)
        set_pose(table, Pose(Point(x=0, y=0, z=stable_z(table, floor)),Euler(roll=0)))


