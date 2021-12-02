import pybullet as p
from pybullet_utils import set_pose, load_pybullet, Pose, Point, stable_z, connect, Euler,get_pose,get_pose_distance,get_center_extent,get_aabb, pairwise_collision, remove_body
from imageio import imwrite
import numpy as np
import random
from skimage import img_as_ubyte
import os
from matplotlib import pyplot as plt


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

def gen_traj(duration=100):
    time = np.arange(duration)#.reshape((-1,1))
    print("time",time.shape)
    roll = 2*time
    x_offset = np.random.rand()
    y_offset = np.random.rand()#+random.random()
    radius = 1.

    mu = 0
    sigma = 0.02
    x_eps = np.random.normal(mu, sigma, duration)#.reshape((-1,1))
    y_eps = np.random.normal(mu, sigma, duration)#.reshape((-1,1))
    print("x_eps",x_eps.shape)
    x = radius*np.cos(roll*np.pi/180+x_offset)#+x_eps
    y = radius*np.sin(roll*np.pi/180+y_offset)#+y_eps

    # x = np.convolve(x,np.ones(4),mode = 'same').reshape((-1,1))
    # y = np.convolve(y,np.ones(4),mode = 'same').reshape((-1,1))
    z =np.random.rand(duration,1)+1.2

    x = x.reshape((-1,1))
    y=y.reshape((-1,1))
    z =z.reshape((-1,1))
    location = np.concatenate((x,y,z),axis=1)
    print("location",location,x.shape,z.shape,location.shape)
    return location

if __name__ == "__main__":
    # start physics server
    # connect()
    seed=12
    folder = "pic_4/"
    random.seed(seed)
    path = folder+"obj"+str(seed)+".txt"
    if(os.path.exists(path)):
        os.remove(path)
    file_obj = open(path,"w")
    pClient = p.connect(p.GUI)
    p.setGravity(0,0,10)
    # Load floor
    floor = p.loadURDF("short_floor.urdf", useFixedBase=1)#,basePosition=[0,0,0])
    set_pose(floor, Pose(Point(x=0, y=0, z=0),Euler(roll=0)))
    file_obj.writelines(str(get_pose(floor)))
    file_obj.writelines("\n")
    # load table
    table = p.loadURDF("19203/19203.urdf", useFixedBase=1)
    set_pose(table, Pose(Point(x=0, y=0, z=stable_z(table, floor)),Euler(roll=0)))
    file_obj.writelines(str(get_pose(table)))
    file_obj.writelines("\n")
    obj_list = ["1c3fccb84f1eeb97a3d0a41d6c77ec7c_model_128_df","1ea9ea99ac8ed233bf355ac8109b9988_model_128_df"]#,"3c0467f96e26b8c6a93445a1757adf6_model_128_df_dec","6faf1f04bde838e477f883dde7397db2_model_128_df_dec"]
    mugs = []
    obj_pose = np.zeros((len(obj_list),7))
    for i in range(len(obj_list)):
        obj = obj_list[i]
        while True:
            mug = load_pybullet("mug_centered_obj/"+obj+".obj", scale=0.5)
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
        obj_pose[i,:3] = np.array(get_pose(mug)[0])
        obj_pose[i,3:] = np.array(get_pose(mug)[1])
        print("obj_pose",get_pose(mug),type(get_pose(mug)))
        file_obj.writelines(str(get_pose(mug)))
        file_obj.writelines("\n")
    file_obj.close()
    # Create camera to view the scene
    roll = 0
    radius = 2
    speed = 0.0001
    end = np.array([0.0, 0.0, 1.0])

    near = 0.001
    far = 5
    projectionMatrix = p.computeProjectionMatrixFOV(60., 1.0, near, far)
    print("prokmatrix",projectionMatrix,type(projectionMatrix))
    t=0
    path =folder+"cam"+str(seed)+".txt"
    if(os.path.exists(path)):
        os.remove(path)
    file_obj = open(path,"w")
    duration = 5
    cam_location  = gen_traj(duration)
    cam_traj = np.zeros((duration,4,4))

    # x_offset = random.random()+random.random()
    # y_offset = random.random()+random.random()
    # z_offset = 12*random.random()
    # while t<10:
    #     if roll>360:
    #         roll=0
    #     roll = t*2.
    #     x = radius*np.cos(roll*np.pi/180+x_offset)
    #     y = radius*np.sin(roll*np.pi/180+y_offset)
    #     # np.conv #np.ones(4)
    #     # print("x,y",x,y)
    #     # p.resetDebugVisualizerCamera(radius, roll+90,-30,[x,y,1])
    for i in range(duration):
        location = cam_location[i,:]#np.array([x,y, z_offset])
        # print("location",location)
        viewMatrix = p.computeViewMatrix(location,[0,0,1], [0, 0, 1])
        file_obj.writelines(str(viewMatrix)+" "+str(projectionMatrix))
        file_obj.writelines("\n")
        # get image (see pybullet quickstart guide https://usermanual.wiki/Document/pybullet20quickstart20guide.479068914/html)
        _, _, im, depth, seg = p.getCameraImage(width=256, height=256, viewMatrix=viewMatrix, projectionMatrix=projectionMatrix)
        depth = far * near / (far  - (far - near) * depth)
        # print("type",type(im),type(depth),type(seg))
        # print("depth",depth.shape)#seg,seg.min(),seg.max())
        # plt.imshow(seg, interpolation='nearest')
        # plt.show()
        # imwrite(folder+"depth"+str(t)+".png",depth)
        # imwrite(folder+"seg"+str(t)+".png",seg)
        # imwrite(folder+"rgb"+str(t)+".png",im)# depth.astype(np.uint16))
        p.stepSimulation()
        t = t+1
        np.savez(folder+"frame"+str(i),im=im,depth=depth,seg=seg)
        cam_traj[i,:] = np.array(viewMatrix).reshape((4, 4)).transpose()
    file_obj.close()
    np.savez(folder+"all_matrix"+str(seed),projectionMatrix=projectionMatrix,obj_pose=obj_pose,cam_traj=cam_traj)
    print("cam traj",cam_traj.shape)
    #
    # with open(path) as f:
    #     lines = f.readlines()
    #     print("lines",lines[0],type(lines[0]))

