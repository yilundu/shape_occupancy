import open3d as o3d
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def pairwise_registration(source, target):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id])
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph


def readTraj(path):
    file = open(path)
    lines = file.readlines()
    rows = len(lines)
    datamat = np.zeros((rows,7))

    row=0
    for line in lines:
        line = line.strip().split("\t")
        datamat[row,:] = line[:]
        row+=1
    return datamat

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    # outlier_cloud.paint_uniform_color([1, 0, 0])
    # inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud])#, outlier_cloud])

name = "ref"
path = "/home/jiahui/OBJ_SLAM/"+name+"/"
filepath = "/home/jiahui/OBJ_SLAM/"+"ref"+".txt"
pcds = []
names = os.listdir(path)
voxel_size = 0.02
i=0
for name in names:
    if name.endswith('.pcd'):
        if i==0 or i==200:
            tmp = o3d.io.read_point_cloud(path+name)
            points = np.asarray(tmp.points)
            print(points[:,0].min(),points[:,1].min(),points[:,2].min())
            print(points[:,0].max(),points[:,1].max(),points[:,2].max())
            o3d.visualization.draw_geometries([tmp])
            # tmp = tmp.select_by_index(np.where(np.asarray(tmp.points)[:,2]>=0.578)[0])
            # o3d.visualization.draw_geometries([tmp])
            tmp = tmp.select_by_index(np.where(np.asarray(tmp.points)[:,2]<=2.15)[0])
            # o3d.visualization.draw_geometries([tmp])
            # tmp = tmp.select_by_index(np.where(points[:,0] <1.46)[0])
            # o3d.visualization.draw_geometries([tmp])
            # tmp = tmp.select_by_index(np.where(points[:,0] >=-2)[0])
            # o3d.visualization.draw_geometries([tmp])
            # tmp = tmp.select_by_index(np.where(np.asarray(tmp.points)[:,1]<0.25)[0])
            # o3d.visualization.draw_geometries([tmp])
            tmp = tmp.select_by_index(np.where(np.asarray(tmp.points)[:,1]>=0.24)[0],invert=True)
            # o3d.visualization.draw_geometries([tmp])
        #     voxel_down_pcd = tmp.voxel_down_sample(voxel_size=voxel_size)
            tmp.estimate_normals()
            pcds.append(tmp)
        i=i+1
o3d.visualization.draw_geometries(pcds)
print("Full registration ...",len(pcds))
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    pose_graph = full_registration(pcds,
                                   max_correspondence_distance_coarse,
                                   max_correspondence_distance_fine)

option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=max_correspondence_distance_fine,
    edge_prune_threshold=0.25,
    reference_node=0)

with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)
print("Transform points and display")

pc_final = o3d.geometry.PointCloud()
current_points = np.array(pcds[0].points)
for point_id in range(1,len(pcds)):
    print(pose_graph.nodes[point_id].pose)
    pcds[point_id].transform(pose_graph.nodes[point_id].pose)
    current_points = np.concatenate([current_points,np.array(pcds[point_id].points)])
print("Current points shape",current_points.shape)
pc_final.points = o3d.utility.Vector3dVector(current_points)
o3d.visualization.draw_geometries([pc_final])

# traj = np.loadtxt(filepath)
# print("traj",traj.shape,len(pcds))
# print(traj)
# pcd_combined = o3d.geometry.PointCloud()
# for point_id in range(len(pcds)):
#     pose = traj[point_id]
#     r = R.from_quat([pose[4],pose[5],pose[6],pose[3]])
#     # transform = np.zeros((4,4))
#     # transform[0:3,0:3] = r.as_matrix()
#     # transform[3,3]=1
#     # transform[0:3,3]=pose[0:3]
#     # pcds[point_id].transform(np.linalg.inv(transform))
#     pcds[point_id].translate(pose[0:3,np.newaxis],relative=False)
#     pcds[point_id].rotate(np.linalg.inv(r.as_matrix()))
#     pcd_combined += pcds[point_id]
# pcd_final = pcd_combined#.voxel_down_sample(voxel_size=voxel_size)
# pcd_final = pc_final.uniform_down_sample(every_k_points=5)
# o3d.visualization.draw_geometries([pcd_final])
# print("finish")
# cl, ind = pc_final.remove_radius_outlier(nb_points=16, radius=0.04)
# display_inlier_outlier(pcd_final, ind)
# print("conbined_down")
# # points = np.asarray(pcd_final.points)
# # print(points)
# # pcd_combined_down = pcd_final.select_by_index(np.where(points[:, 0]-points[:,0].min() <=1.6)[0])
o3d.io.write_point_cloud("/home/jiahui/OBJ_SLAM/ref.pcd", pc_final)
# o3d.visualization.draw_geometries([pcd_final])













