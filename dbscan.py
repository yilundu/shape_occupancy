import open3d as o3d
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# plane detectiona

pcd = o3d.io.read_point_cloud("/home/jiahui/OBJ_SLAM/ref.pcd")
o3d.visualization.draw_geometries([pcd]) #inlier_cloud,

target = pcd
w, index = target.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=2000)
inlier_cloud = pcd.select_by_index(index)
outlier_cloud = pcd.select_by_index(index, invert=True)
inlier_cloud.paint_uniform_color([1, 0, 0])
# outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
# o3d.visualization.draw_geometries([outlier_cloud]) #inlier_cloud,
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(
        outlier_cloud.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

u, counts = np.unique(labels, return_counts=True)
print("u",u)
print("counts",counts)
print("indices",counts.shape,labels.shape)
max_label = labels.max()
for i in range(len(u)):
    print("counts each", counts[i])
    if(counts[i]>1000):
        index = np.where(labels==u[i])[0]
        print("label index",index,len(index))
        tmp = outlier_cloud.select_by_index(index)
        o3d.io.write_point_cloud("/home/jiahui/OBJ_SLAM/"+str(i)+".pcd", tmp)
        o3d.visualization.draw_geometries([tmp])


# print(f"point cloud has {max_label + 1} clusters")
# colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
# colors[labels < 0] = 0
# outlier_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
# o3d.visualization.draw_geometries([outlier_cloud])
