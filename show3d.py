import open3d as o3d
import numpy as np
# read file .ply
input_file = 'reconstructed.ply'
pcd = o3d.io.read_point_cloud(input_file)
# visualize the point cloud within open3d
o3d.visualization.draw_geometries([pcd])
# convert open3d format numpy array, you have point cloud in numpy array
point_cloud_in_numpy=np.asarray(pcd.points)
