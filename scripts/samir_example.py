import open3d as o3d
import numpy as np
import copy
import pickle as pkl
# from stitching import myFolder

voxel_size = 0.2
threshold = 0.02

myFolder = '/home/samir/Desktop/blender/pycode/bldev2/scans/30-91822/simulations/Chess/20230901/13h46m39s/'


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.8559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])



def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=230))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=230))
    return pcd_down, pcd_fpfh



def prepare_dataset(voxel_size, source, target):
    print(":: Load two point clouds and disturb initial pose.")
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    # draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    processed_source, outlier_index = source.remove_radius_outlier(
                                              nb_points=25,
                                              radius=0.5)

    processed_target, outlier_index = target.remove_radius_outlier(
                                              nb_points=25,
                                              radius=0.5)

    return source_down, target_down, source_fpfh, target_fpfh, processed_source, processed_target, trans_init



def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size, tranform):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.95),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(10000000,0.999))
        
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    
    distance_threshold = voxel_size 
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    
    radius_normal = voxel_size * 2
    
    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=100))
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=100))
    
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
     
     # Apply transformation using ICP
    result_icp = o3d.pipelines.registration.registration_colored_icp(source, target, voxel_size, result.transformation)

    return result_icp

def save_transformation(folder_path, source, target, transformation, name = "", save = False):
    with open(folder_path+ '/temp'+ name + '.pkl','wb') as f:
        pkl.dump(transformation, f)

    #test load
    with open(folder_path + '/temp' + name + '.pkl','rb') as f:
        x = pkl.load(f)
        # print(x)

    # save combined pointcloud as pcd file
    if(save):
       return save_registration_result(source, target, x,folder_path +'/' + name + '.ply' )



def save_registration_result(source, target, transformation,
                             title,
                             save_result = True,
                             visualize_result = False):
    
    filename = title
    
    # apply the chosen transformation to source and target
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    
    # combine them and create the newpoint cloud
    newpointcloud = source_temp + target_temp
    # newpointcloud.paint_uniform_color([0,0.5,0.1])
    
    #save
    if save_result == True: 
        o3d.io.write_point_cloud(filename, newpointcloud)
    
    #visualize
    if visualize_result == True:
        o3d.visualization.draw_geometries([newpointcloud],
                                          width=1000, height=800,
                                         window_name='newpointcloud-')
    return newpointcloud


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_down_sample(ind)
    outlier_cloud = cloud.select_down_sample(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
Collapse












