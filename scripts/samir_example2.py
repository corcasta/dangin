import open3d as o3d
from help_functions import *
myFolder = '/home/samir/Desktop/blender/pycode/bldev2/scans/30-91822/simulations/Dental/20231013/10h29m59s/'
external_path = myFolder + "render"

def print_point_cloud(filename, title):
    source = o3d.io.read_point_cloud(filename)
    o3d.visualization.draw_geometries([source], width=1000, height=800, window_name='Open3D-' + str(title))


def global_and_icp_registration(source, target, title="temp"):
    source_down, target_down, source_fpfh, target_fpfh, processed_source, processed_target, trans_init = prepare_dataset(
        voxel_size, source, target)

    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size, trans_init)
    print("result_ransac")
    print("inlier_rmse = ", result_ransac.inlier_rmse)
    print("fitness = ", result_ransac.fitness)

    result_icp = refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac)

    result = save_transformation(source, target, result_icp.transformation, title, True)

    return result, result_icp.transformation

def external(external_path, start, stop, step):
    print("ex", external_path)
    source = o3d.io.read_point_cloud(external_path[0])  # external[0])
    # for i in range(0, 30, 10):
    #     target = o3d.io.read_point_cloud(external_path + str(i + 1) + '/gtcloud.ply')  # external[i+1])
    #     source, transformation = global_and_icp_registration(source, target)
    #     print('i:', i)
    # print('step1')
    # processed_source, outlier_index = source.remove_radius_outlier(nb_points=25, radius=0.5)
    # print('step2')
    # # save pointcloud
    # ext = "/home/samir/Desktop/blender/pycode/bldev2/scans/30-91822/simulations/Dental/20231013/10h29m59s/astitched/external.pcd"  # "../stitched/external.pcd"
    # o3d.io.write_point_cloud(ext, processed_source)
    # print('step3')
    # print_point_cloud(ext, "external")

def external_main_code():
    print("external")
    print('2')
    source = o3d.io.read_point_cloud(external_path + str(0) + '/gtcloud.ply')  # external[0])
    for i in range(0, 30, 10):
        target = o3d.io.read_point_cloud(external_path + str(i + 1) + '/gtcloud.ply')  # external[i+1])
        source, transformation = global_and_icp_registration(source, target)
        print('i:', i)
    print('step1')
    processed_source, outlier_index = source.remove_radius_outlier(nb_points=25, radius=0.5)
    print('step2')
    
  
    # save pointcloud
    ext = "/home/samir/Desktop/blender/pycode/bldev2/scans/30-91822/simulations/Dental/20231013/10h29m59s/astitched/external.pcd"  # "../stitched/external.pcd"
    o3d.io.write_point_cloud(ext, processed_source)
    print('step3')
    # print_point_cloud(ext, "external")