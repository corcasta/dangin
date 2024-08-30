# helper function to load each image, dmap, etc..
import copy
import numpy as np
import open3d as o3d

class Stitcher():
    def __init__(self, voxel_size=0.2, radius_normal=0.4, radius_fpfh_feature=1.0,
                 max_nn_fpfh_feature=230, max_nn_normal=230, max_nn_local_reg=100, 
                 dist_correspond_checker=0.3, edge_length_correspond_checker=0.95,
                 ransac_max_iterations=10000000, ransac_confidence=0.999, max_correspond_dist_local_reg=0.2,
                 nn_stat_outlier=20, std_ratio_outlier=2.0, nn_radius_outlier=25, radius_outlier=0.5):
        """_summary_

        Args:
            voxel_size (float, optional):                       Voxel size to downsample into. 
                                                                Defaults to 0.2.
            radius_normal (float, optional):                    Search radius to estimate normals. 
                                                                It tends to be related with the voxel size (voxel_size * gain).
                                                                Defaults to 0.4.
            radius_fpfh_feature (float, optional):              Search radius to compute fpfh feature. 
                                                                It tends to be related with the voxel size (voxel_size * gain).
                                                                Defaults to 1.0.
            max_nn_fpfh_feature (int, optional):                Maximum neighbors will be searched to compute fpfh. 
                                                                Defaults to 230.
            max_nn_normal (int, optional):                      Maximum neighbors will be searched to estimate normals.
                                                                It affects downsample point cloud. 
                                                                Defaults to 230.
            max_nn_local_reg (int, optional):                   Maximum neighbors will be searched to estimate normals. 
                                                                It affects local registration.
                                                                Defaults to 100.
            dist_correspond_checker (float, optional):          Maximum correspondence points-pair distance. 
                                                                It affects global registration.
                                                                It tends to be related with the voxel size (voxel_size * gain).
                                                                Defaults to 0.3.
            edge_length_correspond_checker (float, optional):   Correspondence checker based on length.
                                                                Float value between 0(loose)-1(strict). 
                                                                It affects global registration.
                                                                Defaults to 0.95.
            ransac_max_iterations (int, optional):              Max iterations for ransac.
                                                                It affects global registration.
                                                                Defaults to 10000000.
            ransac_confidence (float, optional):                Ransac confidence.
                                                                It affects global registration
                                                                Defaults to 0.999.
            max_correspond_dist_local_reg (float, optional):    Maximum correspondence points-pair distance. 
                                                                It affects local registration
                                                                Defaults to 0.2.
            nn_stat_outlier (int, optional):                    Number of neighbors around the target point for statistical ourlier removal. 
                                                                Defaults to 20.
            std_ratio_outlier (float, optional):                Standard deviation ratio for statistical outlier removal. 
                                                                Defaults to 2.0.
            nn_radius_outlier (int, optional):                  Number of points within the radius for outlier removal. 
                                                                Defaults to 25.
            radius_outlier (float, optional):                   Radius of the sphere for outlier removal. 
                                                                Defaults to 0.5.
        """
        self.voxel_size = voxel_size
        self.max_nn_normal = max_nn_normal
        self.radius_normal =  radius_normal
        self.radius_fpfh_feature = radius_fpfh_feature
        self.max_nn_fpfh_feature = max_nn_fpfh_feature
        self.dist_correspond_checker = dist_correspond_checker      
        self.edge_length_correspond_checker = edge_length_correspond_checker            
        self.ransac_max_iterations = ransac_max_iterations                              
        self.ransac_confidence = ransac_confidence                                      
        self.max_correspond_dist_local_reg = max_correspond_dist_local_reg
        self.radius_local_reg = radius_normal
        self.max_nn_local_reg = max_nn_local_reg
        self.current_transform = np.eye(4)                                              # Identity Matrix, represents transform w.r.t. world frame
        self.transforms_history = [self.current_transform]                              # This will store all transforms/poses of each pcd
                                                                                        # is really helpfull to evaluate metrics and comparisons.
    
        self.nn_stat_outlier = nn_stat_outlier
        self.std_ratio_outlier = std_ratio_outlier
        self.nn_radius_outlier = nn_radius_outlier
        self.radius_outlier = radius_outlier
         
         
    @classmethod
    def generate_pointcloud(cls, img: np.array, dmap: np.array, pcd: o3d.geometry.PointCloud) -> None:
        """
        Populates point cloud given the information provided by an image and depth map.

        Args:
            img (np.array): PNG image
            dmap (np.array): Depth map/image
            pcd (o3d.geometry.PointCloud): Point cloud object
        """
        height, width = dmap.shape
        xyz = np.ones(shape=(height, width, 3))  
        x = np.arange(width).reshape((1, width))
        y = np.arange(height).reshape((height, 1))
        xyz[:, :, 0] = xyz[:, :, 0]*x                                 # Fill with horizontal indexes
        xyz[:, :, 1] = xyz[:, :, 1]*y                                 # Fill with vertical indexes  
        xyz[:, :, 2] = dmap                                           # Fill with depth, z coords w.r.t camera frame  
        xyz[:, :, 0] = 0.1655 * (xyz[:, :, 0]-80) * xyz[:, :, 2]/80   # Updated with x coords w.r.t camera frame 
        xyz[:, :, 1] = 0.1655 * (xyz[:, :, 1]-80) * xyz[:, :, 2]/80   # Updated with y coords w.r.t camera frame 
        xyz = xyz.reshape(width*height, -1)                           # Shape required by open3d
        rgb = img.reshape((width*height, -1))[::-1]
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    
    
    @classmethod
    def merge_point_clouds(cls, source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, transform: np.array) -> None:
        """
        Transform a copy of source point cloud to target point cloud reference frame and merges it in target point cloud.
        *Changes are reflected in target object itself as this behaviour is same as being passed by 
        reference in c++ thus no need to return anything.

        Args:
            source (o3d.geometry.PointCloud): Point cloud to be transformed.
            target (o3d.geometry.PointCloud): Point cloud that will be merged with the transformed source point cloud.
            transform (np.array):                        Homogeneous transformation matrix of shape 4x4. 
                                                         This matrix MUST represent the transform form s
                                                         ource to target.
        """
        source_copy = copy.deeepcopy(source)
        source_copy.transform(transform)
        target += source
        
        
    def point_cloud_init_pose(self, pcd: o3d.geometry.PointCloud, transform: np.array) -> None:
        """
        Transforms point cloud (source & target) to the pose given by the transform matrix.
        *Changes are reflected in pcd object itself as this behaviour is same as being passed by 
         reference in c++ thus no need to return anything.

        Args:
            pcd (o3d.geometry.PointCloud): Point cloud to be transformed
            target (o3d.geometry.PointCloud): Point cloud to be transformed
            transform (np.array):                        Homogeneous transformation matrix shape 4x4. 
                                                         This matrix represent the starting pose/location 
                                                         for both point clouds.
        """
        pcd.transform(transform)
        
        
    def global_and_icp_registration_v2(self, source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud) -> o3d.pipelines.registration.RegistrationResult:
        """
        Calculates the results necessary for transforming source point cloud to target point cloud as best as possible. 

        Args:
            source (o3d.geometry.PointCloud): Point cloud that is looking to be transformed.
            target (o3d.geometry.PointCloud): Point Cloud that works as the reference base.

        Returns:
            o3d.pipelines.registration.RegistrationResult: Object that contains registration results 
                                                                      such as transformation, correspondence_set, 
                                                                      fitness, inlier_rmse.
        """
        source_down, source_fpfh = self._preprocess_point_cloud(source)
        target_down, target_fpfh = self._preprocess_point_cloud(target)
        result_ransac = self._global_registration(source_down, target_down, source_fpfh, target_fpfh)
        result_icp = self._local_registration(source, target, result_ransac.transformation)
        return result_icp
    
    
    def outlier_removal(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Removes outliers in point cloud.

        Args:
            pcd (o3d.geometry.PointCloud): Point cloud to be cleaned.

        Returns:
            o3d.geometry.PointCloud: Point cloud without outliers.
        """
        #pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=self.nn_stat_outlier, std_ratio=self.std_ratio_outlier)
        pcd, ind = pcd.remove_radius_outlier(nb_points=self.nn_radius_outlier, radius=self.radius_outlier)
        return pcd
    
    
    def set_current_transform(self, transform: np.array) -> None:
        """
        Updates current transform. The input MUST represent the transform of the newest point cloud to world frame.
        E.g.  current_transform = stitcher.current_transform @ result_icp.transformation  

        Args:
            transform (np.array): Homogeneous transformation matrix shape 4x4. 
        """
        self.current_transform = transform
    

    def _preprocess_point_cloud(self, pcd: o3d.geometry.PointCloud) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
        """
        Creates a simplified point cloud (downsampled) of the input point cloud.

        Args:
            pcd (o3d.geometry.PointCloud): Point cloud

        Returns:
            tuple[o3d.geometry.PointCloud, 
            o3d.pipelines.registration.Feature]: Tuple comtaining simplified point cloud 
                                                            and its corresponding features descriptor.
        """
        pcd_down = pcd.voxel_down_sample(self.voxel_size)
        pcd_down = self.outlier_removal(pcd_down)
        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius_normal, max_nn=self.max_nn_normal))
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius_fpfh_feature, max_nn=self.max_nn_fpfh_feature))
        return pcd_down, pcd_fpfh

    
    def _global_registration(self, source_down: o3d.geometry.PointCloud, 
                             target_down: o3d.geometry.PointCloud, 
                             source_fpfh: o3d.pipelines.registration.Feature, 
                             target_fpfh:o3d.pipelines.registration.Feature) -> o3d.pipelines.registration.RegistrationResult:
        """
        Calculates the results necessary for transforming source point cloud to target point cloud as a best 
        initial allignment. Point clouds can be placed anywehere, global registration will produce the a rough 
        best alignment. This requires fine tunning/local registration to get better allignment. 

        Args:
            source_down (o3d.geometry.PointCloud): Point cloud that is looking to be transformed.
            target_down (o3d.geometry.PointCloud): Point cloud that works as the reference base.
            source_fpfh (o3d.pipelines.registration.Feature): Point cloud corresponding features descriptor.
            target_fpfh (o3d.pipelines.registration.Feature): Point cloud corresponding features descriptor.

        Returns:
            o3d.pipelines.registration.RegistrationResult: Object that contains registration 
                                                                      results such as transformation, 
                                                                      correspondence_set, fitness, 
                                                                      inlier_rmse.
        """
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True, self.dist_correspond_checker,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, 
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(self.edge_length_correspond_checker),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(self.dist_correspond_checker)
            ], 
            o3d.pipelines.registration.RANSACConvergenceCriteria(self.ransac_max_iterations, self.ransac_confidence))
        return result


    def _local_registration(self, source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, transform: np.array) -> o3d.pipelines.registration.RegistrationResult:
        """
        Calculates the results necessary for transforming source point cloud to target point cloud as best as possible.
        Point clouds need to be placed together on the area they are meant to be matched. As this method creates only 
        fine tunning allignment. 

        Args:
            source (o3d.geometry.PointCloud): Point cloud that is looking to be transformed.
            target (o3d.geometry.PointCloud): Point cloud that works as the reference base.
            transform (np.array):                        Homogeneous transformation matrix shape 4x4. 
                                                         This matrix represents the global alignment 
                                                         between the point clouds (source -> target).

        Returns:
            o3d.pipelines.registration.RegistrationResult: Object that contains registration results 
                                                                      such as transformation, correspondence_set, 
                                                                      fitness, inlier_rmse.
        """
        source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius_local_reg, max_nn=self.max_nn_local_reg))
        target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius_local_reg, max_nn=self.max_nn_local_reg))
        result = o3d.pipelines.registration.registration_icp(source, target, self.max_correspond_dist_local_reg, transform, o3d.pipelines.registration.TransformationEstimationPointToPlane())
        #result = o3d.pipelines.registration.registration_colored_icp(source, target, voxel_size, result.transformation)
        return result


    