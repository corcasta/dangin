from fastapi import FastAPI, File, UploadFile, Form
from dangin import Stitcher
import open3d as o3d
import numpy as np
import requests
import tempfile
import pathlib
import copy
import yaml

URL = "http://nndb.danbots.com:81/apis/v1/stitching-simulation-point-cloud"
ROOTH_PATH = str(pathlib.Path('').parent.resolve())
CONFIG_PATH = ROOTH_PATH + "/" + "config.yaml"
ROLLING_WINDOW = 1                  # Represents how many previous point clouds are going to be used 
                                    # as matching reference for the new point cloud .
PCD_SIZE = 160*160                  # Each point cloud is composed of 128*128 points, its self explanatory.
# When you sum point clouds Open3d appends the points at the end,
# ROLLING_WINDOW_POINTS will indicate how many point we need
# to grab starting from the end such that we ensure to be taking 
# the N FULL point clouds define by rolling window.
ROLLING_WINDOW_POINTS = ROLLING_WINDOW * PCD_SIZE
DUMMY_IMG_PATH = ROOTH_PATH + "/datasets/teeth500/images/0.png"
with open(DUMMY_IMG_PATH, 'rb') as f:
    DUMMY_IMG = f.read()


app = FastAPI()
@app.post("/point_clouds/")
async def upload_and_process_point_cloud(
        ply: list[UploadFile] = File(...),  # List of uploaded point cloud files
        position: list[str] = Form(...),    # List of positions from the form
        name: list[str] = Form(...)         # List of names from the form
):  
    with open(CONFIG_PATH, "rt") as f:
        config = yaml.safe_load(f.read())
    stitcher = Stitcher(**config["stitcher_params"])
    pcd_0 = o3d.geometry.PointCloud()  
    pcd_1 = o3d.geometry.PointCloud()
    
    # ***************************** THIS CHUNK IS JUST TO EXTRACT THE FIRST PCD  *********************
    current_file = ply.pop(0)
    current_position = position.pop(0)
    current_name = name.pop(0)
    
    file_content = await current_file.read()                # Read the file content into memory (as bytes)
    # Use tempfile to create a temporary file that is automatically cleaned up
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=True) as temp_file:
        temp_file.write(file_content)                       # Write the uploaded point cloud content to the temporary file
        temp_file.flush()                                   # Ensure allMultiPartParser.max_file_size = 2 * 1024*1024 data is written to disk  
        pcd_0 = o3d.io.read_point_cloud(temp_file.name)     # Load the point cloud from the temporary file using open3d
    # ***************************** THIS CHUNK IS JUST TO EXTRACT THE FIRST PCD  *********************
            
    pcd_history = pcd_history_down = copy.deepcopy(pcd_0)
    history_points_xyz = np.array(pcd_history.points)
    
    # ****************************** THIS CHUNK DOES THE STITCHING ******************************
    for idx, file in enumerate(ply):
        print("PCD:", idx)
        current_position = position[idx]
        current_name = name[idx]
        file_content = await file.read()                    # Read the file content into memory (as bytes)
        
        # Use tempfile to create a temporary file that is automatically cleaned up
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=True) as temp_file:
            temp_file.write(file_content)                   # Write the uploaded point cloud content to the temporary file         
            temp_file.flush()                               # Ensure allMultiPartParser.max_file_size = 2 * 1024*1024 data is written to disk
            pcd_1 = o3d.io.read_point_cloud(temp_file.name) # Load the point cloud from the temporary file using open3d
  
        pcd_1 = stitcher.outlier_removal(pcd_1)
        pcd_1_temp = copy.deepcopy(pcd_1) 
        if len(history_points_xyz) >= (ROLLING_WINDOW_POINTS):                        # This if-statement makes sure that we have enough pcds to make a rolling window
            pass
        else:
            stitcher.point_cloud_init_pose(pcd_0, stitcher.current_transform) 
        
        stitcher.point_cloud_init_pose(pcd_1, stitcher.current_transform)             # As the name implies first initialize PCD on specified loc.
        result_icp = stitcher.global_and_icp_registration_v2(pcd_1, pcd_0)            # Calculate the best transform starting between PCDs. This transformation is from current pcd to previous pcd
        current_transform = stitcher.current_transform @ result_icp.transformation    # Calculate the new current transform of the trayectory. 
        stitcher.set_current_transform(current_transform)  
        stitcher.transforms_history.append(result_icp.transformation)                 # This is useful to have a graph of all the transforms for further analysis                        
        pcd_history += pcd_1.transform(result_icp.transformation)
        
        history_points_xyz = np.array(pcd_history.points)                            
        history_points_rgb = np.array(pcd_history.colors) 
            
        if len(history_points_xyz) >= (ROLLING_WINDOW_POINTS):                        # If we have enough points for rolling windows we proceed with it  
            pcd_0 = o3d.geometry.PointCloud()                                         # it means that pcd_0 will not be just the old point cloud but a group of prev pcds.
            pcd_0.points = o3d.utility.Vector3dVector(history_points_xyz[-(ROLLING_WINDOW_POINTS):, :]) # Get points information representing rolling window
            pcd_0.colors = o3d.utility.Vector3dVector(history_points_rgb[-(ROLLING_WINDOW_POINTS):, :]) # Get points information representing rolling window
        else:
            pcd_0 = copy.deepcopy(pcd_1_temp)  
    # ****************************** THIS CHUNK DOES THE STITCHING ******************************
        
        
    # ****************************** THIS SEND THE STITCHED PCDS ******************************    
        # Save the point cloud to a PLY file in memory
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=True) as temp_file:
            o3d.io.write_point_cloud(temp_file.name, pcd_history)
            temp_file.flush()
            
            files = {
                'point_cloud_file':open(temp_file.name, 'rb').read(),
                'fringe_image': DUMMY_IMG,
                'wrap_image': DUMMY_IMG
            }
            
            data = {
                "model": "396",
                "name": "Hello World",
                "position": f'{idx}'
            }
            response = requests.post(URL, files=files, data=data)
            print(response.status_code)
            #print(response)
        
            
    # ****************************** THIS SEND THE STITCHED PCDS ******************************           
