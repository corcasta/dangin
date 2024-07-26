import cv2
import numpy as np
import open3d as o3d
from typing import Generator

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# |************************* MESSAGE ****************************|
# | This is intended to be replace for a class that will behave  |
# |similar as the DataLoader in pytorch. This is just a temporary|
# |solution that will not escalate appropiately in the future.   |
# ****************************************************************
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def dataloader_dmap(dataset_path: str, ground_truth: bool = False) -> Generator[tuple[np.array, np.array], None, None]:
    """
    Loads depth map and wrench/color image

    Args:
        dataset_path (str): Path to dataset
        ground_truth (bool, optional): Current datasets have 2 files to analyze, if this setting is set to False
                                       You will get back the inference image. Defaults to False.

    Yields:
        Generator[np.array, np.array]: _description_
    """
    counter = 0
    while True:
        try:
            render_path = dataset_path + f'/render{counter}'
            if ground_truth:
                dmap_path = render_path + '/dmap.npy'
            else:
                dmap_path = render_path + '/nndepth.npy'
            wrench_path = render_path + '/image8.png'
            dmap = np.load(dmap_path)                           # Depth maps are in millimeters
            wrench_img = cv2.imread(wrench_path)/255.0
            counter +=1
            yield dmap, wrench_img
        except:
            break
        
def dataloader_ply(dataset_path: str) -> Generator[tuple[np.array, np.array], None, None]:
    """_summary_

    Args:
        dataset_path (str): _description_

    Yields:
        _type_: _description_
    """
    counter = 0
    while True:
        try:
            render_path = dataset_path + f'/render{counter}'
            ply_path = render_path + f'/{counter}.ply'
            ply = o3d.io.read_point_cloud(ply_path)
            counter +=1
            yield ply
        except:
            break