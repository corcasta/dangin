from typing import Annotated
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import open3d as o3d


class Item(BaseModel):
    ply: Annotated[bytes, File()]
    position: str
    name: str

app = FastAPI()

@app.post("/point_clouds")
async def process_point_clouds(item: list[Item]):
    total_pcls = len(item)
    pcls = []
    for i in item:
        pcl = o3d.io.read_point_cloud_from_bytes(item["ply"])
        print("PCL Type: ", type(pcl))
        pcls.append(pcl)
    print("Done")
    return {"message": "Ok"}
