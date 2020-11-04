import os
import shutil
import numpy as np
import open3d as o3d

SAVING_DIR = "/home/arthleu/pcl_downloads/"
print("[INFO] Absolute saving directory is", SAVING_DIR)

SAVING_COUNT = 0


def clear_dir():
    print("[INFO] Removing old cache files")
    shutil.rmtree(SAVING_DIR)
    print("[INFO] Creating new saving directory")
    os.mkdir(SAVING_DIR)


def pcl_save(pcls, filename_prefix) -> None:

    for i in range(len(pcls)):
        filename = "displaying_pcl_%s_%02d.pcd"%(filename_prefix, i)
        fulldir = os.path.join(SAVING_DIR, filename)
        o3d.io.write_point_cloud(fulldir, pcls[i])

    global SAVING_COUNT
    SAVING_COUNT += 1
    print("[INFO] Saved %d point clouds"%SAVING_COUNT)


def test_saving():

    points = (np.random.rand(1000, 3) - 0.5) / 4
    colors = np.random.rand(1000, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(os.path.join(SAVING_DIR, "test.pcd"), pcd)

 
if __name__ == "__main__":
    test_saving()
    print("[INFO] Test saving done.")