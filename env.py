import pybullet as p
from pybullet_utils import set_pose, load_pybullet, Pose, Point, stable_z, connect, Euler
from imageio import imwrite
import numpy as np


if __name__ == "__main__":
    # start physics server
    connect()

    # Load floor
    floor = p.loadURDF("short_floor.urdf")

    # load table
    table = p.loadURDF("19203/19203.urdf")

    set_pose(table, Pose(Point(x=0, y=0, z=stable_z(table, floor))))

    # load mug
    mug = load_pybullet("mug_centered_obj/1ea9ea99ac8ed233bf355ac8109b9988_model_128_df.obj", scale=0.5)
    set_pose(mug, Pose(Point(x=0, y=0, z=stable_z(mug, table)), Euler(roll=np.pi/2)))

    # Create camera to view the scen
    location = np.array([0.1, 0.1, 2.0])
    end = np.array([0.0, 0.0, 1.0])
    viewMatrix = p.computeViewMatrix(location, end, [0, 0, 1])

    near = 0.0001
    far = 4.0
    projectionMatrix = p.computeProjectionMatrixFOV(60., 1.0, near, far)

    # get image (see pybullet quickstart guide https://usermanual.wiki/Document/pybullet20quickstart20guide.479068914/html)
    _, _, im, _, seg = p.getCameraImage(width=256, height=256, viewMatrix=viewMatrix, projectionMatrix=projectionMatrix)
    imwrite("test.png", im)
