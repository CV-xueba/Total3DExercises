import json
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os

from labelme import utils
from collections import defaultdict
from scipy.linalg import svd, cholesky, qr
import open3d as o3d
# from open3d import web_visualizer

from my_utils import *


# load data

json_file_line = os.path.join('labelme_data', 'chessboard_line.json') # label the parallel lines
json_file_plane = os.path.join('labelme_data', 'chessboard_plane.json') # label the vertical planes
data_line = json.load(open(json_file_line))
data_plane = json.load(open(json_file_plane))
# img = utils.img_b64_to_arr(data_plane['imageData'])  # png: 4 channels

img_path = os.path.join('images', 'chessboard.jpg')
img = mpimg.imread(img_path)

line2points, intersect, masks = get_label(img, data_line, data_plane)

# calculate points and lines at infinity from labeled parallel lines

vanish_points, vanish_lines = cal_vanish(line2points)

# according to intersection points of 3 pairs parallel lines which are mutually orthogonal,
# calculate camera matrix

K = calibrate(vanish_points)

# according to lines at infinity of 3 vertical planes and camera matrix, 
# calculate scene plane orientations (normal vectors);
# substitute the common point(assuming projective depth) of plane intersection lines into plane equations, 
# calculate distance between each plane and camera center

Pi = cal_scene(K, vanish_lines, intersect)

# substituting 2D points into corresponding plane equation,
# calculate 3D positions for masked image (up to a unknown scale)

pos, rgb = reconstruction(K, Pi, img, masks)

# save as ply file

create_output(pos, rgb, r'chessboard_3D.ply')


# visualization of point clouds.
pcd = o3d.io.read_point_cloud('chessboard_3D.ply')
# o3d.visualization.draw_geometries([pcd])
# web_visualizer.draw(pcd)