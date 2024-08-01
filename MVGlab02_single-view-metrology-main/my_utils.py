from labelme import utils
import numpy as np
from numpy import linalg as LA
from collections import defaultdict
from scipy.linalg import svd, cholesky

def solution(A):
    '''
    Use SVD, solve Ax = 0, get x*
    '''
    u, e, v = svd(A)
    x = v.T[:, -1]
    return x

def cal_3D(K, p_2d):
    '''
    2d->3d, without taking depth into account.
    '''
    p_2d = np.array(p_2d)
    p_3d = np.dot(np.linalg.inv(K), p_2d)
    p_3d /= np.linalg.norm(p_3d)
    return p_3d

def get_label(img, data_line, data_plane):
    '''
    get line, intersection
    '''
    line2points = defaultdict(list) # line is decided by 2 points
    line_names = ["XY_line", "XZ_line", "YZ_line", "XY_line'", "XZ_line'", "YZ_line'"]
    intersect = []
    for shape in data_line['shapes']:
        for line in line_names:
            if shape['label'] == line:
                # if label matches line, append it to line2points
                line2points[line].append([list(reversed(shape['points'][0])), list(reversed(shape['points'][1]))])
            elif shape['label'] == 'intersection':
                intersect = list(reversed(shape['points'][0])) + [1]
    
    '''
    get mask
    '''
    lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data_plane['shapes'])
    # lbl bool数组
    # lbl_names label名称：background, XZ, YZ, XY
    
    mask = []
    for i in range(1, len(lbl_names)):  # ignore background
        mask.append((lbl == i).astype(np.uint8)) 

    return line2points, intersect, mask
    
def cal_vanish(line2points):
    '''
    影消点 vanish point
    '''
    vanish_points = defaultdict(list)
    for line_in_one_plane, points in line2points.items():
        line = []
        for point_pair in points:
            line.append(solution(np.array([point_pair[0] + [1], point_pair[1] + [1]]))) # 得到一对点的直线方程
        p = solution(np.array(line))    # 同一平面上，平行线的影消点
        vanish_points[line_in_one_plane] = p / p[-1]
    
    '''
    影消线 vanish line
    '''
    vanish_lines = np.array([solution(np.array([vanish_points['XY_line'], vanish_points["XY_line'"]])),
                             solution(np.array([vanish_points['XZ_line'], vanish_points["XZ_line'"]])),
                             solution(np.array([vanish_points['YZ_line'], vanish_points["YZ_line'"]]))]).T
    
    return vanish_points, vanish_lines

def calibrate(vanish_points):
    '''
    get intrisics matrix K
    assume no skew(w2 == 0) & square pixel(w1 == w3)
    '''
    a1, b1, c1 = vanish_points["XY_line"]
    a2, b2, c2 = vanish_points["XZ_line"]
    a3, b3, c3 = vanish_points["YZ_line"]

    '''
    v1.T * W * v2 == 0
    v1.T * W * v3 == 0
    v2.T * W * v3 == 0

    代入 v1, v2, v3, 求w1, w4, w5, w6的线性方程组解
    '''
    W_param = np.array([[a1 * a2 + b1 * b2, c1 * a2 + a1 * c2, c1 * b2 + b1 * c2, c1 * c2], 
                        [a1 * a3 + b1 * b3, c1 * a3 + a1 * c3, c1 * b3 + b1 * c3, c1 * c3],
                        [a2 * a3 + b2 * b3, c2 * a3 + a2 * c3, c2 * b3 + b2 * c3, c2 * c3]])
    
    w1, w4, w5, w6 = solution(W_param)
    W = np.array([
        [w1, 0, w4],
        [0, w1, w5],
        [w4, w5, w6]
    ])
    W /= W[-1, -1]

    '''
    W = (K * K.T).inv
    '''
    L = cholesky(W, lower=False)  # W = L.T * L, L is upper trianguler matrix 
    K = np.linalg.inv(L)  # K is upper trianguler matrix 
    K /= K[-1][-1]

    # print(K)
    return K

def cal_scene(K, vanish_lines, intersect):
    '''
    get plane's normal, n
    '''
    n = np.dot(K.T, vanish_lines)
    n = n / np.linalg.norm(n, axis=0)

    intersect_3D = cal_3D(K, intersect).reshape((3, 1))

    '''
    get D ax + by + cz + d = 0 => d = -ax - by - cz
    '''
    D = -1 * np.dot(n.T, intersect_3D)
    Pi = np.concatenate((n, D.T), axis=0)

    # print(Pi)

    Pi = Pi[:, [1, 2, 0]]
    return Pi

def reconstruction(K, Pi, img, masks):
    '''
    For each pixel, return its 3D positon and its color
    '''
    pos = []
    rgb = []
    for plane, each_mask in enumerate(masks):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if each_mask[i][j] == 1:
                    point = cal_3D(K, [i, j, 1]).reshape((3, 1))
                    # d = -ax - by - cz, -d = ax + by + cz
                    lambda_d = -Pi[-1, plane] / np.dot(Pi[:-1, plane], point)
                    pos.append(point * lambda_d)
                    rgb.append(img[i, j])
    
    return pos, rgb

def create_output(vertices, colors, filename):

    """Creates point cloud file."""

    vertices = np.hstack([np.array(vertices).reshape(-1, 3), np.array(colors).reshape(-1, 3)])
    np.savetxt(filename, vertices, fmt='%f %f %f %d %d %d') 
    ply_header = '''ply\nformat ascii 1.0\nelement vertex %(vert_num)d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'''
    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(vertices)))
        f.write(old)