import numpy as np
from numpy import linalg as LA

class SingleCamera:
    def __init__(self, world_coor, pixel_coor, n):
        self.world_coor = world_coor
        self.pixel_coor = pixel_coor
        self.point_num = n

        self.P = np.empty([self.point_num * 2, 12], dtype=float)
        self.M = np.empty([3, 4], dtype=float)
        self.A = np.empty([3, 3], dtype=float)
        self.b = np.empty([3, 1], dtype=float)
        self.K = np.empty([3, 3], dtype=float)
        self.R = np.empty([3, 3], dtype=float)
        self.t = np.empty([3, 1], dtype=float)

    def composeP(self):
        i = 0
        P = np.empty([self.point_num * 2, 12], dtype=float)
        while i < self.point_num * 2:
            c = i // 2
            p1 = self.world_coor[c]
            p2 = np.array([0, 0, 0, 0])
            if i % 2 == 0:
                p3 = -p1 * self.pixel_coor[c][0]
                P[i] = np.hstack((p1, p2, p3))
            else:
                p3 = -p1 * self.pixel_coor[c][1]
                P[i] = np.hstack((p2, p1, p3))
            i += 1
        print(P)
        self.P = P

    def svdP(self):
        U, sigma, VT = LA.svd(self.P)
        V = np.transpose(VT)
        preM = V[:, -1]
        M = preM.reshape(3, 4)

        A = M[:3, :3].copy()
        b = M[:3, 3:4].copy()

        self.M = M
        self.A = A
        self.b = b

    def workInAndOut(self):
        a1T = self.A[0]
        a2T = self.A[1]
        a3T = self.A[2]
        a3T_norm = LA.norm(a3T)
        
        ro = 1 / a3T_norm

        u0 = ro * ro * (np.dot(a1T, a3T))
        v0 = ro * ro * (np.dot(a2T, a3T))

        a1a3_cross = np.cross(a1T, a3T)
        a2a3_cross = np.cross(a2T, a3T)

        theta = np.arccos((-1) * np.dot(a1a3_cross, a2a3_cross) / (LA.norm(a1a3_cross) * LA.norm(a2a3_cross)))

        alpha = ro * ro * LA.norm(a1a3_cross) * np.sin(theta)
        beta = ro * ro * LA.norm(a2a3_cross) * np.sin(theta)

        K = np.array([
            [alpha, -alpha / np.tan(theta), u0],
            [0, beta / np.sin(theta), v0],
            [0, 0, 1]
        ])

        # intrinsics
        self.K = K

        r1 = a2a3_cross / LA.norm(a2a3_cross)
        r3 = ro * a3T
        r2 = np.cross(r1, r3)

        R = np.vstack((r1, r2, r3))
        self.R = R

        t = ro * np.dot(LA.inv(K), self.b)
        self.t = t

    def selfcheck(self, w_check, c_check):
        my_size = c_check.shape[0]
        my_err = np.empty([my_size])
        for i in range(my_size) :
            test_pix = np.dot(self.M, w_check[i])
            u = test_pix[0] / test_pix[2]
            v = test_pix[1] / test_pix[2]
            u_c = c_check[i][0]
            v_c = c_check[i][1]
            print("you get test point %d with result (%f,%f)" % (i, u, v))
            print("the correct result is (%f,%f)" % (u_c,v_c))
            my_err[i] = (abs(u - u_c) / u_c + abs(v - v_c) / v_c)/2
        average_err = my_err.sum() / my_size    # normalization
        print("The average error is %f ," % average_err)
        if average_err > 0.1:
            print("which is more than 0.1")
        else:
            print("which is smaller than 0.1, the M is acceptable")