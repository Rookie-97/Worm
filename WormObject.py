from softArm import softArm
import numpy as np
from math import cos, sin, atan2, sqrt, pi
from scipy.optimize import least_squares, differential_evolution, brute, basinhopping, shgo, dual_annealing
from scipy.spatial.transform import Rotation
from angles import normalize_angle
import time
import traceback

class target (object):
	def __init__(self, pos=[0,0,0.1], dir=[0,0,1]):
		self.x = pos[0]
		self.y = pos[1]
		self.z = pos[2]

class WormObject(object):
    def __init__(self, worm):
        self.num = len(layer)
        self.worm = worm
        self.layer = dict()
        self.ladder = dict()
        self.target target()

    def getArms(self):
        return 1

    def getConnection(self):
        return 1
        
    def aim(self):
        1
        
    def capture(self):
        1
    		
    def swallow(self):
        1

    def inverse_kinematic(self, pts=[0, 3, 5], n=[0, 1, 0], a=[0, 0, 1], euler=[pi/2, pi/2, -pi/2], model=1, input=[], length=2):
        def test_square(dst, x0, a, n, R):  # a1 a2 a3 b1 b2 b3 r1 r2 r3
             
                # rank = np.linalg.matrix_rank(J)
                w = np.array([1]*7)
                qmin = [0, -2*pi,
                        0, -2*pi,
                        0, -2*pi, 0.3]
                qmax = [pi, 2*pi,
                        pi, 2*pi,
                        pi, 2*pi, 5]
                act_up = [pi * 17 / 18, 2*pi * 17 / 18,
                          pi * 17 / 18, 2*pi * 17 / 18,
                          pi * 17 / 18, 2*pi * 17 / 18, 4.7]
                act_down = [pi * 1 / 18, -2*pi * 17 / 18,
                            pi * 1 / 18, -2*pi * 17 / 18,
                            pi * 1 / 18, -2*pi * 17 / 18, 0.5]
                for i in range(7):
                    if q[i] >= qmax[i] or q[i] <= qmin[i]:
                        w[i] = 0
                    elif act_up[i] < q[i] < qmax[i] or \
                            qmin[i] < q[i] < act_down[i]:
                        if q[i] > act_up[i]:
                            d = float((q[i] - float(act_up[i])) / (float(qmax[i]) - float(act_up[i])))
                        elif q[i] < act_down[i]:
                            d = float(float((act_down[i]) - float(q[i])) / (float(act_down[i]) - float(qmin[i])))
                        else:
                            d = 1
                        w[i] = -d * d + 1
                    else:
                        w[i] = 1
                Hw = np.diag(w ** 2)
                u, s, vh = np.linalg.svd(J.dot(Hw).dot(J.transpose()))
                minimal = s[0]
                w_total = 0
                for i in range(len(w)):
                    w_total += 1-w[i]
                lmax = 0.1 + 0.1*(w_total)
                e = np.sqrt(3 * lmax)
                '''e = 0.4
                lmax =  1.2'''
                for n in s:
                    if minimal > n:
                        minimal = n
                if minimal <= e:
                    l = np.array([(1 - (minimal / e) ** 2) * lmax] * 6)
                else:
                    l = np.array([.0]*6)
                L = np.diag(l)

                I = np.diag([1]*7)
                K = np.diag([1]*7)
                inv = np.linalg.inv(J.dot(Hw).dot(J.transpose()) + L)
                projection = K.dot(I - J.transpose().dot(np.linalg.inv(J.dot(J.transpose()))).dot(J))
                criterion = np.array([0,0,0,0,0,0,1])
                #print(-projection.dot(criterion))
                joint_deta = Hw.dot(J.transpose()).dot(inv).dot(desired) #-projection.dot(criterion)+
                joint_deta = np.array(joint_deta + q)
                return joint_deta[0]
            
            def tranformation_string(res):
                result = [0]*len(res)
                for i in range(int(len(res)/3)):
                    if 0 > res[3*i]:
                        result[3*i] = -res[3*i]
                        result[3*i+1] = res[3*i+1] + pi
                    elif 0 < res[3*i]:
                        result[3*i] = res[3*i]  # a1
                        result[3*i+1] = res[3*i+1]  # b1
                    # lm1
                    result[3*i] = normalize_angle(result[3*i])
                    result[3*i+1] = normalize_angle(result[3*i+1])
                    result[3*i+2] = res[3*i+2] * res[3*i] / sin(res[3*i] / 2) / 2
                return result
            def tranformation_normal(res):
                result = [0]*9
                result[0] = new[0]
                result[1] = new[1]
                result[2] = new[6]

                result[3] = new[2]
                result[4] = new[3]
                result[5] = new[6]

                result[6] = new[4]
                result[7] = new[5]
                result[8] = new[6]
                return result
            now = time.time()
            x0_rosenbrock = np.array(x0).astype('float64')
            # normal type
            '''res = least_squares(test_3_7dofs, x0_rosenbrock,
                                bounds=([0, -2 * pi, 0, -2 * pi, 0, -2 * pi, 0.1],
                                        [pi, 2 * pi, pi, 2 * pi, pi, 2 * pi, 4,]))
            new = np.array([res.x[0], res.x[1],
                            res.x[2], res.x[3],
                            res.x[4], res.x[5], res.x[6]]).astype('float64')
            result = transformation_normal(new)
            '''
            # string type

            try:
                res = least_squares(string_type, x0_rosenbrock,
                                bounds=([-pi, -pi, -pi, -2*pi, -2*pi, -2*pi, 0.09, 0.09, 0.09],
                                        [pi, pi, pi, 2*pi, 2*pi, 2*pi, 0.24, 0.24, 0.24]), ftol = 1e-4, xtol= 1e-4)
                # bounds = [(-pi, pi), (-2*pi, 2*pi), (0.09, 0.24), (-pi, pi), (-2*pi, 2*pi), (0.09, 0.24),
                #           (-pi, pi), (-2*pi, 2*pi), (0.09, 0.24)]
                # res = dual_annealing(Global, bounds)
                new = np.array([res.x[0], res.x[3], res.x[6],
                                res.x[1], res.x[4], res.x[7],
                                res.x[2], res.x[5], res.x[8]
                                ]).astype('float64')  # a1 b1 l1 a2 b2 l2 a3 b3 l3
                self.result = tranformation_string(new)

            except Exception as e:
                print(e)


            # jacobian method test
            '''res = jacobian(x0_rosenbrock, dst, R)
            result = tranformation(res)'''
            # end
            return self.result

        if model:
            # a1 a2 a3 b1 b2 b3 l1 l2 l3
            if euler is not None: # using euler angle or not
                R = self.euler_transform(euler)
                n = [R[0][0],R[1][0],R[2][0]]
                a = [R[0][2],R[1][2],R[2][2]]
            else:
                self.dst_dir = n
            # normal type
            '''pos_now = [self.seg[0].alpha*3, self.seg[0].beta,
                       self.seg[3].alpha*3, self.seg[3].beta,
                       self.seg[6].alpha*3, self.seg[6].beta,
                       self.seg[6].length*3
                       ]'''
            # string type
            pos_now = [self.seg[0].alpha * 3,
                       self.seg[3].alpha * 3,
                       self.seg[6].alpha * 3,
                       self.seg[0].beta,
                       self.seg[3].beta,
                       self.seg[6].beta,
                       self.seg[0].length*3,
                       self.seg[3].length*3,
                       self.seg[4].length*3
                       ]
            self.dst_pos = pts
            desired = test_square(pts, pos_now, a, n, R)
        else:
            desired = input

        for i in range(3): #更新目标位置信
            alphaD = desired[3*i]/3
            betaD = desired[1+3*i]
            lengthD = desired[2+3*i]/3

            if self.seg[3*i].UpdateD(alphaD, betaD, lengthD)\
                and self.seg[3*i+1].UpdateD(alphaD, betaD, lengthD)\
                and self.seg[3*i+2].UpdateD(alphaD, betaD, lengthD):
                1
            else:
                print('desired posture is unachievable')

        self.desired = desired
        return self.desired

    def path_tracking(self):
        self.incre = 20

        # 3X3 控制模式
        for i in range(3):
            self.incre_alpha[i] = (result[i] - self.alpha[3 * i] * 3) / self.incre / 3
            self.incre_beta[i] = (result[3 + i] - self.beta[3 * i]) / self.incre
            self.incre_length[i] = (result[6 + i] - self.lm[3 * i] * 3) / self.incre / 3

            if self.incre:
                for i in range(3):
                    for j in range(3):
                        self.soft[3*i+j].alphaD += self.incre_alpha[i]
                        self.soft[3*i+j].lengthD += self.incre_length[i]
                        self.soft[3*ij].betaD += self.incre_beta[i]

    def euler_transform(self, euler): #alpha beta gamma
        R1 = Rotation.from_euler('zyz', euler).as_matrix()
        return R1

if __name__ == '__main__':
    x = [pi / 18, pi/18, pi/18, pi/4, pi/4, pi/4,0.2,0.2,0.2]
    soft1 = softArm(alpha=x[0], beta=x[3], length=x[6])
    soft2 = softArm(alpha=x[1], beta=x[3], length=x[7])
    soft3 = softArm(alpha=x[2], beta=x[3], length=x[8])
    soft4 = softArm(alpha=x[0], beta=x[4], length=x[6])
    soft5 = softArm(alpha=x[1], beta=x[4], length=x[7])
    soft6 = softArm(alpha=x[2], beta=x[4], length=x[8])
    soft7 = softArm(alpha=x[0], beta=x[5], length=x[6])
    soft8 = softArm(alpha=x[1], beta=x[5], length=x[7])
    soft9 = softArm(alpha=x[2], beta=x[5], length=x[8])

    softArms = SoftObject(soft1, soft2, soft3, soft4, soft5, soft6, soft7, soft8, soft9)
    softArms.inverse_kinematic()
