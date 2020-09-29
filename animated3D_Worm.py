# -*- coding: utf-8 -*-
"""
    Animated 3D sinc function
"""
from PyQt5.QtCore import  pyqtSignal, QObject, QThread
from pyqtgraph.Qt import QtCore
from PyQt5.QtWidgets import QWidget, QApplication
import pyqtgraph.opengl as gl
import pyqtgraph as pg
#from angles import normalize_angle
import numpy as np
from numpy import cos, sin, pi, degrees, arccos, sqrt, subtract
from math import acos, atan2
import time
from scipy import linalg

import sys
import threading
#from scipy.spatial.transform import Rotation

try:
    import rospy
    from origarm_ros.msg import *
    ros_OK = 0
except:
    ros_OK = 0

class Signal(QObject):
    ValueSign = pyqtSignal()

class Thread(QThread):
    def __init__(self):
        super(Thread, self).__init__()
        self.ABL = Command_ABL()
        for i in range(9):
            self.ABL.segment[i].L = 0.055

        rospy.init_node('Display_Node', anonymous=True)
        rospy.Subscriber("Cmd_ABL_joy", Command_ABL, self.update_ABL)
        rospy.Subscriber("Cmd_ABL_ik", Command_ABL, self.update_ABL)
        self.rate = rospy.Rate(60) # 10hz

    def run(self):
        while not rospy.is_shutdown():
            self.rate.sleep()

    def update_ABL(self, ABL):
        self.ABL = ABL

class Visualizer(QObject):
    def __init__(self, sf):
        super().__init__()
        #set parameters
        # self.module_initial(sf)

        self.pts = dict()
        self.pts2 = dict()

        self.flag1 = 0
        self.display = 1
        self.multi = 40

        self.dst_pos = [5, 5, 5]
        self.dst_dir = [0, 1, 0]

        self.traces = dict()
        self.lines = dict()
        self.incre_alpha = dict()
        self.incre_beta = dict()
        self.incre_length = dict()
        self.pointer = dict()

        self.normal = [.0, .0, .0]
        self.angle = dict()

        self.incre = 0

        # self.num_seg = sf.num
        self.multi = 1
        self.num_pipe = 1
        self.angle = {0:[0, 1, 0],1:[0, 1, 0],2:[0, 1, 0]}

        self.alpha = dict()
        self.beta = dict()
        self.lm = dict()

        self.nx = {0:[1, 0, 0]}
        self.nz = {0:[0, 0, 1]}
        self.n = [0,0,0]
        self.a = [0,0,0]
        self.o = [0,0,0]
        self.start_3 = 0
        self.oz = 0
        self.euler_beta = 0
        self.euler_alpha = 0
        self.euler_gamma = 0

        self.angle = np.array([0, pi/3, pi*2/3, pi, pi*4/3, pi*5/3]) #
        
        self.x_offset = np.array([1]*8)*self.multi*0.0615
        for i in range(8):
            self.x_offset[i] = pi*(1/8 + i/4)

        self.visualizer_initial()
        self.create_ladder()
        self.create_connection()
        self.create_layer()
        
        self.update()

    def create_connection(self):
        self.connection = [[np.array([0,0,0]) for col in range(8)] for row in range(4)]
        for row in range(4):
            for col in range(8):
                self.connection[row][col] = [1.5*cos(self.x_offset[col]), 1.5*sin(self.x_offset[col]), row*-3]
    

    def create_ladder(self):
        self.ladder = [[0 for col in range(8)] for row in range(3)]
        for row in range(3):
            for col in range(8):
                self.ladder[row][col] = gl.GLLinePlotItem(color=pg.glColor((40+30*(row+1), 100+10*(row+1))),
                                                width=2, antialias=True)
                self.w.addItem(self.ladder[row][col])

    def create_layer(self):
        self.layer = [[0 for col in range(8)] for row in range(4)]
        for row in range(4):
            for col in range(8):
                self.layer[row][col] = gl.GLLinePlotItem(color=pg.glColor((40+30*(row+1), 100+10*(row+1))),
                                                width=2, antialias=True)
                self.w.addItem(self.layer[row][col])

    def create_centralline(self):
        self.centralline = [0 for i in range(3)]
        for segment in range(3):
            self.centralline[segment] = gl.GLLinePlotItem(color=pg.glColor((40+30*(row+1), 100+10*(row+1))),
                                            width=2, antialias=True)
            self.w.addItem(self.line[segment])

    def module_initial(self, sf):
        if ros_OK:
            self.T = Thread()
            self.T.start()
            self.ABL = self.T.ABL
        elif not ros_OK:
            try:
                self.joystick = myJoyStick()
                self.joystick.start()
                self.Sign = Signal()
            except:
                print('joystick have not connected')
        self.Arm = sf

    def visualizer_initial(self):
        # set parameters
        self.w = gl.GLViewWidget()
        self.w.pan(0,0,-10)
        self.w.opts['distance'] = 40
        self.w.setWindowTitle('pyqtgraph example: GLLinePlotItem')
        self.flag = 0

        # create the background grids
        gx = gl.GLGridItem()
        gx.rotate(90, 0, 1, 0)
        gx.translate(-10, 0, -10)
        self.w.addItem(gx)
        gy = gl.GLGridItem()
        gy.rotate(90, 1, 0, 0)
        gy.translate(0, -10, -10)
        self.w.addItem(gy)
        gz = gl.GLGridItem()
        gz.translate(0, 0, 0)
        self.w.addItem(gz)
        self.w.setGeometry(20, 20, 900, 900)

        self.x = np.array([.0]*2)
        self.y = np.array([.0]*2)
        self.z = np.array([.0]*2)

        self.Sign = Signal()

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QApplication.instance().exec_()
            self.alpha = self.alpha

    def update(self):
        self.ROS()
        
        #self.connection = self.Arm.getConnection()

        self.connect_ladder()
        self.connect_layer()

        if self.incre and not ros_OK:
            self.Sign.ValueSign.emit()
            self.incre -= 1

    def connect_ladder(self):
        for num in  range(3):
            for col in range(8):
                for i in range(2):
                    self.x[i] = self.connection[num+i][col][0]
                    self.y[i] = self.connection[num+i][col][1]
                    self.z[i] = self.connection[num+i][col][2]

                line = np.vstack([self.x, self.y, self.z]).transpose()
                self.ladder[num][col].setData(pos=line)

    def connect_layer(self):
        for num in range(4):
            for col in [0,1,2,3,4,5,6,-1]:
                for i in range(2):
                    self.x[i] = self.connection[num][col+i][0]
                    self.y[i] = self.connection[num][col+i][1]
                    self.z[i] = self.connection[num][col+i][2]

                line = np.vstack([self.x, self.y, self.z]).transpose()
                self.layer[num][col].setData(pos=line)

    def create_Line(self): # 扩展接口：
        x = np.array([.0]*19)
        y = np.array([0]*19)
        z = np.array([.0]*19)
        for seg in range(3):
            if not alpha:
                lm = np.linspace(0, lm, 19)
                for i in range(19):
                    z[i] = lm[i]
                    x[i] = 0
            else:
                theta = np.linspace(0, alpha, 19)
                for i in range(19):
                    x[i] = lm/alpha*(1-cos(theta[i]))
                    z[i] = lm/alpha*sin(theta[i])
            transB = np.array([[cos(beta), sin(beta), 0],
                    [-sin(beta), cos(beta), 0],
                    [0, 0, 1]])
            self.line[seg].setData(pos=np.vstack([x, y, z]).transpose().dot(transB))

    def create_circle(self):
        def create(L, r):
            num_res = 200
            pos3 = np.zeros((num_res, num_res, 3))
            pos3[:, :, :2] = np.mgrid[:num_res, :num_res].transpose(1, 2, 0) * [-0.1, 0.1]
            pos3 = pos3.reshape(num_res**2, 3)
            d3 = (pos3 ** 2).sum(axis=1) ** 0.5
            area = L #产生备选点的区域
            ring_res = 0.15 #环粗细
            for i in range(num_res):
                pos3[i * num_res:num_res * (i + 1), 0] = -area + 2*area*i/num_res
                pos3[i * num_res:num_res * (i + 1), 1] = np.linspace(-area, area, num_res)
                pos3[i * num_res:num_res * (i + 1), 2] = 0
            count = 0
            list1 = list()
            rad_ring = r #环圆心距离
            ring = 0.06*10 #环半径
            for i in range(num_res**2):
                if  (ring - ring_res) ** 2 < ((pos3[i, 1]) ** 2 + (pos3[i, 0]-rad_ring) ** 2 )< ring**2  or\
                    (ring - ring_res) ** 2 < ((pos3[i, 1]+rad_ring*0.866) ** 2 + (pos3[i, 0]-rad_ring/2) ** 2)<ring**2 or\
                    (ring - ring_res) ** 2 < ((pos3[i, 1]+rad_ring*0.866) ** 2 + (pos3[i, 0]+rad_ring/2) ** 2)< ring**2  or\
                    (ring - ring_res) ** 2 < ((pos3[i, 1]-rad_ring*0.866) ** 2 + (pos3[i, 0]-rad_ring/2) ** 2)< ring**2  or\
                    (ring - ring_res) ** 2 < ((pos3[i, 1]-rad_ring*0.866) ** 2 + (pos3[i, 0]+rad_ring/2) ** 2)< ring**2  or\
                    (ring - ring_res) ** 2 < ((pos3[i, 1]) ** 2 + (pos3[i, 0]+rad_ring) ** 2)< ring**2  :
                    list1.append(i)
            backup = list()
            for i in list1:
                backup.append(pos3[i])
            return backup

        self.backup = create(L = 3, r = 0.0615*self.multi)
        self.sp = list()
        self.base = list()

        color = {0:pg.glColor(40,20,5),1:pg.glColor(40,20,5),2:pg.glColor(40,20,5),3:pg.glColor(40,40,0),4:pg.glColor(40,40,0),5:pg.glColor(40,40,0),6:pg.glColor(0,40,40),7:pg.glColor(0,40,40),8:pg.glColor(0,40,40)}

        for i in range(self.num_seg+1):
            self.sp.append(gl.GLScatterPlotItem(pos=self.backup, size=0.08, pxMode=False, color = color[1]))
            self.w.addItem(self.sp[i])

    def vector_display(self, vector, pos, num, multipy=0, rgb=0):
        color = [0,pg.glColor(255,0,0),pg.glColor(0,255,0),pg.glColor(0,0,255)]
        if not num in self.pointer.keys():
            if not rgb:
                self.pointer[num] = gl.GLLinePlotItem(color=pg.glColor((40*num, 50)),
                                                width=2, antialias=True)
            else:
                self.pointer[num] = gl.GLLinePlotItem(color=color[rgb],
                                                width=2, antialias=True)
            self.w.addItem(self.pointer[num])
        length = 1
        if multipy:
            length = multipy
        x = np.linspace(0, float(vector[0])*length, 10)
        y = np.linspace(0, float(vector[1])*length, 10)
        z = np.linspace(0, float(vector[2])*length, 10)
        pts = np.vstack([x, y, z]).transpose() + pos
        self.pointer[num].setData(pos=pts)

    def update_circle(self, seg):
        #for seg in range(self.num_seg):
        if seg == self.num_seg:
            #母本
            data = self.backup
            self.sp[seg].setData(pos=np.add(data, self.pts[0][0][:]))
        else:
            vector1 = np.subtract(self.pts[seg][1], self.pts[seg][0])
            vector2 = np.subtract(self.pts[seg][2], self.pts[seg][0])

            result = -np.cross(vector1, vector2)
            # 化为单位向量
            mod = np.sqrt(np.square(result[0])+np.square(result[1])+np.square(result[2]))
            if mod:
                result = np.divide(result, mod)
            # 旋转轴
            
            if not seg:
                data = self.backup
            else:
                data = np.subtract(self.sp[seg - 1].pos, self.pts[seg - 1][18])
    
            spin = -np.array(linalg.expm(np.multiply(self.alpha[seg], self.hat(result))))

            self.sp[seg].setData(pos=np.add(np.dot(data, spin), self.pts[seg][18][:]))

    def hat(self, vector):
        hat = np.array([
            [0, -vector[2], vector[1]],
            [vector[2], 0, -vector[0]],
            [-vector[1], vector[0], 0]
        ])
        return hat

    def transfer(self) :
        # 每个seg 两次旋转
        for seg in range(1, self.num_seg):
            #print(seg)
            angle_x = acos(np.dot(self.nx[seg], self.nx[0]))
            #前一个节点的x轴和后一个节点的x轴叉乘
            axis_x = np.cross(self.nx[seg], self.nx[0])
            mod = np.sqrt(axis_x[0]**2+axis_x[1]**2+axis_x[2]**2)
            if mod:
                axis_x = np.divide(axis_x, mod)
            spin_x = np.array(linalg.expm(np.multiply(angle_x, self.hat(axis_x))))

            nz = np.dot(self.nz[0], spin_x)

            angle_z = arccos(np.clip(np.dot(nz, self.nz[seg]), -1.0, 1.0))
            #对比旋转后结果 不符合即反转
            right = 1
            while right:
                spin_z = np.array(linalg.expm(np.multiply(angle_z, self.hat(self.nx[seg]))))
                check = np.dot(nz, spin_z) - self.nz[seg]
                if -0.005<check[0] <0.005 and -0.005<check[1] <0.005 and -0.005<check[2] <0.005:
                    right = 0
                else:
                    angle_z = -angle_z

            self.pts[seg] = np.dot(np.dot(self.pts[seg], spin_x), spin_z)
            self.pts[seg] += self.pts[seg-1][18]

            self.nx[seg+1] = np.dot(np.dot(self.nx[seg+1], spin_x), spin_z)
            self.nz[seg+1] = np.dot(np.dot(self.nz[seg+1], spin_x), spin_z)

            self.angle[seg] = self.nz[seg+1]

        for i in range(self.num_seg):
            for j in range(19):  # 翻转z坐标y坐标
                self.pts[i][j][1] = -self.pts[i][j][1]
                self.pts[i][j][2] = -self.pts[i][j][2]
            self.traces[i].setData(pos=self.pts[i])
        if self.circle_show:
            1
            # self.update_circle()
    # 基于向量

    def transfer_line(self):
        def hat(vector, theta):
            trans = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            hat = np.array([
                [0, -vector[2], vector[1]],
                [vector[2], 0, -vector[0]],
                [-vector[1], vector[0], 0]
            ])
            result = trans + hat * sin(theta) + (1 - cos(theta)) * hat.dot(hat)
            result[2][2] = cos(theta)
            return result

        transform = np.eye(4)

        a = dict()
        b = dict()
        multi = self.multi
        for i in range(self.num_seg):
            a[i] = self.alpha[i]
            b[i] = self.beta[i]
            self.create_Line(i, self.alpha[i], b[i], self.lm[i] * multi)

        for i in range(1, self.num_seg):
            for j in reversed(range(i)):
                transform = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]). \
                    dot(hat(np.array([-sin(b[j]), cos(b[j]), 0]), a[j])).transpose()
                if self.alpha[j] == 0:
                    base = [
                        0, 0,
                        self.lm[j] * multi
                    ]
                else:
                    base = np.array([self.lm[j] / self.alpha[j] * (1 - cos(self.alpha[j])) * multi,
                            0,
                            self.lm[j] / self.alpha[j] * sin(self.alpha[j]) * multi
                            ]).dot(transB(b[j]))
                self.pts[i] = self.pts[i].dot(transform)
                self.pts[i] += base
                # for j in range(6):
                #     self.pts2[6*i + j] = self.pts2[6*i + j].dot(transform)
                #     self.pts2[6*i + j] += base

        for seg in range(self.num_seg):
            for j in range(19):
                self.pts[seg][j][1] = -self.pts[seg][j][1]
                self.pts[seg][j][2] = -self.pts[seg][j][2]
            self.traces[seg].setData(pos=self.pts[seg])

        for i in range(self.num_seg):
            for j in range(6):
                for k in range(19):
                    self.pts2[6*i + j][k][0] =  self.pts2[6*i + j][k][0] + self.x_offset[j]
                    self.pts2[6*i + j][k][1] = -self.pts2[6*i + j][k][1] - self.y_offset[j]
                    self.pts2[6*i + j][k][2] = -self.pts2[6*i + j][k][2]
                self.lines[6*i + j].setData(pos=self.pts2[6*i + j])

        for seg in range(self.num_seg+1):
            if self.circle_show:
                self.update_circle(seg)

    def move(self):
        if self.incre:
            for i in range(3):
                for j in range(3):
                    self.soft[3*i+j].alpha += self.incre_alpha[i]
                    self.soft[3*i+j].beta += self.incre_beta[i]
                    self.soft[3*i+j].length += self.incre_length[i]
    
    def ROS(self):
        # for i in range(3):
            # if self.T.ABL.segment[2*i].A < 0:
            #     for j in range(3):
            #         self.soft[3*i+j].alpha = -self.T.ABL.segment[2*i].A*2/3
            #         self.soft[3*i+j].beta = normalize_angle(self.T.ABL.segment[2*i].B + pi)
            #         self.soft[3*i+j].length = self.T.ABL.segment[2*i].L*2/3
            # elif self.T.ABL.segment[2*i].A > 0:
            #     for j in range(3):
            #         self.soft[3*i+j].alpha = self.T.ABL.segment[2*i].A*2/3
            #         self.soft[3*i+j].beta = self.T.ABL.segment[2*i].B
            #         self.soft[3*i+j].length = self.T.ABL.segment[2*i].L*2/3
            # elif self.T.ABL.segment[2*i].A == 0:
            #     for j in range(3):
            #         self.soft[3*i+j].alpha = 0
            #         self.soft[3*i+j].beta = 0
            #         self.soft[3*i+j].length = self.T.ABL.segment[2*i].L*2/3
        if ros_OK:
            for i in range(6):
                if self.T.ABL.segment[i].A < 0:
                        self.soft[i].alpha = -self.T.ABL.segment[i].A
                        self.soft[i].beta = normalize_angle(self.T.ABL.segment[i].B + pi)
                        self.soft[i].length = self.T.ABL.segment[i].L
        # self.Sign.ValueSign.emit()

    def inverse_kinematic(self, pts=[0,0,0], n=[0,0,0], a=[0,0,0], euler=0, model=1, input=[],length=2):
        result = self.Arms.inverse_kinematic(pts=pts, n=n, a=a, euler=euler, model=model, input=input,length=length)

        self.incre = 1

        # # 3X3 控制模式
        for i in range(3):
            incre_alpha = result[3*i] - self.alpha[3 * i] * 3
            incre_beta = result[1 + 3*i] - self.beta[3 * i]
            incre_length = result[2 + 3*i] - self.lm[3 * i] * 3
            self.incre_alpha[i] = incre_alpha / self.incre / 3
            self.incre_beta[i] = incre_beta / self.incre
            self.incre_length[i] = incre_length / self.incre / 3

        return 0

    def animation(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(40)

# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    app = QApplication(sys.argv)

    test =  Visualizer(1)
    test.w.show()
    test.animation()

    sys.exit(app.exec_())


