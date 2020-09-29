import sys

from PyQt5.QtGui import QPainter, QFont, QColor, QPen
from animated3D_Worm import Visualizer
from SoftWorm import Worm
from math import pi, degrees, acos, atan2, sqrt,radians
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import threading
import time

class Example(QWidget):
    def __init__(self, sf):
        super().__init__()
        self.initUI(sf)

    def initUI(self, sf):
        self.flag = 0
        self.backward_flag = 0
        self.model = True
        self.pre = [.0] * 6

        grid = QGridLayout()
        self.gridLayout = QGridLayout()
        self.gridLayout_2 = QGridLayout()

        self.lab1 = QLabel('0')
        self.lab1.setFixedSize(100, 12)
        self.lab2 = QLabel('0')
        self.lab2.setFixedSize(100, 12)
        self.lab3 = QLabel('0')
        self.lab3.setFixedSize(100, 12)
        self.lab4 = QLabel('0')
        self.lab4.setFixedSize(100, 12)
        self.lab5 = QLabel('0')
        self.lab5.setFixedSize(100, 12)
        self.lab6 = QLabel('0')
        self.lab6.setFixedSize(100, 12)
        self.lab7 = QLabel('0')
        self.lab7.setFixedSize(100, 12)
        self.lab8 = QLabel('0')
        self.lab8.setFixedSize(100, 12)
        self.lab9 = QLabel('0')
        self.lab9.setFixedSize(100, 12)

        grid.addWidget(self.lab1, 0, 0)
        grid.addWidget(self.lab2, 1, 0)
        grid.addWidget(self.lab3, 2, 0)
        grid.addWidget(self.lab4, 0, 2)
        grid.addWidget(self.lab5, 1, 2)
        grid.addWidget(self.lab6, 2, 2)
        grid.addWidget(self.lab7, 0, 4)
        grid.addWidget(self.lab8, 1, 4)
        grid.addWidget(self.lab9, 2, 4)

        self.lab10 = QLabel('x')
        self.lab11 = QLabel('y')
        self.lab12 = QLabel('z')

        grid.addWidget(self.lab10, 3, 0)
        grid.addWidget(self.lab11, 4, 0)
        grid.addWidget(self.lab12, 5, 0)

        self.lab13 = QLabel('nx')
        self.lab14 = QLabel('ny')
        self.lab15 = QLabel('nz')

        grid.addWidget(self.lab13, 3, 2)
        grid.addWidget(self.lab14, 4, 2)
        grid.addWidget(self.lab15, 5, 2)

        self.lab16 = QLabel('nx')
        self.lab17 = QLabel('ny')
        self.lab18 = QLabel('nz')

        self.lab20 = QLabel('P1')
        self.lab21 = QLabel('P2')
        self.lab22 = QLabel('P3')
        self.lab23 = QLabel('P4')
        self.lab24 = QLabel('P5')
        self.lab25 = QLabel('P6')

        grid.addWidget(self.lab20, 0, 6, 1, 3)
        grid.addWidget(self.lab21, 1, 6, 1, 3)
        grid.addWidget(self.lab22, 2, 6, 1, 3)
        grid.addWidget(self.lab23, 3, 6, 1, 3)
        grid.addWidget(self.lab24, 4, 6, 1, 3)
        grid.addWidget(self.lab25, 5, 6, 1, 3)

        self.edit10 = QDoubleSpinBox()  # alpha2 beta2 length2
        self.edit10.setSingleStep(1)
        self.edit10.setMinimum(-1000)
        self.edit10.setMaximum(1000)
        self.edit11 = QDoubleSpinBox()
        self.edit11.setSingleStep(1)
        self.edit11.setMaximum(1000)
        self.edit11.setMinimum(-1000)
        self.edit12 = QDoubleSpinBox()
        self.edit12.setSingleStep(1)
        self.edit12.setMaximum(1000)
        self.edit12.setMinimum(-1000)

        grid.addWidget(self.edit10, 3, 1)
        grid.addWidget(self.edit11, 4, 1)
        grid.addWidget(self.edit12, 5, 1)

        self.edit13 = QDoubleSpinBox()  #
        self.edit13.setSingleStep(1)
        self.edit13.setMaximum(360)
        self.edit13.setMinimum(-360)
        self.edit14 = QDoubleSpinBox()
        self.edit14.setSingleStep(1)
        self.edit14.setMaximum(360)
        self.edit14.setMinimum(-360)
        self.edit15 = QDoubleSpinBox()
        self.edit15.setSingleStep(1)
        self.edit15.setMaximum(360)
        self.edit15.setMinimum(-360)

        self.edit17 = QDoubleSpinBox()  #
        self.edit17.setSingleStep(1)
        self.edit17.setMaximum(360)
        self.edit17.setMinimum(0)
        self.edit18 = QDoubleSpinBox()
        self.edit18.setSingleStep(1)
        self.edit18.setMaximum(360)
        self.edit18.setMinimum(-360)
        self.edit19 = QDoubleSpinBox()
        self.edit19.setSingleStep(1)
        self.edit19.setMaximum(1000)
        self.edit19.setMinimum(-1000)

        self.edit20 = QDoubleSpinBox()
        self.edit20.setSingleStep(1)
        self.edit20.setMaximum(360)
        self.edit20.setMinimum(0)
        self.edit21 = QDoubleSpinBox()
        self.edit21.setSingleStep(1)
        self.edit21.setMaximum(360)
        self.edit21.setMinimum(-360)
        self.edit22 = QDoubleSpinBox()
        self.edit22.setSingleStep(1)
        self.edit22.setMaximum(1000)
        self.edit22.setMinimum(-1000)

        self.edit23 = QDoubleSpinBox()
        self.edit23.setSingleStep(1)
        self.edit23.setMaximum(360)
        self.edit23.setMinimum(0)
        self.edit24 = QDoubleSpinBox()
        self.edit24.setSingleStep(1)
        self.edit24.setMaximum(360)
        self.edit24.setMinimum(-360)
        self.edit25 = QDoubleSpinBox()
        self.edit25.setSingleStep(1)
        self.edit25.setMaximum(1000)
        self.edit25.setMinimum(-1000)

        self.edit27 = QDoubleSpinBox()
        self.edit27.setSingleStep(1)
        self.edit27.setMaximum(200)
        self.edit27.setMinimum(-180)
        self.edit28 = QDoubleSpinBox()
        self.edit28.setSingleStep(1)
        self.edit28.setMaximum(360)
        self.edit28.setMinimum(-360)
        self.edit29 = QDoubleSpinBox()
        self.edit29.setSingleStep(1)
        self.edit29.setMaximum(360)
        self.edit29.setMinimum(-360)

        grid.addWidget(self.edit13, 3, 3)
        grid.addWidget(self.edit14, 4, 3)
        grid.addWidget(self.edit15, 5, 3)

        grid.addWidget(self.edit17, 0, 1)
        grid.addWidget(self.edit18, 1, 1)
        grid.addWidget(self.edit19, 2, 1)

        grid.addWidget(self.edit20, 0, 3)
        grid.addWidget(self.edit21, 1, 3)
        grid.addWidget(self.edit22, 2, 3)

        grid.addWidget(self.edit23, 0, 5)
        grid.addWidget(self.edit24, 1, 5)
        grid.addWidget(self.edit25, 2, 5)

        '''grid.addWidget(self.edit27, 3, 5)
        grid.addWidget(self.edit28, 4, 5)
        grid.addWidget(self.edit29, 5, 5)'''

        self.btn1 = QPushButton('numerical', self)
        self.btn2 = QPushButton('change', self)
        self.btn3 = QPushButton('exit', self)
        # self.btn4 = QPushButton('display', self)

        grid.addWidget(self.btn1, 0 ,9)
        grid.addWidget(self.btn2, 1, 9)
        grid.addWidget(self.btn3, 2, 9)
        # grid.addWidget(self.btn4, 3, 9)

        self.view = Visualizer(sf)
        grid.addWidget(self.view.w, 6, 0, 6, 10)

        self.gridLayout_2.addLayout(self.gridLayout, 0, 1, 1, 1)
        self.gridLayout_2.addLayout(grid, 0, 0, 1, 1)

        self.btn1.clicked.connect(self.changeModel)
        self.btn2.clicked.connect(self.changeValue)
        self.btn3.clicked.connect(sys.exit)
        # self.btn4.clicked.connect(self.display)
        self.view.Sign.ValueSign.connect(self.Update)

        self.setGeometry(0, 50, 800, 800)
        self.setWindowTitle('feedback')
        self.muti = self.view.multi

        self.Update()
        self.setLayout(self.gridLayout_2)

        self.show()
        self.view.animation()
        # self.edit10.setValue(self.view.pts[8][18][0]/self.muti*100)
        # self.edit11.setValue(-self.view.pts[8][18][1]/self.muti*100)
        # self.edit12.setValue(-self.view.pts[8][18][2]/self.muti*100)

        '''self.edit13.setValue(self.view.a[0])
        self.edit14.setValue(self.view.a[1])
        self.edit15.setValue(self.view.a[2])

        self.edit27.setValue(self.view.n[0])
        self.edit28.setValue(self.view.n[1])
        self.edit29.setValue(self.view.n[2])'''

    def start(self):
        if not self.flag:
            self.serial_enable()
        else:
            self.flag = 1

    def changeModel(self):
        self.model = not self.model
        if self.model:
            self.btn1.setText('numerical')
        else:
            self.btn1.setText('forwarding')

    def changeValue(self):
        if self.model:
            pts = [float(self.edit10.text()) / 100, float(self.edit11.text()) / 100, float(self.edit12.text()) / 100]
            euler=[float(self.edit15.text())/180*pi, float(self.edit14.text())/180*pi, float(self.edit13.text())/180*pi]
            length=float(self.edit27.text())
            #re = self.view.backward_position(pts=pts, n=n, a=a)
            re = self.view.inverse_kinematic(pts=pts, euler=euler, length=length)
        else:
            a1 = float(self.edit17.text())/180*pi
            b1 = float(self.edit18.text())/180*pi
            l1 = float(self.edit19.text())/100

            a2 = float(self.edit20.text())/180*pi
            b2 = float(self.edit21.text())/180*pi
            l2 = float(self.edit22.text())/100

            a3 = float(self.edit23.text())/180*pi
            b3 = float(self.edit24.text())/180*pi
            l3 = float(self.edit25.text())/100
            self.view.inverse_kinematic(model=0, input=[a1,b1,l1,a2,b2,l2,a3,b3,l3])

    def display(self):
        self.view.display = not self.view.display

    def Update(self):
        1
        1
        1
        1
        # self.lab1.setText('alpha1 {0:.1f}'.format(3*degrees(self.view.alpha[0])))
        # self.lab2.setText('beta1  {0:.1f}'.format(degrees(self.view.beta[0])))
        # self.lab3.setText('length1 {0:.1f} mm'.format(300*self.view.lm[0]))
        # self.lab4.setText('alpha2 {0:.1f}'.format(3*degrees(self.view.alpha[3])))
        # self.lab5.setText('beta2  {0:.1f}'.format(degrees(self.view.beta[3])))
        # self.lab6.setText('length2 {0:.0f} mm'.format(300*self.view.lm[3]))
        # self.lab7.setText('alpha3 {0:.1f}'.format(3*degrees(self.view.alpha[6])))
        # self.lab8.setText('beta3  {0:.1f}'.format(degrees(self.view.beta[6])))
        # self.lab9.setText('length3 {0:.1f} mm'.format(300*self.view.lm[6]))

        # self.edit17.setValue(degrees(self.view.alpha[0]*3))
        # self.edit18.setValue(degrees(self.view.beta[0]))
        # self.edit19.setValue(self.view.lm[0]*300)

        # self.edit20.setValue(degrees(self.view.alpha[3]*3))
        # self.edit21.setValue(degrees(self.view.beta[3]))
        # self.edit22.setValue(self.view.lm[3]*300)

        # self.edit23.setValue(degrees(self.view.alpha[6]*3))
        # self.edit24.setValue(degrees(self.view.beta[6]))
        # self.edit25.setValue(self.view.lm[6]*300)

        # self.lab10.setText('x {0:.2f}mm '.format(self.view.pts[8][18][0]/self.muti*100))
        # self.lab11.setText('y {0:.2f}mm '.format(-self.view.pts[8][18][1]/self.muti*100))
        # self.lab12.setText('z {0:.2f}mm '.format(-self.view.pts[8][18][2]/self.muti*100))

        # self.lab13.setText('alpha {0:.2f} '.format(degrees(self.view.euler_alpha)))
        # self.lab14.setText('beta {0:.2f} '.format(degrees(self.view.euler_beta)))
        # self.lab15.setText('gamma {0:.2f} '.format(degrees(self.view.euler_gamma)))

        # self.lab20.setText('seg1.P1 {0:.1f} kPa seg2.P1.{1:.1f} kPa seg3.P1 {2:.1f} kPa'.format(self.view.soft[0].pressureD[0]/1000, self.view.soft[3].pressureD[0]/1000, self.view.soft[6].pressureD[0]/1000))
        # self.lab21.setText('seg1.P2 {0:.1f} kPa seg2.P2.{1:.1f} kPa seg3.P2 {2:.1f} kPa'.format(self.view.soft[0].pressureD[1]/1000, self.view.soft[3].pressureD[1]/1000, self.view.soft[6].pressureD[1]/1000))
        # self.lab22.setText('seg1.P3 {0:.1f} kPa seg2.P3.{1:.1f} kPa seg3.P3 {2:.1f} kPa'.format(self.view.soft[0].pressureD[2]/1000, self.view.soft[3].pressureD[2]/1000, self.view.soft[6].pressureD[2]/1000))
        # self.lab23.setText('seg1.P4 {0:.1f} kPa seg2.P4.{1:.1f} kPa seg3.P4 {2:.1f} kPa'.format(self.view.soft[0].pressureD[3]/1000, self.view.soft[3].pressureD[3]/1000, self.view.soft[6].pressureD[3]/1000))
        # self.lab24.setText('seg1.P5 {0:.1f} kPa seg2.P5.{1:.1f} kPa seg3.P5 {2:.1f} kPa'.format(self.view.soft[0].pressureD[4]/1000, self.view.soft[3].pressureD[4]/1000, self.view.soft[6].pressureD[4]/1000))
        # self.lab25.setText('seg1.P6 {0:.1f} kPa seg2.P6.{1:.1f} kPa seg3.P6 {2:.1f} kPa'.format(self.view.soft[0].pressureD[5]/1000, self.view.soft[3].pressureD[5]/1000, self.view.soft[6].pressureD[5]/1000))

if __name__ == '__main__':
    app = QApplication(sys.argv)

    Arm = Worm()
    ex = Example(Arm)

    sys.exit(app.exec_())