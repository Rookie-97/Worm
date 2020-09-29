# designed for entire worm
from numpy import pi, cos, sin, tan
import numpy as np

class segment():
	def __init__(self):
		self.alphaD = 0
		self.betaD = 0
		self.lengthD = 0.05
		self.length0 = 0
		self.width_start = 0.4
		self.width_end = 0.4

class Worm():
	def __init__(self, layer=4, ladder=3,cir=8):
		self.layer = layer
		self.ladder = ladder
		self.cir = cir
		self.k = 1 # spring index 
		
		self.length_layer = 0.06
		self.length_ladder = 0.05
		
		self.pressure_for_ladder = [[0 for col in range(self.cir)]for row in range(self.layer)]
		self.pressure_for_layer = [0 for row in range(4)]
		
		self.wide_start = 0
		self.wide_end = 0

		self.initialAngle = [pi/8, 3*pi/8, 5*pi/8, 7*pi/8, 9*pi/8, 11*pi/8, 13*pi/8, 15*pi/8]

	def ABL2PD(self, deta_pressure, beta, deta_length):
		# for ladder to aim at the target
		for i in range(self.layer - 1):
			b1 = 2 * self.c1 * (self.lengthD - self.length0) / self.radR / 6
			btemp = self.c1 * self.alphaD / 6
			b2 = btemp * cos(self.betaD)
			b3 = 1.7320508 * btemp * sin(self.betaD)
			self.pressure_for_ladder[i][0] = b1 + b2 * 2
			self.pressure_for_ladder[i][1] = b1 + b2 + b3
			self.pressure_for_ladder[i][2] = b1 - b2 + b3
			self.pressure_for_ladder[i][3] = b1 - b2 * 2
			self.pressure_for_ladder[i][4] = b1 - b2 - b3
			self.pressure_for_ladder[i][5] = b1 + b2 - b3
		 
		return self.pressureD
		
	def PD2ABL(self):
		1
		
	def B2PD(self, deta_pressure, beta, deta_length, wide):
		# for ladder to aim at the target
		for i in range(self.layer - 1):
			for j in range(self.cir):
				self.pressure_for_ladder[i][j] = deta_pressure*sin(self.initialAngle[j]-beta)
		
		# for layer to enlarge 
		for i in range(self.layer):
			self.pressure_for_layer[i] = wide*tan(pi/8)*self.k
		
if __name__ == '__main__':
	Worm = Worm()
	
