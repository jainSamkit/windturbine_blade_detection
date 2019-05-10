#!/usr/bin/env python

import rospy
import sys
from sensor_msgs.msg import Image
from std_msgs.msg import String,Int32,Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import time
import glob
import matplotlib.pyplot as plt
from skimage import exposure
import imutils
from numpy import linalg
from sklearn.cluster import KMeans
import math
import operator

global first_time
first_time = True

# 1. Make way for two processes ,one clustering and another prediction.
# 2. In clustering section ,initialise the hough bundler once and keep classifying every 2 secs. Update the current classifier after every run.
# 3. In prediction module ,use the current classifier to predict the cluster for each pixel and publish it to the topic.Initialise a different hough bundler object for
#     prediction seperately.


class HoughBundler:
	'''Clasterize and merge each cluster of cv2.HoughLinesP() output
	a = HoughBundler()
	foo = a.process_lines(houghP_lines, binary_image)
	'''

	def __init__(self):
		self.min_distance_to_merge = 10
		self.min_angle_to_merge = 5

	def get_orientation(self, line):
		'''get orientation of a line, using its length
		https://en.wikipedia.org/wiki/Atan2
		'''
		orientation = math.atan2(abs((line[0] - line[2])), abs((line[1] - line[3])))
		return math.degrees(orientation)

	def checker(self, line_new, groups, min_distance_to_merge, min_angle_to_merge):
		'''Check if line have enough distance and angle to be count as similar
		'''
		for group in groups:
			# walk through existing line groups
			for line_old in group:
				# check distance
				if self.get_distance(line_old, line_new) < min_distance_to_merge:
					# check the angle between lines
					orientation_new = self.get_orientation(line_new)
					orientation_old = self.get_orientation(line_old)
					# if all is ok -- line is similar to others in group
					if abs(orientation_new - orientation_old) < min_angle_to_merge:
						group.append(line_new)
						return False
		# if it is totally different line
		return True

	def DistancePointLine(self, point, line):
		"""Get distance between point and line
		http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
		"""
		px, py = point
		x1, y1, x2, y2 = line

		def lineMagnitude(x1, y1, x2, y2):
			'Get line (aka vector) length'
			lineMagnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
			return lineMagnitude

		LineMag = lineMagnitude(x1, y1, x2, y2)
		if LineMag < 0.00000001:
			DistancePointLine = 9999
			return DistancePointLine

		u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
		u = u1 / (LineMag * LineMag)

		if (u < 0.00001) or (u > 1):
			#// closest point does not fall within the line segment, take the shorter distance
			#// to an endpoint
			ix = lineMagnitude(px, py, x1, y1)
			iy = lineMagnitude(px, py, x2, y2)
			if ix > iy:
				DistancePointLine = iy
			else:
				DistancePointLine = ix
		else:
			# Intersecting point is on the line, use the formula
			ix = x1 + u * (x2 - x1)
			iy = y1 + u * (y2 - y1)
			DistancePointLine = lineMagnitude(px, py, ix, iy)

		return DistancePointLine

	def get_distance(self, a_line, b_line):
		"""Get all possible distances between each dot of two lines and second line
		return the shortest
		"""
		dist1 = self.DistancePointLine(a_line[:2], b_line)
		dist2 = self.DistancePointLine(a_line[2:], b_line)
		dist3 = self.DistancePointLine(b_line[:2], a_line)
		dist4 = self.DistancePointLine(b_line[2:], a_line)

		return min(dist1, dist2, dist3, dist4)

	def merge_lines_pipeline_2(self, lines):
		'Clusterize (group) lines'
		groups = []  # all lines groups are here
		# Parameters to play with
		min_distance_to_merge = 15
		min_angle_to_merge = 5
		# first line will create new group every time
		groups.append([lines[0]])
		# if line is different from existing gropus, create a new group
		for line_new in lines[1:]:
			if self.checker(line_new, groups, self.min_distance_to_merge, self.min_angle_to_merge):
				groups.append([line_new])

		return groups

	def merge_lines_segments1(self, lines):
		"""Sort lines cluster and return first and last coordinates
		"""
		orientation = self.get_orientation(lines[0])

		# special case
		if(len(lines) == 1):
			return [lines[0][:2], lines[0][2:]]

		# [[1,2,3,4],[]] to [[1,2],[3,4],[],[]]
		points = []
		for line in lines:
			points.append(line[:2])
			points.append(line[2:])
		# if vertical
		if 45 < orientation < 135:
			#sort by y
			points = sorted(points, key=lambda point: point[1])
		else:
			#sort by x
			points = sorted(points, key=lambda point: point[0])

		# return first and last point in sorted group
		# [[x,y],[x,y]]
		return [points[0], points[-1]]

	def process_lines(self, lines):
		'''Main function for lines from cv.HoughLinesP() output merging
		for OpenCV 3
		lines -- cv.HoughLinesP() output
		img -- binary image
		'''

		time1 = time.time()
		lines_x = []
		lines_y = []
		# for every line of cv2.HoughLinesP()
		for line_i in [l[0] for l in lines]:
				orientation = self.get_orientation(line_i)
				# if vertical
				if 45 < orientation < 135:
					lines_y.append(line_i)
				else:
					lines_x.append(line_i)

		lines_y = sorted(lines_y, key=lambda line: line[1])
		lines_x = sorted(lines_x, key=lambda line: line[0])
		merged_lines_all = []

		# for each cluster in vertical and horizantal lines leave only one line
		for i in [lines_x, lines_y]:
				if len(i) > 0:
					groups = self.merge_lines_pipeline_2(i)
					merged_lines = []
					for group in groups:
						merged_lines.append(self.merge_lines_segments1(group))

					merged_lines_all.extend(merged_lines)

		time2 = time.time()
		print ""
		print "The time taken to process merged lines is:",time2-time1

		return merged_lines_all



class blade_detector():

	def __init__(self):
		self.bridge = CvBridge()
		self.init_pub_sub()
		self.bgr_image = None
		self.actual_blade_slope = None

		self.down_sample_img_c = 2  #down sample FULL HD image to run clustering.
		self.down_sample_img_p = 2  #down sample FULL HD image to run prediction.

		#   The HoughBundler() class is used to merge close lines (by distance and slope) together.
		self.clustering_hough_bundler = HoughBundler()
		self.prediction_hough_bundler = HoughBundler()

		#   Values to tweak. These parameters are used to merge hough lines (by distance
		self.clustering_hough_bundler.min_distance_to_merge = 20
		self.clustering_hough_bundler.min_angle_to_merge = 5

		self.prediction_hough_bundler.min_angle_to_merge = 5
		self.prediction_hough_bundler.min_distance_to_merge = 15

		#   Clustering params
		self.n_clusters = 4 #Number of clusters to perform KMeans clustering.
		self.if_already_training = False
		self.clt = None
		self.clt_label = None

		#   rospy timer funtions
		# rospy.Timer(rospy.Duration(5), self.trainer)
		rospy.Timer(rospy.Duration(0.5),self.classifier)

		#   parameters to save images in a folder
		self.curr = 0
		self.folder_name = "/home/darth/Desktop/windturbine/X5S_Imagery/t1/b1/hp/"

		#hough lines parameters
		self.minLineLength = 150
		self.maxLineGap = 90

		#Cluster Info
		self.cluster_info = {}
		self.curr_slope = None



	def raw_img_callback(self,data):
		try:
			self.bgr_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
			self.perform_training()
			# print self.bgr_image
		except CvBridgeError, e:
			print e

	def init_pub_sub(self):
		self.rgb_dji = rospy.Subscriber('/flytos/dji_sdk/main_camera_images',Image,self.raw_img_callback,queue_size = 1)
		self.blade_pose_data = rospy.Publisher('/flytos/blade_detection/blade_pose',Float32MultiArray,queue_size = 1)
		self.color_img_topic = rospy.Publisher('/flytos/blade_detection/blade_pose_image', Image, queue_size=10)
		# self.trigger_detection = rospy.Service('blade_detection/trigger', SurfaceDetection, self.handle_trigger_profiling)

	def init_services(self):
		self.slope_req_handler = rospy.Service('/flytos/blade_detection/set_slope',SET_BLADE_SLOPE,self.handle_training)

	def pre_process(self,image):

		image = cv2.GaussianBlur(image, (5, 5), 0)
		kernel = np.ones((5, 5), np.uint8)
		erosion = cv2.erode(image, kernel, iterations=1)
		dilation = cv2.dilate(erosion, kernel, iterations=2)

		return dilation




	def check_sanity(self,img,num_pixels):
		
		img_list = img.flatten().tolist()
		white_pixel_count = np.sum(map(lambda v: v==255,img_list))
		# print "blade_pixel_count:",num_pixels
		# print "white pixel count:",white_pixel_count
		if num_pixels/white_pixel_count > 0.4:
			return True
		else:
			return False


	def checkpDist(self,line1,line2):
		
		line1_x1 = line1[0][0]
		line1_y1 = line1[0][1]
		line1_x2 = line1[1][0]
		line1_y2 = line1[1][1]
		
		line2_x1 = line2[0][0]
		line2_y1 = line2[0][1]
		line2_x2 = line2[1][0]
		line2_y2 = line2[1][1]
		
		line1_dx = line1_x2 - line1_x1
		line1_dy = line1_y2 - line1_y1
		
		line2_dx = line2_x2 - line2_x1
		line2_dy = line2_y2 - line2_y1
		
		line1_length = math.sqrt(line1_dy**2 + line1_dx**2)
		line2_length = math.sqrt(line2_dy**2 + line2_dx**2)
		
		
		
		if abs(line1_dx) < 0.01:
			if line1_dx <0:
				line1_dx = -0.01
			else:
				line1_dx = 0.01
		
		if abs(line2_dx) < 0.01:
			if line2_dx <0:
				line2_dx = -0.01
			else:
				line2_dx= 0.01
		
		m1 = line1_dy/float(line1_dx)
		m2 = line2_dy/float(line2_dx)
		
		c1 = line1_y1 - float(m1)*line1_x1
		c2 = line2_y1 - float(m2)*line2_x1
		
	#     print "lines:",line1,line2
	#     print m1,m2,c1,c2
		
				
		if line1_length >line2_length:
			m = m1
			c = c1
			x_mid = (line2_x1 + line2_x2)/2.0
			y_mid = (line2_y1 + line2_y2)/2.0

			pDist = (abs(y_mid - m*x_mid - c))/float(math.sqrt(1 + m**2))
		else:
			m = m2
			c = c2
			x_mid = (line1_x1 + line1_x2)/2.0
			y_mid = (line1_y1 + line1_y2)/2.0

			pDist = (abs(y_mid - m*x_mid - c))/float(math.sqrt(1 + m**2))



		# pDist = abs(c1-c2)/float(math.sqrt(1 + m**2))
	#     print ""
		
	#     print "distance inside checkpDist:",pDist
		return pDist



	# def plot_clusters(self, clt,image, num_clusters):

	# 	print ""
	# 	time1 = time.time()
	# 	img_copy = image.copy()
	# 	img = image.reshape((image.shape[0] * image.shape[1], 3))
	# 	labels = clt.predict(img)

	# 	images = []

	# 	for i in range(num_clusters):
	# 		img = np.zeros((img_copy.shape[0], img_copy.shape[1], 3), np.uint8)
	# 		images.append(img)

	# 	for i in range(len(labels)):
	# 		row = i // img_copy.shape[1]
	# 		col = i % img_copy.shape[1]

	# 		label = int(labels[i])
	# 		images[label][row, col, :] = (255, 255, 255)
	# 	time2 = time.time()
	# 	print "time taken to plot clusters are:",time2 - time1

	# 	return images

	def plot_clusters(self, clt,image, num_clusters):

		print ""
		time1 = time.time()
		img_copy = image.copy()
		img = image.reshape((image.shape[0] * image.shape[1], 3))
		labels = clt.predict(img)
		img_width = img_copy.shape[1]

		images = []

		for i in range(num_clusters):
			img = np.zeros((img_copy.shape[0], img_copy.shape[1]), np.uint8)
			cluster = np.where(labels==i)
			img = img.ravel()
			img[cluster] = 255
			img = img.reshape(img_copy.shape[0], img_copy.shape[1])
			img = np.tile(img[:,:,None],[1,1,3])
			images.append(img)

		# cluster_0 = np.where(labels==0)
		# images[0] = images[0].ravel()
		# images[0][cluster_0] = 255
		# images[0] = images[0].reshape(img_copy.shape[0], img_copy.shape[1])
		# images[0] = np.tile(images[0][:, :, None], [1, 1, 3])

		# cluster_1 = np.where(labels==1)
		# images[1] = images[1].ravel()
		# images[1][cluster_0] = 255
		# images[1] = images[1].reshape(img_copy.shape[0], img_copy.shape[1])
		# images[1] = np.tile(images[1][:, :, None], [1, 1, 3])

		# cluster_2 = np.where(labels==2)
		# images[0] = images[0].ravel()
		# images[0][cluster_0] = 255
		# images[0] = images[0].reshape(img_copy.shape[0], img_copy.shape[1])
		# images[0] = np.tile(images[0][:, :, None], [1, 1, 3])

		# for i in range(len(labels)):
		# 	row = i // img_copy.shape[1]
		# 	col = i % img_copy.shape[1]

		# 	label = int(labels[i])
		# 	images[label][row, col, :] = (255, 255, 255)
		time2 = time.time()
		print "time taken to plot clusters are:",time2 - time1

		return images



	def find_lines_slope(self,hough_lines, slope):
		req_lines = []
		#     slope = abs(slope)
		for line in hough_lines:
			for x1, y1, x2, y2 in line:
				dy = y2 - y1
				dx = x2 - x1

				if abs(dx) < 0.01:
					if dx < 0:
						dx = -0.01
					else:
						dx = 0.01

				slope_req = math.atan2(dy, dx) * 180 / math.pi
				#             print slope_req
				if slope - 10 < slope_req < slope + 10:
					#                 print slope,slope_req
					req_lines.append(line)

				if slope_req > 0:
					slope_req -= 180
				elif slope_req < 0:
					slope_req += 180

				if slope - 10 < slope_req < slope + 10:
					#                 print "trgt"
					req_lines.append(line)

		return req_lines



	def check_sign(self,m, c, min_x_y, max_x_y):
		xmin = min_x_y[0]
		xmax = max_x_y[0]
		ymax = max_x_y[1]
		ymin = min_x_y[1]
		x = (xmin + xmax) / float(2.0)
		y = (ymin + ymax) / float(2.0)

		check1 = y - m[0] * x - c[0]
		check2 = y - m[1] * x - c[1]

		if abs(check1)<0.01:
			if check1<0:
				check1 = -0.01
			else:
				check1 = 0.01

		if abs(check2)<0.01:
			if check2<0:
				check2 = -0.01
			else:
				check2 = 0.01

		check1 = check1 / abs(check1)
		check2 = check2 / abs(check2)

		if check1 * check2 > 0:
			return True
		else:
			return False



	def check_same_side(self,img, y, x, m, c, min_x_y, max_x_y, sign_to_check):

		min_x = min_x_y[0]
		max_x = max_x_y[0]
		min_y = min_x_y[1]
		max_y = max_x_y[1]

		#     print "min_x:",min_x,"min_y:",min_y,"max_x:",max_x,"max_y:",max_y
		if not (min_x <= x <= max_x and min_y <= y <= max_y):
			# print "outside boundaries---------------:",y,x
			return 0

		check1 = y - m[0] * x - c[0]
		check2 = y - m[1] * x - c[1]

		if check1 != 0:
			check1 = check1 / abs(check1)
		if check2 != 0:
			check2 = check2 / abs(check2)
		# #     print "check1:",check1
		# #     print "check2:",check2

		product = check1 * check2

		if sign_to_check is True:
			if product >= 0:
				if img[y, x] == 255:
					return 1
				else:
					return 0
			else:
				return 0
		elif sign_to_check is False:
			if product <= 0:
				# print "okay1"
				if img[y, x] == 255:
					return 1
				else:
					# print "okay2"
					return 0
			else:
				return 0


	def check_same_side_center(self,img, y, x, m, c, min_x_y, max_x_y, sign_to_check):

		min_x = min_x_y[0]
		max_x = max_x_y[0]
		min_y = min_x_y[1]
		max_y = max_x_y[1]

		#     print "min_x:",min_x,"min_y:",min_y,"max_x:",max_x,"max_y:",max_y
		if not (min_x <= x <= max_x and min_y <= y <= max_y):
			# print "outside boundaries---------------:",y,x
			return 0

		check1 = y - m[0] * x - c[0]
		check2 = y - m[1] * x - c[1]

		if check1 != 0:
			check1 = check1 / abs(check1)
		if check2 != 0:
			check2 = check2 / abs(check2)
		# #     print "check1:",check1
		# #     print "check2:",check2

		product = check1 * check2

		if sign_to_check is True:
			if product >= 0:
				return 1
			else:
				return 0
		elif sign_to_check is False:
			if product <= 0:
				return 1
			else:
				return 0



	def check_point_side(self,lines,point,img):

		center_x = point[0]
		center_y = point[1]
		m = []
		c = []
		min_x = None
		min_y = None
		max_x = None
		max_y = None

		# for line 1
		x1 = lines[0][0][0]
		x2 = lines[0][1][0]
		y1 = lines[0][0][1]
		y2 = lines[0][1][1]

		min_x = min(x1, x2)
		max_x = max(x1, x2)

		min_y = min(y1, y2)
		max_y = max(y1, y2)

		dx1 = x2 - x1
		if abs(dx1) < 0.01:
			if dx1 < 0:
				dx1 = -0.01
			else:
				dx1 = 0.01
		dy1 = y2 - y1
		#     print dy1,dx1
		line1_m = dy1 / float(dx1)
		line1_c = y1 - line1_m * x1
		m.append(line1_m)
		c.append(line1_c)

		# for line 2
		x1 = lines[1][0][0]
		x2 = lines[1][1][0]
		y1 = lines[1][0][1]
		y2 = lines[1][1][1]

		min_x = min(min_x, min(x1, x2))
		max_x = max(max_x, max(x1, x2))

		min_y = min(min_y, min(y1, y2))
		max_y = max(max_y, max(y1, y2))

		min_x_y = [min_x, min_y]
		max_x_y = [max_x, max_y]

		dy1 = y2 - y1
		dx1 = x2 - x1
		if abs(dx1) < 0.01:
			if dx1 > 0:
				dx1 = 0.01
			else:
				dx1 = -0.01
		#     print dy1,dx1
		line2_m = dy1 /float(dx1)
		line2_c = y1 - line2_m * x1
		m.append(line2_m)
		c.append(line2_c)

		img_copy = img.copy()

		#     img_copy = img_copy[min_y:max_y,min_x:max_x]
		width = img.shape[1]
		height = img.shape[0]

		ret, img_copy = cv2.threshold(img_copy, 127, 255, cv2.THRESH_BINARY)
		
		sign_to_check = self.check_sign(m, c, min_x_y, max_x_y)

		# print "The sign to check is-----------------------:",sign_to_check

		check_inner_points = self.check_same_side_center(img_copy, center_y, center_x, m, c, min_x_y, max_x_y, sign_to_check)

		# print "The check inner point is:----------------",check_inner_points

		return check_inner_points



	def count_pixels(self,lines, img):

		#     print img.shape
		m = []
		c = []
		min_x = None
		min_y = None
		max_x = None
		max_y = None

		# for line 1
		x1 = lines[0][0][0]
		x2 = lines[0][1][0]
		y1 = lines[0][0][1]
		y2 = lines[0][1][1]

		min_x = min(x1, x2)
		max_x = max(x1, x2)

		min_y = min(y1, y2)
		max_y = max(y1, y2)

		dx1 = x2 - x1
		if abs(dx1) < 0.01:
			if dx1 < 0:
				dx1 = -0.01
			else:
				dx1 = 0.01
		dy1 = y2 - y1
		#     print dy1,dx1
		line1_m = dy1 / float(dx1)
		line1_c = y1 - line1_m * x1
		m.append(line1_m)
		c.append(line1_c)

		# for line 2
		x1 = lines[1][0][0]
		x2 = lines[1][1][0]
		y1 = lines[1][0][1]
		y2 = lines[1][1][1]

		min_x = min(min_x, min(x1, x2))
		max_x = max(max_x, max(x1, x2))

		min_y = min(min_y, min(y1, y2))
		max_y = max(max_y, max(y1, y2))

		min_x_y = [min_x, min_y]
		max_x_y = [max_x, max_y]

		dy1 = y2 - y1
		dx1 = x2 - x1
		if abs(dx1) < 0.01:
			if dx1 > 0:
				dx1 = 0.01
			else:
				dx1 = -0.01
		#     print dy1,dx1
		line2_m = dy1 /float(dx1)
		line2_c = y1 - line2_m * x1
		m.append(line2_m)
		c.append(line2_c)

		# print "m:", math.atan(m[0]) * 180 / math.pi, math.atan(m[1]) * 180 / math.pi
		# print "c:", c

		# bounding a rectangle using two lines

		img_copy = img.copy()

		#     img_copy = img_copy[min_y:max_y,min_x:max_x]
		width = img.shape[1]
		height = img.shape[0]

		#     print img_copy
		#     cv2.rectangle(img_copy,(min_x,min_y),(max_x,max_y),(255,255,255),5)
		ret, img_copy = cv2.threshold(img_copy, 127, 255, cv2.THRESH_BINARY)
		t = img_copy.shape
		a = t[0] * t[1]
		# print "shape of the array:", a
		b = np.sum(img_copy[:, :] == 0)
		# print "ratio:", b / float(a)

		img_list = img_copy.flatten().tolist()
		#     plt.imshow(img_copy,cmap='gray')
		#     plt.show()
		sign_to_check = self.check_sign(m, c, min_x_y, max_x_y)

		check_inner_points = map(lambda v: self.check_same_side(img_copy, int(v / width), int(v % width), m, c, min_x_y, max_x_y, sign_to_check),range(len(img_list)))		

		#     print t
		#     plt.imshow(img_copy,cmap='gray')
		#     plt.show()
		img_count = np.zeros((img_copy.shape[0], img_copy.shape[1]), np.uint8)
		img_count = img_count.ravel()


		a = np.where(check_inner_points==1)
		b = np.where(check_inner_points==0)

		img_count[a] = 255
		img_count[b] = 0

		img_count = img_count.reshape(img_copy.shape[0],img_copy.shape[1])
		
		img_count = np.tile(img_count[:,:,None],[1,1,3])

		# for i in range(len(t)):
		# 	row = int(i / width)
		# 	col = int(i % width)
		# 	if t[i] == 1:
		# 		img_count[row, col, :] = (255, 255, 255)
		# 	else:
		# 		img_count[row, col, :] = (0, 0, 255)
		return np.sum(check_inner_points), img_count





	def detect_hough_lines(self,img):
		#     gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
		#     kernel = np.ones((5,5),np.uint8)
		#     gray_img = cv2.dilate(gray_img,kernel,iterations = 2)
		#     edges = cv2.Canny(gray_img,10,150,apertureSize = 3)

		# #     edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
		#     edges = cv2.dilate(edges,kernel,iterations = 2)
		#     edges = cv2.erode(edges ,kernel,iterations =1)
		#     plt.imshow(edges,cmap='gray')
		#     plt.show()


		lines = cv2.HoughLinesP(img, 1, np.pi / 180, 200, self.minLineLength, self.maxLineGap)
		hough_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
		#     print lines.shape
		if lines is not None:
			for line in lines:
				for x1, y1, x2, y2 in line:
					cv2.line(hough_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
		return hough_img, lines



	# def find_cluster(self,merged_lines, img):
	# 	max_index = None
	# 	max_value = None

	# 	second_max_value = None
	# 	second_max_index = None

	# 	for i in range(len(merged_lines)):
	# 		line = merged_lines[i]
	# 		x1 = line[0][0]
	# 		y1 = line[0][1]
	# 		x2 = line[1][0]
	# 		y2 = line[1][1]

	# 		length = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
	# 		#         print "length:",length
	# 		#         print "corresponding line:",merged_lines[i]
	# 		if max_index is None:
	# 			max_index = i
	# 			max_value = length
	# 		else:
	# 			if length > max_value:

	# 				second_max_index = max_index
	# 				second_max_value = max_value

	# 				max_index = i
	# 				max_value = length

	# 			elif second_max_value < length < max_value:
	# 				second_max_index = i
	# 				second_max_value = length
	# 	#         print "max_index:"," ",max_index," ","second index:",second_max_index

	# 	lines = []
	# 	if max_index is None or second_max_index is None:
	# 		#         print "No indexes updated"
	# 		return 0, lines, None
	# 	elif max_index != second_max_index:
	# 		lines.append(merged_lines[max_index])
	# 		lines.append(merged_lines[second_max_index])
	# 		num_pixels, img_count = self.count_pixels(lines, img)
	# 		print "two indexes updated:", lines
	# 		return num_pixels, lines, img_count
	# 	elif max_index == second_max_index:
	# 		lines.append(merged_lines[max_index])
	# 		#         print "only one index updated:",lines
	# 		return 0, lines, None


	def find_cluster(self,merged_lines,img):
	
		max_index = None
		max_value = None
		
		line_lengths = []
		line_slopes = []
		for i in range(len(merged_lines)):
			line =merged_lines[i]
			x1 = line[0][0]
			y1 = line[0][1]
			x2 = line[1][0]
			y2 = line[1][1]
			
			length = math.sqrt((x1-x2)**2 + (y1-y2)**2)
			line_lengths.append(length)
			if max_index is None:
				max_index = i
				max_value = length
			else:
				if length>max_value:
					max_index = i
					max_value = length
					
			dy = y2-y1
			dx = x2-x1
			if abs(dx)<0.01:
				if dx<0:
					dx = -0.01
				else:
					dx = 0.01
			
			slope = math.atan2(dy,dx)
			if slope <0:
				slope+=math.pi
			
			slope = slope*180/math.pi
			line_slopes.append(slope)
			
		costs = {}
		lines=[]
		
		if max_index is not None:
			
			max_line_slope = line_slopes[max_index]
			height = img.shape[0]
			width = img.shape[1]
			# print "height_width",height,width

			for i in range(len(merged_lines)):
				if i!=max_index:
					length_cost = line_lengths[i]/float(line_lengths[max_index])
					slope_cost = abs(line_slopes[i] - line_slopes[max_index])/20.0
					slope_cost = math.exp(-1*slope_cost) * 2
					pDist = self.checkpDist(merged_lines[i],merged_lines[max_index])
					# print "pDist:",pDist
					if 45<max_line_slope<135:
						# print "inside_width"
					 
						slope = (max_line_slope *math.pi)/180.0
						dist_cost = abs(pDist * math.sin(slope)/float(width))
					else:
						# print "inside height"
						slope = (max_line_slope *math.pi)/180.0
						dist_cost = abs(pDist*math.cos(slope)/float(height))
					
					# print "slope_cost:",slope_cost,"length_cost:",length_cost,"dist_cost:",dist_cost
					total_cost = slope_cost + dist_cost + length_cost
					# total_cost = slope_cost + dist_cost
					costs[i] = total_cost
			
			# print "costs:",costs
			# print "lengths",line_lengths
			# print ""
			
			if len(costs)>0:
	#             max_cost_index = np.argmax(costs) + 1
				max_cost_index = max(costs.iteritems(), key=operator.itemgetter(1))[0]
				lines.append(merged_lines[max_index])
				lines.append(merged_lines[max_cost_index])

				time1 = time.time()
				num_pixels,img_count = self.count_pixels(lines,img)
				time2 = time.time()
				print "time taken to count the number of pixels in the image is:",time2-time1
				return num_pixels,lines,img_count
			else:
				lines.append(merged_lines[max_index])
				return 0,lines,None
		else:
			return 0,lines,None
				




	def detect_blade_cluster(self,image_cluster):
		# print "cluster detection running"
		if self.clt is not None:
			# f, ax = plt.subplots(len(image_cluster), 5, figsize=(50, 50))
			num_pixels = []
			# print len(image_cluster)
			cluster_sanity = {}

			for i in range(len(image_cluster)):
				img = image_cluster[i]
				gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
				ret, gray_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
				gray_img = self.pre_process(gray_img)
				# ax[i][0].imshow(gray_img, cmap='gray')
				#     plt.show()

				#     edge_img = cv2.Canny(gray_img,)
				edges = cv2.Canny(gray_img, 10, 150, apertureSize=3)
				#     ax[i][1].imshow(edges,cmap='gray')
				#     plt.imshow(edges,cmap='gray')
				#     plt.show()

				kernel = np.ones((5, 5), np.uint8)
				dilated_img = cv2.dilate(edges, kernel, iterations=2)
				erosion_img = cv2.erode(dilated_img, kernel, iterations=1)
				#     ax[i][2].imshow(erosion_img,cmap='gray')

				#     print erosion_img.shape
				hough_img, hough_lines = self.detect_hough_lines(erosion_img)
				# ax[i][1].imshow(hough_img, cmap='gray')

				#     print hough_lines
				slopes = []

				if hough_lines is not None:
					for line in hough_lines:
						for x1, y1, x2, y2 in line:
							dy = y2 - y1
							dx = x2 - x1

							if abs(dx) < 0.01:
								if dx < 0:
									dx = -0.01
								else:
									dx = 0.01

							slope = math.atan2(dy, dx) * 180.0 / math.pi
							slopes.append(slope)

					# for x in range(len(slopes)):
					# 	if slopes[x] < 0:
					# 		slopes[x]+=180

					slope = -85

					# if slope<0:
					# 	slope+=180

					print "slopes:",slopes

					slopes_array = np.asarray(slopes)
					print "Slopes_array:",slopes_array.shape

					interval_bins = [-180,-150,-120,-90,-60,-30,0,30,60,90,120,150,180]
					# N, bins, patches = ax[0].hist(depth_array, edgecolor='white', linewidth=1, bins=interval_bins)

					N, bins, patches = plt.hist(slopes_array, edgecolor='white', linewidth=1,
														  bins=interval_bins)
					max_slope_index = np.argmax(N)

					a = np.digitize(slopes_array,bins)
					a = np.asarray(a)

					req_slope_indexes = np.where(a==max_slope_index)
					req_slopes = slopes_array[req_slope_indexes]

					print "slopes_array:",slopes_array

					slopes_array = slopes_array.tolist()
					print "slopes_Array_list:",slopes_array
					slopes_array.sort()
					print "slopes_array_sorted:",slopes_array
					print "slopes_array:",slopes_array

					len_slopes_array = len(slopes_array)

					if len_slopes_array%2==0:
						slope_median = (slopes_array[len_slopes_array//2] + slopes_array[len_slopes_array//2 -1])/float(2)
					else:
						slope_median = slopes_array[len_slopes_array//2]
						
					if abs(slope_median-slope)<20:
						self.curr_slope = slope_median
					else:
						self.curr_slope = slope

					print "slope_median-----------",slope_median
					print "curr_slope:----------",self.curr_slope
					


					print slopes
					filtered_lines_slope = self.find_lines_slope(hough_lines, slope)


					merged_lines = self.clustering_hough_bundler.process_lines(filtered_lines_slope)


					cluster_points, cluster_lines, img_count = self.find_cluster(merged_lines, gray_img)

					num_pixels.append(cluster_points)


					sanity = self.check_sanity(gray_img, cluster_points)
					cluster_sanity[i] = sanity

			max_index = None
			max_value = None

			# print "num_pixels:",num_pixels
			cluster_index = []
			print "cluster_sanity:",cluster_sanity
			for num_cluster in cluster_sanity:
				if cluster_sanity[num_cluster] is False: 
					pass
				else:
					cluster_index.append(num_cluster)
					if max_index is None:
						max_index = num_cluster
						# print "The cluster is ---------:",cluster
						# print "the num pixel is:",num_pixels[cluster]
						max_value = num_pixels[num_cluster]
					elif num_pixels[num_cluster] > max_value:
						max_index = num_cluster
						max_value = num_pixels[max_index]

			# cluster_index = max_index

			if len(cluster_index)==0:
				# print "cluster index is None---------------->"
				cluster_index.append(np.argmax(num_pixels))


			print num_pixels
			# print len(num_pixels[0]),len(num_pixels[1]),num_pixels[2],len(num_pixels[3])
			print cluster_index
			# plt.savefig('/home/darth/Desktop/windturbine/blades1/test_low_pressure/' + str(self.curr) + '.png')
			self.curr+=1

			return cluster_index





	def perform_training(self):
		global first_time

		if first_time is True:
			
			if self.bgr_image is not None and self.if_already_training is False:
				
				print "inside i--------------------"
				time1 = time.time()
				self.if_already_training = True
				print "shape:---->:",self.bgr_image.shape
				bgr_img = cv2.resize(self.bgr_image, (self.bgr_image.shape[0] // 2, self.bgr_image.shape[1] // 2))
				rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

				self.pre_process(rgb_img)

				rgb_copy = rgb_img.copy()

				rgb_img = rgb_img.reshape((rgb_img.shape[0] * rgb_img.shape[1], 3))
				training_img = rgb_img

				#KMeans clustering --> clusters data into four different clusters on the basis of rgb colorspace of each pixel.Tweak the parameter n_cluster
				#to 3-5 to see the best resutls.

				clt = KMeans(self.n_clusters)
				clt.fit(training_img)
				time2 = time.time()

				print "time taken to train is:",time2-time1
				# self.clt = clt

				# self.if_already_training = False

				time3 = time.time()
				# f, ax = plt.subplots(1, 5, figsize=(50, 50))

				#In this step ,the rgb image is segmented down into n_cluster images ,each image depicting the pixels belonging to that cluster.

				image_clusters = self.plot_clusters(clt,rgb_copy, self.n_clusters)
				# print len(image_clusters)
				time4 = time.time()
				# print "time taken to predict each pixel is:", time4 - time3

				self.clt = clt
				clt_label = self.detect_blade_cluster(image_clusters)

				if len(clt_label)>0:
					print clt_label
					self.clt_label = clt_label
				else:
					first_time = False
					self.if_already_training = False
				
				# self.clt_label = clt_label

				# if self.clt_label is None or len(self.clt_label)==0:
				# 	pass
				# else:
				# 	first_time = False
				# 	self.if_already_training = False

				# print self.clt,self.clt_label
				# self.if_already_training = False
				# first_time = False


	def handle_training(self,req):
		# print "trainer running"
		# print self.bgr_image ,self.if_already_training

		self.curr_slope = req.slope
		success = self.perform_training()

		if success is False:
			message = "KMeans model failed to train"
			return success,message
		else:
			message = "KMeans model saved successfully.The slope is updated to "," ",str(self.curr_slope)
			return success,message




	def get_pixel_density(self,line1,line2,binary_img,area):
		x_list = [line1[0][0],line1[1][0],line2[0][0],line2[1][0]]
		x_min = min(x_list)
		x_max = max(x_list)

		y_list = [line1[0][1],line1[1][1],line2[0][1],line2[1][1]]
		y_min = min(y_list)
		y_max = max(y_list)

		area = abs((y_min-y_max) * (x_min-x_max))

		num_pixels = np.sum(binary_img[y_min:y_max,x_min:x_max]==255)
		return num_pixels/float(area)

	def length_calculator(self,line):
		x1 = line[0][0]
		x2 = line[1][0]
		y1 = line[0][1]
		y2 = line[1][1]

		return math.sqrt((x2-x1)**2 + (y1-y2)**2)

	def pointDist(self,x,y,line2):

		x_mid = x

		

		y_mid = y
		# line2

		line2_x1 = line2[0][0]
		line2_x2 = line2[1][0]
		line2_y1 = line2[0][1]
		line2_y2 = line2[1][1]

		dy = line2_y2 - line2_y1
		dx = line2_x2 - line2_x1

		if abs(dx)<0.01:
			if dx<0:
				dx = -0.01
			else:
				dx = 0.01

		m = dy/float(dx)
		c = line2_y1 - m*line2_x1

		dist = abs(y_mid -m*x_mid - c)/float(math.sqrt(1+m**2))
		return dist

	def find_req_merged_lines(self,merged_lines,gray_img,slope):

		max_index_length = None
		max_length_value = None
		img_height = gray_img.shape[0]
		img_width = gray_img.shape[1]
		#	Find the largest line that matches the current slope.

		line_info_list = [] #	This dict contains info about line's length and slope.
		line_cost_dict = {}

		slope = slope*180/math.pi
		if slope < 0:
			slope+=180
	
		for i in range(len(merged_lines)):
			line = merged_lines[i]
			x1 = line[0][0]
			y1 = line[0][1]
			x2 = line[1][0]
			y2 = line[1][1]

			length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
			dy = y2 - y1
			dx = x2 - x1
			if abs(dx) < 0.01:
				if dx<0:
					dx = -0.01
				else:
					dx = 0.01
			slope_line = math.atan2(dy,dx)
			
			slope_line *=180/math.pi

			if slope_line<0:
				slope_line+=180

			slope_diff = abs(slope_line - slope)/10.0
			slope_cost = math.exp(-1*slope_diff)

			slope_rad = slope_line*math.pi/180.0
			if 45<slope_line<135:
				length_ratio = abs(length * math.cos(slope_rad))/img_height
				length_cost = length_ratio 
			else:
				length_ratio = abs(length * math.sin(slope_rad))/img_width
				length_cost = length_ratio

			total_cost = length_cost + slope_cost

			dict_ = {}
			dict_['line'] = line
			dict_['cost'] = total_cost
			dict_['length'] = length
			dict_['slope'] = slope_line

			line_info_list.append(dict_)

			line_cost_dict[i] = total_cost

		if len(line_cost_dict) == 0:
			lines = []
			return lines

		max_index_length = max(line_cost_dict.iteritems(), key=operator.itemgetter(1))[0]
		max_length_value = line_info_list[max_index_length]['length']
		max_line_slope = line_info_list[max_index_length]['slope']

		#	Find pixel density of each line

		max_pDist_index = None
		max_pDist_value = None

		max_crieteria_line_index = None
		max_crieteria_line_value = None

		line_cost = None
		line_cost_index = None

		ret, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

		for i in range(len(line_info_list)):
			
			if i != max_index_length:
				index_line = line_info_list[i]['line']
				line_length = line_info_list[i]['length']
				line_slope = line_info_list[i]['slope']

				index_x_mid = (index_line[0][0] + index_line[1][0])/float(2.0)
				index_y_mid = (index_line[0][1] + index_line[1][1])/float(2.0)


				pDist = self.pointDist(index_x_mid,index_y_mid,merged_lines[max_index_length])

				# slope_diff = abs(line_slope - max_line_slope)/20.0

				if 45<max_line_slope<135:
					line_slope = line_slope*math.pi/180.0
					pDist_cost = (pDist * math.sin(max_line_slope))/img_width
				else:
					line_slope = line_slope*math.pi/180.0
					pDist_cost = (pDist * math.cos(max_line_slope))/img_height

				# pDist_cost = (pDist_cost)

				area = pDist * max_length_value
				# lines = []
				# lines.append(index_line)
				# lines.append(merged_lines[max_index_length])

				pixel_density = self.get_pixel_density(index_line,merged_lines[max_index_length],binary_img,area)
				# pixel_density_cost = 
				# density = num_pixels/float(area)
				

				pixel_density = (1+pixel_density)**2

				length_cost = (1+(line_length / float(max_length_value)))**2

				cost = pixel_density + length_cost + pDist_cost

				if line_cost is None:
					line_cost = cost
					line_cost_index = i
				else:
					if cost > line_cost:
						line_cost = cost
						line_cost_index = i

				print "The total cost is:",cost
				

		lines = []

		

		if max_index_length is None:
			return lines
		else:
			lines.append(merged_lines[max_index_length])
			if line_cost is None:
				return lines
			else:
				lines.append(merged_lines[line_cost_index])
				return lines


	def find_approach_slope(self,line,x,y,pDist):

		x1 = line[0][0]
		x2 = line[1][0]
		y1 = line[0][1]
		y2 = line[1][1]

		dy = y2-y1
		dx = x2-x1

		if abs(dx)<0.01:
			if dx<0:
				dx = 0.01
			else:
				dx = -0.01

		slope = math.atan2(dy,dx)
		slope_req1 = slope + math.pi/2

		x_req = x + pDist*math.cos(slope_req1)
		y_req = y + pDist*math.sin(slope_req1)
		dist1 = self.pointDist(x_req,y_req,line)

		slope_req2 = slope - math.pi/2
		x_req = x + pDist*math.cos(slope_req2)
		y_req = y + pDist*math.sin(slope_req2)
		dist2 = self.pointDist(x_req,y_req,line)

		if dist2>dist1:
			return slope_req1
		else:
			return slope_req2




	def classifier(self,event):

		# print "inside classification module"
		# print self.clt
		print self.clt_label

		if self.clt is not None and self.clt_label is not None and self.curr_slope is not None:
			# print "performing classification"
			try:
				# print "I am inside classifier to detect!"
				

				bgr_img = self.bgr_image

				bgr_img = cv2.resize(bgr_img,(bgr_img.shape[0]//2,bgr_img.shape[1]//2))
				color_img = np.zeros((bgr_img.shape[0],bgr_img.shape[1],3),np.uint8)
				rgb_img = cv2.cvtColor(bgr_img,cv2.COLOR_BGR2RGB)
				rgb_img = self.pre_process(rgb_img)
				rgb_img = rgb_img.reshape((rgb_img.shape[0] * rgb_img.shape[1],3))
				clt = self.clt
				clt_label = self.clt_label

				k = clt.predict(rgb_img)

				color_img = np.zeros((bgr_img.shape[0],bgr_img.shape[1]),np.uint8)
				color_img = color_img.ravel()
				for i in clt_label:
					a = np.where(k==i)
					color_img[a] = 255

				color_img = color_img.reshape(bgr_img.shape[0], bgr_img.shape[1])
				color_img = np.tile(color_img[:,:,None],[1,1,3])


				# f, ax = plt.subplots(1,5,figsize=(50,50))
				# ax[0].imshow(bgr_img)
				
				
				
				# for i in range(len(k)):
				# 	row = i//bgr_img.shape[1]
				# 	col = i%bgr_img.shape[1]
					
				# 	if k[i] in clt_label:
				# 		color_img[row,col,:] = (255,255,255)



			#         print "done2"

				# ax[1].imshow(color_img)
					
				gray_img = cv2.cvtColor(color_img,cv2.COLOR_RGB2GRAY)
				ret,gray_img = cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY)
				gray_img = self.pre_process(gray_img)
			#         ax[i][0].imshow(gray_img,cmap='gray')
			#     plt.show()

			#     edge_img = cv2.Canny(gray_img,)
				edges = cv2.Canny(gray_img,10,150,apertureSize = 3)
			#     ax[i][1].imshow(edges,cmap='gray')
			#     plt.imshow(edges,cmap='gray')
			#     plt.show()

				kernel = np.ones((5,5),np.uint8)
				dilated_img = cv2.dilate(edges,kernel,iterations = 2)
				erosion_img = cv2.erode(dilated_img,kernel,iterations = 1)
			#     ax[i][2].imshow(erosion_img,cmap='gray')

			#     print erosion_img.shape
				hough_img,hough_lines = self.detect_hough_lines(erosion_img)
			#     print "done3"
				# ax[2].imshow(hough_img,cmap='gray')

			#     print hough_lines
				slopes= []
				if hough_lines is not None:
			#         print "done4"
					for line in hough_lines:
						for x1,y1,x2,y2 in line:
							dy = y2-y1
							dx = x2-x1

							if abs(dx) < 0.01:
								if dx<0:
									dx = -0.01
								else:
									dx = 0.01

							slope = math.atan2(dy,dx) * 180.0/math.pi
							slopes.append(slope)




					print "slopes:",slopes

					slopes_array = np.asarray(slopes)
					print "Slopes_array:",slopes_array.shape

					interval_bins = [-180,-150,-120,-90,-60,-30,0,30,60,90,120,150,180]
					# N, bins, patches = ax[0].hist(depth_array, edgecolor='white', linewidth=1, bins=interval_bins)

					N, bins, patches = plt.hist(slopes_array, edgecolor='white', linewidth=1,
														  bins=interval_bins)
					max_slope_index = np.argmax(N)

					a = np.digitize(slopes_array,bins)
					a = np.asarray(a)

					req_slope_indexes = np.where(a==max_slope_index)
					req_slopes = slopes_array[req_slope_indexes]

					print "slopes_array:",slopes_array

					slopes_array = slopes_array.tolist()
					print "slopes_Array_list:",slopes_array
					slopes_array.sort()
					print "slopes_array_sorted:",slopes_array
					print "slopes_array:",slopes_array

					len_slopes_array = len(slopes_array)

					if len_slopes_array%2==0:
						slope_median = (slopes_array[len_slopes_array//2] + slopes_array[len_slopes_array//2 -1])/float(2)
					else:
						slope_median = slopes_array[len_slopes_array//2]
						
					# if abs(slope_median-slope)<20:
					# 	self.curr_slope = slope_median
					# else:
					# 	self.curr_slope = slope

					print "slope_median-----------",slope_median
					# print "curr_slope:----------",self.curr_slope
					# slope = self.curr_slope
					self.curr_slope = slope_median
					slope = slope_median


					print slopes
					filtered_lines_slope = self.find_lines_slope(hough_lines,slope)
			#         print "done5"
				#     print selected_lines
					# hough_img_req = np.zeros((hough_img.shape[0],hough_img.shape[1],3),np.uint8)
					# for line in filtered_lines_slope:
					#     for x1,y1,x2,y2 in line:
					#         cv2.line(hough_img_req,(x1,y1),(x2,y2),(0,0,255),2)

					# ax[3].imshow(hough_img_req,cmap='gray')

					# hough_class = HoughBundler()
					merged_lines = self.prediction_hough_bundler.process_lines(filtered_lines_slope)
					# if merged_lines is not None:
					# 	# print "lines are merged-------------------------------------------------------->"
					# 	# clustered_img= np.zeros((hough_img.shape[0],hough_img.shape[1],3),np.uint8)

					# 	for line in merged_lines:
					# 		x1 = line[0][0]
					# 		y1 = line[0][1]
					# 		x2 = line[1][0]
					# 		y2 = line[1][1]
					# 		# cv2.line(merged_img,(x1,y1),(x2,y2),(255,0,0),2)
					# 		cv2.line(bgr_img,(x1,y1),(x2,y2),(0,255,0),5)

					#segregating lines based on the distance
				#     print len(filtered_lines_slope)
					# print "num_merged_lines:",len(merged_lines)
					time1 = time.time()
					# cluster_points, cluster_lines, img_count = self.find_cluster(merged_lines, gray_img)

					req_lines = self.find_req_merged_lines(merged_lines,gray_img,slope)
					time2 = time.time()
					print "time taken to find cluster is:",time2-time1

					if req_lines is not None:
						# print "lines are  clustered-------------------------------------------------------->"
						clustered_img= np.zeros((hough_img.shape[0],hough_img.shape[1],3),np.uint8)

						for line in req_lines:
							x1 = line[0][0]
							y1 = line[0][1]
							x2 = line[1][0]
							y2 = line[1][1]
							# cv2.line(merged_img,(x1,y1),(x2,y2),(255,0,0),2)
							cv2.line(bgr_img,(x1,y1),(x2,y2),(255,0,0),5)
					else:
						print "No lines clustered."

					img_width = bgr_img.shape[1]
					img_height = bgr_img.shape[0]

					if len(req_lines) ==0:
						print "The number of edges detected is zero."
					elif len(req_lines) ==1:
						print "The number of edges detected is one"

						# make a rectangle
						# count the number of white pixels in the rectangle
						# If the ratio of white pixels to total number of pixels is greater than a threshold,consider it lying inside the blade area else outside.

						line = req_lines[0]
						x1 = img_width/2 - 25
						x2 = img_width/2 + 25

						y1 = img_height/2 -25
						y2 = img_height/2 + 25

						ret, gray_threshold = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
						white_pixel_count = np.sum(gray_threshold[y1:y2,x1:x2]==255)
						ratio = white_pixel_count/float(50**2)
						pDist = self.pointDist(img_width/2,img_height/2,line)
						approach_slope = self.find_approach_slope(line,img_width/2,img_height/2,pDist)
						if ratio > 0.5:  
							approach_slope = approach_slope + math.pi

						approach_slope_remainder = approach_slope%(2*math.pi)
						if approach_slope_remainder > 0:
							approach_slope+=approach_slope_remainder
						else:
							approach_slope-=approach_slope_remainder

						if approach_slope>0:
							x_coord = img_width
							y_coord = img_height
						else:
							x_coord = 0
							y_coord = 0

						# Take x coord as 0
						dist_x = abs((x_coord - (img_width/2)) / float(math.cos(approach_slope)))
						# Take x coord as img_width
						dist_y = abs((y_coord - (img_height/2))/float(math.sin(approach_slope)))
						min_dist = min(dist_y,dist_x)

						dist = min(dist_y,dist_x)
						error_x = dist * math.cos(approach_slope)
						error_y = dist * math.sin(approach_slope)

						req_point_y = int(img_height/2 + error_y)
						req_point_x = int(img_width/2 + error_x)


						pose = Float32MultiArray()
						data =[error_x,error_y,self.curr_slope]
						pose.data = data
						self.blade_pose_data.publish(pose)


						cv2.circle(bgr_img,(req_point_x,req_point_y),10,(255,0,0),1)
						cv2.circle(bgr_img, (img_width/2, img_height/2), 10, (0, 255, 0), 1)

					else:
						print "The number of edges detected is two."

						

						img_center = [img_width/2,img_height/2]
						check_center_side = self.check_point_side(req_lines,img_center,gray_img)

						line1 = req_lines[0]
						line2 = req_lines[1]

						dist_btw_line = self.checkpDist(line1,line2)
						pDist1 = self.pointDist(img_width/2,img_height/2,line1)
						pDist2 = self.pointDist(img_width/2,img_height/2,line2)

						if check_center_side ==1:
							# print "Inside--------------------------------->"

							if pDist1 > pDist2:
								approch_slope = self.find_approach_slope(line1,img_width/2,img_height/2,pDist1)
								dist = abs(dist_btw_line/float(2.0) - pDist1)
								error_x = dist*math.cos(approch_slope)
								error_y = dist*math.sin(approch_slope)

								req_point_x = img_width/2 + int(error_x)
								req_point_y = img_height/2 + int(error_y)


								pose = Float32MultiArray()
								data =[error_x,error_y,self.curr_slope]
								pose.data = data
								self.blade_pose_data.publish(pose)

								# return [error_x,error_y,True]

							else:
								approch_slope = self.find_approach_slope(line2,img_width/2,img_height/2,pDist2)
								dist = abs(dist_btw_line/float(2.0) - pDist2)
								error_x = dist*math.cos(approch_slope)
								error_y = dist*math.sin(approch_slope)

								req_point_x = img_width/2 + int(error_x)
								req_point_y = img_height/2 + int(error_y)

								pose = Float32MultiArray()
								data =[error_x,error_y,self.curr_slope]
								pose.data = data
								self.blade_pose_data.publish(pose)


							cv2.circle(bgr_img, (img_width/2, img_height/2), 10, (0, 255, 0), 1)
							cv2.circle(bgr_img,(req_point_x,req_point_y),10,(255,0,0),1)
						else:

							print "I am outside the two lines"
							line1_length = self.length_calculator(line1)
							line2_length = self.length_calculator(line2)

							if pDist1 > pDist2:
								error_dist = pDist2 + dist_btw_line/float(2.0)
							else:
								error_dist = pDist1 + dist_btw_line/float(2.0)

							if line1_length>line2_length:
								approch_slope = self.find_approach_slope(line1,img_width/2,img_height/2,pDist1)
							else:
								approch_slope = self.find_approach_slope(line2,img_width/2,img_height/2,pDist2)

							error_x = error_dist * math.cos(approch_slope)
							error_y = error_dist * math.sin(approch_slope)

							pose = Float32MultiArray()
							data =[error_x,error_y,self.curr_slope]
							pose.data = data
							self.blade_pose_data.publish(pose)

							req_point_x = img_width/2 + int(error_x)
							req_point_y = img_height/2 + int(error_y)

							cv2.circle(bgr_img,(req_point_x,req_point_y),10,(255,0,0),1)
							cv2.circle(bgr_img, (img_width/2, img_height/2), 10, (0, 255, 0), 1)




					self.color_img_topic.publish(self.bridge.cv2_to_imgmsg(bgr_img, "bgr8"))
					# time2 = time.time()
					# print "time taken to classify is:",time2-time1

			except KeyboardInterrupt as e:
				print e


if __name__=="__main__":
	rospy.init_node('Blade_detection_node', anonymous=True)
	print "Started blade detection node"

	bd = blade_detector()
	# bd.perform_training()

	try:
		rospy.spin()
	except KeyboardInterrupt:
		print "shutting down the node"
