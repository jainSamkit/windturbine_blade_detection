import rospy
import cv2
import numpy as np
import sys
import math
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
import math
# import keyboard
import matplotlib.pyplot as plt
import os
import random
from std_msgs.msg import Float32MultiArray

# global raw_color_image
# raw_color_image = None
# image = None
# image_counter = 0
# global bridge
# bridge = CvBridge()
#
# global count
# count = 1
#
# global horizontal_angle
# global vertical_angle
# global depth_image
# depth_image = None
# depth_data = None
#
# horizontal_angle = 86 * math.pi / 180.0
# vertical_angle = 54 * math.pi / 180.0
# global color_palette
# # color_palette = ['c','m','g','r','b','y','o','0.75']
#
# # color_palette = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
# # color_palette = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
# color_palette = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
#
# global color_img_topic, color_img_raw_topic
# global threshold_bin_size
# threshold_bin_size = 8000



folder_name = "/home/darth/Desktop/windturbine/wt_zed_topic_data_analysis/hist_and_depth/t4b1/"
curr = 0.0

class detect_depth():
    def __init__(self):
    	print "inside init"
        self.raw_color_image = None
        self.bridge = CvBridge()
        self.count = 1
        self.horizontal_angle = 86 * math.pi / 180.0
        self.vertical_angle = 54 * math.pi / 180.0
        self.color_palette = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
        self.color_img_topic = None
        self.color_img_raw_topic = None
        self.depth_image = None
        self.depth_data = None
        self.threshold_bin_size = 5000
        self.bin_scaling = 2
        self.im_topic = None
        self.depth_topic = None
        self.histogram_bins = np.array([2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10])

        self.initialise_pub_and_sub()
        rospy.Timer(rospy.Duration(1.5), self.depth_to_hist)



    def save_images(self,image_name,color_img,raw_color_img,color_img_raw):
    	cv2.imwrite(image_name + '_depth.jpg', color_img)
        cv2.imwrite(image_name + '_color.jpg', raw_color_img)
        cv2.imwrite(image_name + '_raw_depth.jpg', color_img_raw)

        # cv2.imwrite('./t3_h/' + str(image_counter) + '.jpeg', image)
        # image_counter += 1


    def put_color(self,color_img, depth_list, row_list, col_list, a, plot_y, plot_x, index_data):

        print "reaching here too"

        for x in range(len(depth_list)):

            index = a[x] - 1

            color = self.color_palette[index % 8]
            row = row_list[x]
            col = col_list[x]

            # if index==max_index:
            # 	# print index,max_index
            # 	plot_x.append(col)
            # 	plot_y.append(row)

            if index in index_data:
                plot_x.append(col)
                plot_y.append(row)

            if color == 'red':
                color_img[row][col][0] = 0
                color_img[row][col][1] = 0
                color_img[row][col][2] = 255
            elif color == 'blue':
                color_img[row][col][0] = 255
                color_img[row][col][1] = 0
                color_img[row][col][2] = 0
            elif color == 'green':
                color_img[row][col][0] = 0
                color_img[row][col][1] = 255
                color_img[row][col][2] = 0
            elif color == 'orange':
                color_img[row][col][0] = 0
                color_img[row][col][1] = 165
                color_img[row][col][2] = 255
            elif color == 'brown':
                color_img[row][col][0] = 42
                color_img[row][col][1] = 42
                color_img[row][col][2] = 165
            elif color == 'purple':
                color_img[row][col][0] = 128
                color_img[row][col][1] = 0
                color_img[row][col][2] = 128
            elif color == 'pink':
                color_img[row][col][0] = 180
                color_img[row][col][1] = 105
                color_img[row][col][2] = 255
            elif color == 'gray':
                color_img[row][col][0] = 128
                color_img[row][col][1] = 128
                color_img[row][col][2] = 128

        return color_img, plot_x, plot_y




    def blade_center_error(self,color_img,img_center,p_distance,z):

    	blade_slope = math.atan(z[0])
    	print "pdist:",p_distance

    	slope1 = blade_slope + math.pi/2
    	slope2 = blade_slope - math.pi/2

    	img_center_x = img_center[1]
    	img_center_y = img_center[0]

    	x1 = img_center_x + math.sin(slope1) * p_distance
    	y1 = img_center_y + math.cos(slope1) * p_distance

    	x2 = img_center_x + math.sin(slope2) * p_distance
    	y2 = img_center_y + math.cos(slope2) * p_distance

    	pDist1 = abs(x1 - z[0]*y1 - z[1])/math.sqrt(1 + z[0]**2)
    	pDist2 = abs(x2 - z[0]*y2 - z[1])/math.sqrt(1 + z[0]**2)

    	if pDist1 < pDist2:
    		error_x = p_distance * math.sin(slope1)
    		error_y = p_distance * math.cos(slope1)

    		cv2.circle(color_img, (int(x1), int(y1)), 20, (0, 0, 0), -1)
    		cv2.circle(color_img,(int(img_center_x),int(img_center_y)),20,(0,0,255),-1)
    	else:
    		error_x = p_distance * math.sin(slope2)
    		error_y = p_distance * math.cos(slope2)

    		cv2.circle(color_img, (int(x2),int(y2)), 20, (0, 0, 0), -1)
    		cv2.circle(color_img,(int(img_center_x),int(img_center_y)),20,(0,0,255),-1)

    	return color_img,error_x,error_y




    def fit_line(self,color_img, plot_x, plot_y):
        plot_x = np.asarray(plot_x)
        plot_y = np.asarray(plot_y)

        z = np.polyfit(plot_y, plot_x, 1)

        a = z[0]
        b = z[1]

        # line = []
        points_y = []
        
        min_y =None
        max_y =None
        point1 = []
        point2 = []

        print color_img.shape

        for i in range(color_img.shape[1]):
            point_y = int(a * i + b)

            if min_y is None and 0<=point_y<color_img.shape[0]:
            	# print "a"
            	min_y = point_y
            	max_y = point_y
            	point1 = [i,point_y]
            	point2 = [i,point_y]

            elif 0<=point_y<color_img.shape[0]:
            	# print "b"
            	if point_y<min_y:
            		min_y = point_y
            		point1 = [i,point_y]
            	elif point_y>max_y:
            		max_y = point_y
            		point2 = [i,point_y]
            # cv2.circle(color_img, (point_y, i), 2, (0, 0, 0), -1)
        
        

        # print "min max:",min(points_y),max(points_y)
        # print "length of line:",len(line)
        print "point:",point1,point2

        if len(point1) >0 and len(point2)>0:
        	print "inside here"
        	# print 
        	cv2.line(color_img , (point1[1],point1[0]),(point2[1],point2[0]),(0,255,0),5)

        img_center = [color_img.shape[0]/2,color_img.shape[1]/2]
        
        pDist = abs(img_center[1] - a*img_center[0] - b)/math.sqrt(1 + a**2)



        color_img , error_x,error_y = self.blade_center_error(color_img,img_center,pDist,z)

        return color_img,error_x,error_y,a




    def find_indexes_depth(self,N, index, max_index):

        # This API finds the two closest indexes to max_index in order of their heights.

        depth_index = []
        first_index = None
        second_index = None

        first_index_value = None
        second_index_value = None

        for i in range(len(N)):
            if i != max_index:
                if first_index is None:
                    first_index = i
                    second_index = i
                    first_index_value = N[i]
                    second_index_value = N[i]
                else:
                    value = N[i]
                    if value > second_index_value:
                        first_index = second_index
                        first_index_value = second_index_value
                        second_index = i
                        second_index_value = N[i]

                    if first_index_value < value < second_index_value:
                        first_index = i
                        first_index_value = N[i]
        depth_index.append(max_index)
        if first_index is not None:
            depth_index.append(first_index)

        if second_index is not None and second_index != first_index:
            depth_index.append(second_index)

        return depth_index

    def get_indexes(self,N):
        # global threshold_bin_size
        index = []
        max_index = np.argmax(N)
        max_value = N[max_index]

        if_enough_data = False  # This is important to record to say if there are enough data points in the highest bin.

        if max_value > self.threshold_bin_size:
            if_enough_data = True

        index.append(max_index)
        for i in range(len(N)):
            value = N[i]
            if i not in index and value * self.bin_scaling >= max_value:
                index.append(i)

        index_slope = index # This captures the bins with all the bins with heights atleast half the size of the highest bin.
        index_depth = self.find_indexes_depth(N, index, max_index)  # This captures the bins with maximum height and next two bins with smaller heighs.

        return index_slope, index_depth, if_enough_data

    def calculate_depth(self,N, index, bins):
        mean_depth = []
        weights = []
        depth = 0
        if len(index) > 0:
            for i in index:
                depth_acc = np.mean(bins[i])
                mean_depth.append(depth_acc)
                weights.append(N[i])

            for i in range(len(mean_depth)):
                weight = weights[i] / np.sum(weights)
                depth += weight * mean_depth[i]

        return depth


    def handle_depth_calc(self,req):

        if self.depth_data is not None and self.depth_image is not None:
            try:
                x1 = req.x1
                y1 = req.y1
                x2 = req.x2
                y2 = req.y2

                depth = np.mean(depth_data[x1:x2,y1:y2])
                return True,depth
            except Exception as e:
                rospy.loginfo(e)
                return False,-1
        else:
            return False,-1

    def depth_to_hist(self, event):
    	print "inside depth_to_hist"

        global folder_name,curr

        image_name = folder_name + str(curr) + ".png"

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 500)
        fontScale = 1
        fontColor = (0, 0, 0)
        lineType = 2

        # global depth_data, depth_image, count, color_palette, raw_color_image, color_img_topic, bridge, color_img_raw_topic, threshold_bin_size

        raw_color_img = self.raw_color_image

        depth_list = []
        depth_list_raw = []
        col_list = []
        row_list = []

        gray_img = self.depth_image

        time1 = time.time()

        if self.depth_data is not None and self.depth_image is not None:

            for j in range(self.depth_data.height):
                for i in range(self.depth_data.width):
                    k = gray_img[j, i]
                    # print i,j
                    if not math.isnan(k) and not math.isinf(k):
                        # print ("inside if statement")
                        # h_angle = (self.horizontal_angle) * abs((i - self.depth_data.width / 2)) / (
                        # self.depth_data.width)
                        # v_angle = (self.vertical_angle) * abs((j - self.depth_data.height / 2)) / (
                        # self.depth_data.height)
                        # dist = k * math.cos(h_angle) * math.cos(v_angle)
                        # # print dist
                        # depth_list.append(dist)
                        depth_list_raw.append(k)
                        # depth_list2.append(k)
                        col_list.append(i)
                        row_list.append(j)

            if len(depth_list_raw) > 0:
                # print depth_list
                # depth_array = np.asarray(depth_list)
                depth_array_raw = np.asarray(depth_list_raw)

                fig, ax = plt.subplots(2, 1, sharex='col', sharey='row')

                interval_bins = self.histogram_bins
                # N, bins, patches = ax[0].hist(depth_array, edgecolor='white', linewidth=1, bins=interval_bins)

                N_raw, bins_raw, patches_raw = ax[1].hist(depth_array_raw, edgecolor='white', linewidth=1,
                                                          bins=interval_bins)

                # bin_heights = sorted(N.tolist(), reverse=True)

                for i in range(len(patches_raw)):
                    color = self.color_palette[i % 8]
                    # patches[i].set_facecolor(color)
                    patches_raw[i].set_facecolor(color)

                # max_index = np.argmax(N)
                max_index_raw = np.argmax(N_raw)

                # index_slope, index_depth, if_enough_data = self.get_indexes(N)
                index_raw_slope, index_raw_depth, if_enough_data = self.get_indexes(N_raw)

                if if_enough_data is True:
                	confidence = 1
                else:
                	confidence = -1

                # print max_index, max_index_raw
                print max_index_raw

                # ax[0].set_ylabel('Number of data points')
                # ax[0].set_xlabel('Distance')

                # ax[1].set_ylabel('Number of data points')
                # ax[1].set_xlabel('Distance')

                # ax[0].set_title('Normed Histogram')
                # ax[1].set_title('Raw Histogram')

                # depth = self.calculate_depth(N, index_depth, bins)
                depth_raw = self.calculate_depth(N_raw, index_raw_depth, bins_raw)

                # print depth, depth_raw
                print depth_raw

                # if curr % 4 == 0:
                #     plt.savefig(image_name)
                # plt.close()

                # plotting colors on depth image
                # a = np.digitize(depth_array, bins)
                a_raw = np.digitize(depth_array_raw, bins_raw)
                # print max(a)
                # print min(a)
                # color_img = cv2.cvtColor(gray_img,cv2.COLOR_GRAY2RGB)

                # plot_x = []
                # plot_y = []

                plot_x_raw = []
                plot_y_raw = []

                # color_img = np.ones([self.depth_image.shape[0], self.depth_image.shape[1], 3], dtype=np.uint8) * 255
                color_img_raw = np.ones([self.depth_image.shape[0], self.depth_image.shape[1], 3], dtype=np.uint8) * 255

                # color_img, plot_x, plot_y = self.put_color(color_img, depth_list, row_list, col_list, a, plot_y, plot_x,
                                                           # index_slope)
                color_img_raw, plot_x_raw, plot_y_raw = self.put_color(color_img_raw, depth_list_raw, row_list,
                                                                       col_list, a_raw,
                                                                       plot_y_raw, plot_x_raw, index_raw_slope)

                

                # if len(plot_x) > 0:
                #     color_img,error_x,error_y,slope = self.fit_line(color_img, plot_x, plot_y)

                print "reaching here"
                if len(plot_x_raw) > 0:
                    print "okay okay"
                    color_img_raw,error_x_raw,error_y_raw,slope_raw = self.fit_line(color_img_raw, plot_x_raw, plot_y_raw)


                pose = Float32MultiArray()
                # pose.data = [depth,slope,error_x,error_y,confidence]
                pose.data = [depth_raw,slope_raw,error_x_raw,error_y_raw,confidence]
                self.blade_pose_publisher.publish(pose)



                # cv2.putText(color_img, 'Depth:' + str(depth) + '\n'+ 'error_x:'+str(error_x) + '\n' + 'error_y:'+str(error_y), bottomLeftCornerOfText, font, fontScale, fontColor,
                #             lineType)
                cv2.putText(color_img_raw, 'Depth:' + str(depth_raw) + '\n'+ 'error_x:'+str(error_x_raw) + '\n' + 'error_y:'+str(error_y_raw), bottomLeftCornerOfText, font, fontScale,
                            fontColor,
                            lineType)

                # self.color_img_topic.publish(self.bridge.cv2_to_imgmsg(color_img, "bgr8"))
                self.color_img_raw_topic.publish(self.bridge.cv2_to_imgmsg(color_img_raw, "bgr8"))

                # image_name.replace('.png','')

                # if curr % 4 == 0:
                # 	self.save_images(image_name,color_img,raw_color_img,color_img_raw)
                #     # cv2.imwrite(image_name + '_depth.jpg', color_img)
                #     # cv2.imwrite(image_name + '_color.jpg', raw_color_img)
                #     # cv2.imwrite(image_name + '_raw_depth.jpg', color_img_raw)
                # curr+=2

        time2 = time.time()
        print "time taken:",time2-time1

        # b = Button(master, text="OK", command=depth_to_hist)
        # b.pack()

    def im_callback(self,data):

        self.raw_color_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')

    def depth_callback(self,data):
        # global bridge, horizontal_angle, vertical_angle, depth_image, depth_data
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
            self.depth_data = data
        except CvBridgeError, e:
            print e

        # print depth_data.width/2
        # depth_array = np.array(depth_image, dtype=np.float32)

        # depth_array = depth_array.flatten().tolist()
        # # print ("Depth List:",len(depth_array))
        # for i in depth_array:
        # 	if ~np.isnan(i):
        # 		print i

        # print ('Average distance:'," ",avg_dist)

        # print('Image size: {width}x{height}'.format(width=depth_data.width,height=depth_data.height))

        # u = depth_data.width/2
        # v = depth_data.height/2

        # print('Center depth: {dist} m'.format(dist=depth_array[u,v]))

    def initialise_pub_and_sub(self):
    	print "inside subs and pubs"
        self.im_topic = rospy.Subscriber('flytos/zed/left/image_raw_color', Image, self.im_callback)
        self.depth_topic = rospy.Subscriber('/flytos/zed/depth/depth_registered', Image, self.depth_callback)
        self.color_img_topic = rospy.Publisher('/flytos/zed/depth/color_hist', Image, queue_size=10)
        self.color_img_raw_topic = rospy.Publisher('/flytos/zed/depth/color_hist_raw', Image, queue_size=10)
        self.blade_pose_publisher = rospy.Publisher('/flytos/blade_detection/pose_estimation',Float32MultiArray,queue_size = 10)

    def init_services(self):
        self.calc_depth_handle = rospy.Service('/flytos/wtinspect/zed_stereo_pix',CAPTURE_DEPTH,self.handle_depth_calc)



if __name__ == '__main__':
    
    rospy.init_node("Depth_detector_ZED",anonymous = True)
    print "Started depth detector node."
    depth_detector = detect_depth()

    try:
        rospy.spin()
    except KeyboardInterrupt():
        print "Shutting down the node."


    # while not rospy.is_shutdown():
    #     print("recorded plot and disparity image")
    #
    #     curr = int(curr)
    #     depth_to_hist(folder_name + str(curr) + ".png", curr)
    #     # save_depth_image(folder_name + str(curr) + '_depth.jpeg')
    #     curr += 1.0
    #     rospy.sleep(0.5)
    #
    # rospy.spin()
