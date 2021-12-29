# -*- coding: utf-8 -*-
"""
Image Stitching Problem
(Due date: Nov. 26, 11:59 P.M., 2021)

The goal of this task is to stitch two images of overlap into one image.
You are given 'left.jpg' and 'right.jpg' for your image stitching code testing. 
Note that different left/right images might be used when grading your code. 

To this end, you need to find keypoints (points of interest) in the given left and right images.
Then, use proper feature descriptors to extract features for these keypoints. 
Next, you should match the keypoints in both images using the feature distance via KNN (k=2); 
cross-checking and ratio test might be helpful for feature matching. 
After this, you need to implement RANSAC algorithm to estimate homography matrix. 
(If you want to make your result reproducible, you can try and set fixed random seed)
At last, you can make a panorama, warp one image and stitch it to another one using the homography transform.
Note that your final panorama image should NOT be cropped or missing any region of left/right image. 

Do NOT modify the code provided to you.
You are allowed use APIs provided by numpy and opencv, except “cv2.findHomography()” and
APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “cv2.BFMatcher()” and
“cv2.Stitcher.create()”.
If you intend to use SIFT feature, make sure your OpenCV version is 3.4.2.17, see project2.pdf for details.
"""

import cv2
import numpy as np
# np.random.seed(<int>) # you can use this line to set the fixed random seed if you are using np.random
import random
# random.seed(<int>) # you can use this line to set the fixed random seed if you are using random


def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result panorama image which is stitched by left_img and right_img
    """
    sift = cv2.xfeatures2d.SIFT_create() 
    kp_left, des_left = sift.detectAndCompute(left_img, None)
    kp_right, des_right = sift.detectAndCompute(right_img, None)
    all_points = [0]*len(kp_left)*2
    dist=[0]*len(kp_left)*2
    
    #Calculate distance KNN=2
    for i in range(len(kp_left)):
        distance_array = distance_norm(kp_left[i],kp_right,des_left[i],des_right)
        arg = np.argpartition(distance_array,2)
        all_points[2*i]=arg[0]
        all_points[2*i+1] = arg[1]
        dist[2*i] = distance_array[all_points[2*i]]
        dist[2*i+1] = distance_array[all_points[2*i+1]]

    
    # Ratio Testing
    i=0
    thresh = 0.75
    new_points=[]
    while(i<len(all_points)-1):
        if(dist[i]/dist[i+1]<thresh):
            new_points.append((int(i/2),all_points[i]))
        i=i+2
    
    inlier_list ,inlier,final_h = Ransac(new_points,kp_left,kp_right)
    h = homography_estimate(inlier_list,kp_left,kp_right)
    h=h/h[8]
    final_h=h.reshape(3,3)
    result_img = offset_warp_stitch(right_img,left_img,final_h)
    
    return result_img


def distance_norm(left,right,x,y):
    return np.sqrt(np.sum(np.square(x - y),axis=1))


def homography_estimate(points,kp_left,kp_right):
    matrix=np.zeros([len(points)*2,9])
    for i in range(len(points)):
        x_left,y_left = kp_left[points[i][0]].pt
        x_right,y_right = kp_right[points[i][1]].pt
        
        j=2*i
        matrix[j]=[x_right,y_right,1,0,0,0,-x_left*x_right,-x_left*y_right,-x_left]
        matrix[j+1]=[0,0,0,x_right,y_right,1,-y_left*x_right,-y_right*y_left,-y_left]
    #print(matrix)
    U, D, V = np.linalg.svd(matrix)
    return V[8]


def Ransac(points,kp_left,kp_right):
    inlier_max = 0
    final_h = []
    check=[]
    in_l = []
    for i in range(1000):
        sample_point = random.sample(points, 4)
        h_matrix = homography_estimate(sample_point,kp_left,kp_right)
        h_matrix = (1/h_matrix[8]) * h_matrix
        h_matrix = h_matrix.reshape(3,3)
        temp_inlier = []
        inlier = 0
        for j in range(len(points)):
            
            
            x_right,y_right = kp_right[points[j][1]].pt
            x_left,y_left = kp_left[points[j][0]].pt
            #arr = np.array([x_right,y_right,1])
            
            #a = np.dot(h_matrix,arr)
            right = np.transpose(np.matrix([x_right, y_right, 1]))
            left_cal = np.dot(h_matrix, right)
            left_cal = (1/left_cal.item(2))*left_cal
            #x_left_cal = a[0]/a[2]
            #y_left_cal = a[1]/a[2]
            left = np.transpose(np.matrix([x_left, y_left, 1]))
            error = left - left_cal
            error = np.linalg.norm(error)
            
            
            #if(np.sqrt((np.square(x_left-x_left_cal)+np.square(y_left-y_left_cal))) < 5 ):
            if(error<1):   
                temp_inlier.append(points[j])
                inlier = inlier+1
               
        if(inlier > inlier_max):
            #inlier_max = temp_inlier
            #check = sample_point
            inlier_max = inlier
            final_h = h_matrix
            in_l = temp_inlier
    
    return in_l,inlier_max,final_h




def offset_warp_stitch(right,left,h):
    (right_width , right_height) = right.shape[0:2]
    (left_width, left_height) = left.shape[0:2]
    
    right_boundary = np.array([[0,0],[0,right_width], [right_height,right_width], [right_height,0]])
    boundary_transform = cv2.perspectiveTransform(np.float32(right_boundary.reshape(-1,1,2)), h)
    left_boundary = np.array([[0,0],[0,left_width], [left_height,left_width], [left_height,0]]) 
    boundary_new = boundary_transform.reshape(4,2)
    all_boundary_x = np.concatenate((left_boundary[:,0] , boundary_new[:,0]))
    x_min = int(min(all_boundary_x))
    x_max = int(max(all_boundary_x))
    all_boundary_y = np.concatenate((left_boundary[:,1] , boundary_new[:,1]))
    y_min = int(min(all_boundary_y))
    y_max = int(max(all_boundary_y))
    translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]]) 
    print(translation_matrix.dot(h))
    print((x_max-x_min,y_max-y_min))
    result_img = cv2.warpPerspective(right, np.matmul(translation_matrix,h), (x_max-x_min,y_max-y_min))
    result_img[-y_min:right_width - y_min, -x_min:right_height - x_min] = left_img
    
    return result_img
if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_img = solution(left_img, right_img)
    cv2.imwrite('results/task1_result.jpg', result_img)