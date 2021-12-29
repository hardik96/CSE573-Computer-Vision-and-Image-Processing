# -*- coding: utf-8 -*-
"""
 Grayscale Image Processing
(Due date: Nov. 26, 11:59 P.M., 2021)

The goal of this task is to experiment with two commonly used 
image processing techniques: image denoising and edge detection. 
Specifically, you are given a grayscale image with salt-and-pepper noise, 
which is named 'task2.png' for your code testing. 
Note that different image might be used when grading your code. 

You are required to write programs to: 
(i) denoise the image using 3x3 median filter;
(ii) detect edges in the denoised image along both x and y directions using Sobel operators (provided in line 30-32).
(iii) design two 3x3 kernels and detect edges in the denoised image along both 45° and 135° diagonal directions.
Hint: 
• Zero-padding is needed before filtering or convolution. 
• Normalization is needed before saving edge images. You can normalize image using the following equation:
    normalized_img = 255 * frac{img - min(img)}{max(img) - min(img)}

Do NOT modify the code provided to you.
You are NOT allowed to use OpenCV library except the functions we already been imported from cv2. 
You are allowed to use Numpy for basic matrix calculations EXCEPT any function/operation related to convolution or correlation. 
You should NOT use any other libraries, which provide APIs for convolution/correlation ormedian filtering. 
Please write the convolution code ON YOUR OWN. 
"""

from cv2 import imread, imwrite, imshow, IMREAD_GRAYSCALE, namedWindow, waitKey, destroyAllWindows
import numpy as np

# Sobel operators are given here, do NOT modify them.
sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).astype(int)
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(int)


def filter(img):
    """
    :param img: numpy.ndarray(int), image
    :return denoise_img: numpy.ndarray(int), image, same size as the input image

    Apply 3x3 Median Filter and reduce salt-and-pepper noises in the input noise image
    """

    # TO DO: implement your solution here
    pad_row =  [0]*img.shape[1]

    pad_column = [0] * (img.shape[0]+2)
    img_pad = np.vstack([pad_row,img,pad_row])
    img_pad = np.column_stack((pad_column,img_pad,pad_column))

    median_array=[]
    for i in range(1,img_pad.shape[0]-1):
        for j in range(1,img_pad.shape[1]-1):
            temp_array = img_pad[i-1:i+2,j-1:j+2]
            temp_array_flat = temp_array.flatten()
            temp_array_flat.sort()
            median_array.append(median(temp_array_flat))
    
    denoise_img = np.array(median_array).reshape(img.shape)
    
    return denoise_img


def convolve2d(img, kernel):
    """
    :param img: numpy.ndarray, image
    :param kernel: numpy.ndarray, kernel
    :return conv_img: numpy.ndarray, image, same size as the input image

    Convolves a given image (or matrix) and a given kernel.
    """

    # TO DO: implement your solution here
    padded = padding(img)
    conv_img=[]
    for i in range(1,padded.shape[0]-1):
        for j in range(1,padded.shape[1]-1):
            temp_array = padded[i-1:i+2,j-1:j+2]
            temp_array.astype(int)
            conv_img.append(sum((temp_array*kernel).flatten()))
    conv_img = np.array(conv_img).reshape(img.shape)
    
    return conv_img


def edge_detect(img):
    """
    :param img: numpy.ndarray(int), image
    :return edge_x: numpy.ndarray(int), image, same size as the input image, edges along x direction
    :return edge_y: numpy.ndarray(int), image, same size as the input image, edges along y direction
    :return edge_mag: numpy.ndarray(int), image, same size as the input image, 
                      magnitude of edges by combining edges along two orthogonal directions.

    Detect edges using Sobel kernel along x and y directions.
    Please use the Sobel operators provided in line 30-32.
    Calculate magnitude of edges by combining edges along two orthogonal directions.
    All returned images should be normalized to [0, 255].
    """

    # TO DO: implement your solution here
    
    flipped_x = np.flip(sobel_x)
    flipped_y = np.flip(sobel_y)
    edge_x = convolve2d(img,flipped_x)
    edge_y = convolve2d(img,flipped_y)
    edge_mag = np.sqrt(edge_x*edge_x+edge_y*edge_y)
    return normalize(edge_x), normalize(edge_y), normalize(edge_mag)


def edge_diag(img):
    """
    :param img: numpy.ndarray(int), image
    :return edge_45: numpy.ndarray(int), image, same size as the input image, edges along x direction
    :return edge_135: numpy.ndarray(int), image, same size as the input image, edges along y direction

    Design two 3x3 kernels to detect the diagonal edges of input image. Please print out the kernels you designed.
    Detect diagonal edges along 45° and 135° diagonal directions using the kernels you designed.
    All returned images should be normalized to [0, 255].
    """

    # TO DO: implement your solution here
    
    sobel_45 = np.array([[0,1,2] , [-1,  0,  1] , [-2 ,  -1 , 0]]).astype(int)
    sobel_135 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]]).astype(int)
    print("sobel_45 is:", sobel_45) # print the two kernels you designed here
    print("sobel_135 is:" , sobel_135)
    edge_45 = convolve2d(img,sobel_45)
    edge_135 = convolve2d(img,sobel_135)
    return normalize(edge_45), normalize(edge_135)

def median(array):
    if(len(array)%2!=0):
        return array[int(len(array)/2)]
    else:
        return (array[len(array)/2-1]+array[len(array)/2])/2

def padding(img):
    pad_row =  [0]*img.shape[1]

    pad_column = [0] * (img.shape[0]+2)
    img_pad = np.vstack([pad_row,img,pad_row])
    img_pad = np.column_stack((pad_column,img_pad,pad_column))
    return img_pad

def normalize(img):
    val_min = img.min()
    val_max = img.max()
    return (255 * ((img-val_min)/(val_max-val_min))) 
    

if __name__ == "__main__":
    noise_img = imread('task2.png', IMREAD_GRAYSCALE)
    denoise_img = filter(noise_img)
    imwrite('results/task2_denoise.jpg', denoise_img)
    edge_x_img, edge_y_img, edge_mag_img = edge_detect(denoise_img)
    imwrite('results/task2_edge_x.jpg', edge_x_img)
    imwrite('results/task2_edge_y.jpg', edge_y_img)
    imwrite('results/task2_edge_mag.jpg', edge_mag_img)
    edge_45_img, edge_135_img = edge_diag(denoise_img)
    imwrite('results/task2_edge_diag1.jpg', edge_45_img)
    imwrite('results/task2_edge_diag2.jpg', edge_135_img)