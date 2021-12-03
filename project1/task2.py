###############
##Design the function "calibrate" to  return 
# (1) intrinsic_params: should be a list with four elements: [f_x, f_y, o_x, o_y], where f_x and f_y is focal length, o_x and o_y is offset;
# (2) is_constant: should be bool data type. False if the intrinsic parameters differed from world coordinates. 
#                                            True if the intrinsic parameters are invariable.
#It is ok to add other functions if you need
###############
import numpy as np
from cv2 import imread, cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners

def calibrate(imgname):
    pattern=(9,4)
    image=imread('checkboard.png')
    image_bw = cvtColor(image, COLOR_BGR2GRAY )
    retval, image_cord = findChessboardCorners(image=image_bw, patternSize=pattern)
    image_cord= np.squeeze(image_cord)
    criteria = (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 40, 0.001)
    image_cord_refine = cornerSubPix(image_bw, image_cord, (12,12), (-1,-1), criteria=criteria)
    j=0
    image_new_cord = np.zeros([32,2])
    for i in range(len(image_cord_refine)):
        if((i==4) or (i==13) or (i==22) or (i==31)):

            continue


        image_new_cord[j]=image_cord_refine[i]
        j=j+1

    world_cord = np.zeros([32,3])
    for i in range(0,32):
        z= (int(i/8)+1)*10
        rem=i%8
        if(rem<4):
            x=(4-rem)*10
            world_cord[i]=[x,0,z]
        else:
            y=(4-rem-1)*(-10)
            world_cord[i]=[0,y,z]


    matrix=np.zeros([64,12])
    for i in range(0,len(world_cord)):
        Xw=world_cord[i][0]
        Yw=world_cord[i][1]
        Zw=world_cord[i][2]
        x=image_new_cord[i][0]
        y=image_new_cord[i][1]
        j=2*i
        matrix[j]=[Xw,Yw,Zw,1,0,0,0,0,-x*Xw,-x*Yw,-x*Zw,-x]
        matrix[j+1]=[0,0,0,0,Xw,Yw,Zw,1,-y*Xw,-y*Yw,-y*Zw,-y]
    U, D, V = np.linalg.svd(matrix)
    
    matrix_m=V[11]
    m1=matrix_m[0:3]
    m2=matrix_m[4:7]
    m3=matrix_m[8:11]
    r=[matrix_m[8:11]]
    value=np.linalg.norm(r)
    value2=1/value

    m1=m1*value2
    m2=m2*value2
    m3=m3*value2
    ox=np.dot(m1.transpose(),m3)
    fx=(np.dot(m1.transpose(),m1)-ox*ox)**0.5
    oy=np.dot(m2,m3)
    fy=(np.dot(m2,m2)-oy*oy)**0.5
    return [fx,fy,ox,oy],True
    #......
    

if __name__ == "__main__":
    intrinsic_params, is_constant = calibrate('checkboard.png')
    print(intrinsic_params)
    print(is_constant)