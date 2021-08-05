#
import cv2
import numpy as np
import pickle
import time
import glob
from  matplotlib import pyplot as plt


#Author : Sergio Omar Martinez Garcia
#The aim of this project is to find the lane line of a road,
#the imput will be a stream of video

#1 Get the matrix camera and distortion matrix
#2 use the distortion matrix to undistort images
#3 simplyfy the images using gray and s channel from hvs color space
#4 use sovel operator to get the gradient to extract the line lines
#5 mask the region of interest
#6 use warp transformation to make birds-eye view of the road
#7 use sliding window and histogram to follow the path of the lane line
#8 get a second grade function for left and right lane line
#9 find the curvature of the road
#10 unwarp the image
#11 add curvature and the center of the car according to the lane_line

#look for camera_matrix file
def retrieve_camera_values():
    try:
        with open('camera_matrix.pickle','rb') as camera_matrix_file:
            mtx, dist = pickle.load(camera_matrix_file)
    except:
        print('Camera matrix and distortion  matrix do not exist.')
        print('starting camera calibration')
        camera_calibration()
    return mtx,dist

def camera_calibration():
    obj_points = []
    img_points = []
    chess_dir = 'camera_cal/*.jpg'
    list_of_files = []
    chess_images = []
    objp = np.zeros((9*6,3),np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    for address in glob.glob(chess_dir):
        list_of_files.append(address)

    for file in list_of_files:
        img = cv2.imread(file)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray,(9,6),None)
        if (ret == True):
            img_points.append(corners)
            obj_points.append(objp)

    ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(obj_points,img_points,gray.shape[::-1],None,None)
    print(ret)
    print('camera matrix successfull')
    with open('camera_matrix.pickle','wb') as camera_matrix_file:
        pickle.dump((mtx,dist),camera_matrix_file,protocol=pickle.HIGHEST_PROTOCOL)

def undistort(img):
    mtx,dist = retrieve_camera_values()
    undistorted  = cv2.undistort(img,mtx,dist,None,mtx)
    return undistorted

def convert_hvs_space(img):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    s_channel = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)[:,:,2]
    return s_channel

def apply_sobel(gray,min,max):
    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= min) & (scaled_sobel <= max)] = 255
    return sxbinary

def stake_in_binnary_channels(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    s_channel = convert_hvs_space(img)
    gray_binary = np.zeros_like(gray)
    sobel_gray = apply_sobel(gray,20,150)
    #gray_binary[(gray>80)&(gray<180)]=255
    s_channel_binnary = np.zeros_like(s_channel)
    s_channel_binnary[(s_channel>110)&(s_channel<255)] = 255
    #sobel_s_channel = apply_sobel(s_channel_binnary,20,150)
    stk = np.zeros_like(s_channel_binnary)
    stk[(sobel_gray==255)|(s_channel_binnary==255)] = 255
    return stk

def apply_mask(img):
    region_select = np.zeros(img.shape)

    x_size = img.shape[1]
    y_size = img.shape[0]

    left_bottom = [200,720]
    right_bottom = [1150,720]
    left_top =  [610,400]
    right_top = [1280-610,400]

    #let's generate the four lines that will delimit our mask
    fit_top = np.polyfit((left_top[0],right_top[0]),(left_top[1],right_top[1]),1)
    fit_left = np.polyfit((left_bottom[0],left_top[0]),(left_bottom[1],left_top[1]),1)
    fit_right = np.polyfit((right_bottom[0],right_top[0]),(right_bottom[1],right_top[1]),1)
    fit_bottom  = np.polyfit((left_bottom[0],right_bottom[0]),(left_bottom[1],right_bottom[1]),1)

    XX,YY = np.meshgrid(np.arange(0,x_size),np.arange(0,y_size))
    region_threshold = (YY > (XX*fit_left[0]+ fit_left[1])) &\
            (YY > (XX*fit_right[0] + fit_right[1])) & \
            (YY > (XX*fit_top[0] + fit_top[1])) & \
            (YY < (XX*fit_bottom[0] + fit_bottom[1]))

    region_select[region_threshold] = [255]
    masked_stk = np.zeros_like(img)
    masked_stk[(img > 0) & (region_select > 250 )] = 255
    return masked_stk

def warp_image(img):
    x_shape = img.shape[1]
    y_shape = img.shape[0]

    red_p = (192,720)
    blue_p = (592,450)
    yellow_p =(1280 - 160,720)
    green_p = (687,450)

    src = np.float32(
    [[192,720],
    [592,450],
    [(1280-160),720],
    [(687),450]]
    )

    dst = np.float32([
    [192+200,720],#derecha
    [192+200,100],
    [(1280 - 160)/2+100,720],#izquierda abajo
    [(1280 - 160)/2+100,100]] #izquierda arriba
    )
    m = cv2.getPerspectiveTransform(src, dst)
    img_size = (img.shape[1],img.shape[0])
    warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)
    return warped

def sliding_window(warped):
    histogram = np.sum(warped[warped.shape[0]//2:,:],axis=0)
    #plt.plot(histogram)
    #plt.show()
    first_max = np.argmax(histogram)
    left_margin = np.argmax(histogram) - 50
    right_margin = np.argmax(histogram) + 50
    histogram[(left_margin):(right_margin)]=0
    second_max = np.argmax(histogram)
    watch_dog = 0

   # while not((second_max > first_max-270-30) and (second_max < first_max-270+30)
   #         or ((second_max < first_max +270 + 30) and (second_max > first_max+270 -30))):
   #     #plt.plot(histogram)
   #     #plt.show()
   #     histogram[(second_max-50):(second_max+50)]= 0
   #     second_max = np.argmax(histogram)
   #     watch_dog +=1
   #     if watch_dog > 5:
   #         break
    #left =0 , right =1
    second_max = first_max + 270
    second_max_side = True
    if (second_max > first_max-270-30) and (second_max < first_max-270+30):
        second_max_side =  False
    if (second_max < first_max +270 + 30) and (second_max > first_max+270 -30):
        second_max_side = True
        
        lane_center = (first_max + second_max) // 2
        
       #hiper parameters
    n_windows = 8
    margin = 35
    min_pix = 15
    #set the hight of the window
    window_height = warped.shape[0]//n_windows
    non_zero = warped.nonzero()
    non_zero_y = non_zero[0]
    non_zero_x = non_zero[1]
    left_current = first_max
    right_current = second_max
    warped = cv2.circle(warped,(first_max,700),radius=10,color=(255,255,255),thickness=-1)
    warped = cv2.circle(warped,(second_max,700),10,(255,255,255),-1)
    left_lane_inds = []
    right_lane_inds = []
    for window in range(n_windows):
        win_y_low = warped.shape[0]- (1+window) * window_height
        win_y_high = warped.shape[0] - window * window_height
        win_xleft_low = left_current - margin
        win_xleft_high = left_current + margin
        win_xright_low = right_current - margin
        win_xright_high = right_current + margin
        #draw the fucking boxex
        nz_left_index = ((non_zero_y >= win_y_low) & (non_zero_y <=win_y_high)&
                (non_zero_x >= win_xleft_low) & (non_zero_x <= win_xleft_high)).nonzero()[0]
        nz_right_index = ((non_zero_y >= win_y_low) & (non_zero_y <=win_y_high)&
                (non_zero_x >= win_xright_low) & (non_zero_x <= win_xright_high)).nonzero()[0]
        if(len(nz_left_index)>min_pix):
            left_current = int(np.mean(non_zero_x[nz_left_index]))
        
        if(len(nz_right_index)>min_pix):
            right_current = int(np.mean(non_zero_x[nz_right_index]))
        else:
            right_current = left_current + 270
        cv2.rectangle(warped,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(255,255,255), 2)
        cv2.rectangle(warped,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(255,255,0), 2)
        left_lane_inds.append(nz_left_index)
        right_lane_inds.append(nz_right_index)

    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except:
        pass
    #extract left and right pixel position
    leftx = non_zero_x[left_lane_inds]
    lefty = non_zero_y[left_lane_inds]
    rightx = non_zero_x[right_lane_inds]
    righty = non_zero_y[right_lane_inds]
    return leftx,lefty,rightx,righty

def return_polynomial(p_left_fit,p_right_fit,leftx,lefty,rightx,righty,img):
    try:
        left_fit = np.polyfit(lefty,leftx,2)
        right_fit = np.polyfit(righty,rightx,2)
    except:
        left_fit = p_left_fit
        right_fit =p_right_fit
    #generate x and y for ploting
    ploty = np.linspace(0,img.shape[0]-1, img.shape[0])
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2+right_fit[1]*ploty + right_fit[2]
    except:
        print("the function fail to fit a line")
        left_fitx = ploty**2 + ploty + 1
        right_fitx = ploty**2 + ploty + 1
    #img[lefty,leftx] = [255]
    #img[righty,rightx] = [255]
    left_fitx = left_fitx.astype(int)
    right_fitx = right_fitx.astype(int)
    return ploty,left_fit,right_fit

def search_around_poly(img,left_fit,right_fit):
    margin = 70
    #this is for testing
    #out_img = np.zeros_like(img)
    #left_inds_t = (nonzerox>(nonzeroy*left_fit[0]**2+nonzeroy*left_fit[1]+left_fit[2]-margin))&(nonzerox<(nonzeroy*left_fit[0]**2+nonzeroy*left_fit[1]+left_fit[2]+margin))

    nonzero = np.nonzero(img)
    nonzerox = np.array(nonzero[1])
    nonzeroy = np.array(nonzero[0])
    #get the indices filtered near to the polylines
    left_inds =(nonzerox>(nonzeroy*left_fit[0]**2+nonzeroy*left_fit[1]+left_fit[2]-margin))&(nonzerox<(nonzeroy*left_fit[0]**2+nonzeroy*left_fit[1]+left_fit[2]+margin))
    right_inds = (nonzerox>(nonzeroy*right_fit[0]**2+nonzeroy*right_fit[1]+right_fit[2]-margin))&(nonzerox <(nonzeroy*right_fit[0]**2+nonzeroy*right_fit[1]+right_fit[2]+margin))
    leftx = nonzerox[left_inds]
    lefty = nonzeroy[left_inds]
    rightx = nonzerox[right_inds]
    righty = nonzeroy[right_inds]
    return leftx, lefty, rightx, righty

def draw_green_path(img,ploty,left_fit,right_fit):
    out_img = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
    left_points = []
    right_points = []
    lane_points = []
    for y in ploty:
        x_left = left_fit[0]*(y**2) + left_fit[1]*y + left_fit[2]
        x_right = right_fit[0]*(y**2) + right_fit[1]*y + right_fit[2]
        left_points.append([x_left,y])
        right_points.append([x_right,y])
    #pts = np.column_stack((ploty,left_fitx))
    #print(type(pts))
    #print(left_fitx.shape)
    cv2.polylines(out_img,[np.array(left_points,np.int32)],False,(255,0,0),5)
     #cv2.drawContours(img, [np.array(x_right)], 0, (0,0,255), 2)
    lane_points = left_points + list(reversed(right_points))
    cv2.fillPoly(out_img,np.array([lane_points],dtype=np.int32),(0,255,0))
    return out_img

def unwarp(img):
    x_shape = img.shape[1]
    y_shape = img.shape[0]

    red_p = (192,720)                                 
    blue_p = (592,450)                                
    yellow_p =(1280 - 160,720)                        
    green_p = (687,450)     

    src = np.float32(
    [[192,720],
    [592,450],
    [(1280-160),720],
    [(687),450]]
    )

    dst = np.float32([
    [192+200,720],
    [192+200,100],
    [(1280 - 160)/2+100,720],
    [(1280 - 160)/2+100,100]]
    )
    m = cv2.getPerspectiveTransform(dst,src)
    img_size = (img.shape[1],img.shape[0])
    unwarped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)
    return unwarped

def measure_curvature_pixels(ploty, left_fit, right_fit,pre_curvature):
    y_eval = np.max(ploty)
    ym_per_pix = 30/720
    xm_per_pix = 3.7/700
    left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    #left_curved and right_curved  are for only pixels
    #left_curverad =  ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**(1.5))/np.absolute(2*left_fit[0])
    #right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**(1.5))/np.absolute(2*right_fit[0])
    return (left_curverad/1000)

def lpf_curvature(prev_est,cur_meassurement):
    alpha =  0.9
    est = alpha*(float(prev_est)) + (1-alpha)*float(cur_meassurement)
    return est

def add_information(img,curvature,center):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    pos                    = (30,50)
    scale                  = 1
    color                  = (255,255,255)
    line_type               = 2
    str_curvature = "Current curvature: " + "%.2f" % curvature + "km"
    cv2.putText(img,str_curvature,pos,font,scale,color,line_type)
    str_center = "Center of the car: " + str(int(center)) +" cm"
    pos = (30,80)
    cv2.putText(img,str_center,pos,font,scale,color,line_type)
    return img

def center_of_car(img,left_fit,right_fit):
    car_center = (1280/2)
    y = 720
    left_point = (left_fit[0]*(y**2) + y*left_fit[1]+left_fit[2])
    right_point = (right_fit[0]*(y**2) + y*right_fit[1]+right_fit[2])
    #note that the adjust of 118 is because the warping transformation
    center_of_lane = (left_point+right_point)/2 - (car_center-118)
    center_in_cm = center_of_lane * .74 
    #draw the center of the car
    img = cv2.circle(img, (1280//2,700), 5,(255,0,0), thickness=2)
    img = cv2.circle(img,(int(left_point+right_point)//2+118,700),4,(0,0,255),3)
    return img,center_in_cm

def process_images(frame,p_left_fit,p_right_fit,prev_est):
    #1)Camera calibration and apply undistortion
    u_img = undistort(frame)
    #2)color transformation and stacking them
    stk = stake_in_binnary_channels(u_img)
    #3)Masking the area of interest
    masked = apply_mask(stk)
    #this masked is just for testing purposes
    #masked  = cv2.circle(masked,(1280//2,700),8,(255,255,255),10)
    #4) Apply perspective transformation
    warped = warp_image(masked)
    if( len(p_left_fit) == 0):
        #5)Apply sliding window method to find lane lines 
        leftx,lefty,rightx,righty = sliding_window(warped)
    else:
        #6)Use the previous poly function to find the lane lines
        leftx,lefty,rightx,righty = sliding_window(warped)
        #leftx,lefty,rightx,righty = search_around_poly(warped,left_fit,right_fit)
    #7) Get the polynomial function 
    ploty, left_fit, right_fit = return_polynomial(p_left_fit,p_right_fit,leftx,lefty,rightx,righty,warped)
    #8) Get the curvature of the lane line
    cur_meassurement = measure_curvature_pixels(ploty,left_fit,right_fit,prev_est)
    #9) Use a low pass filter to smooth the curvature
    curvature = lpf_curvature(prev_est,cur_meassurement)
    warped = cv2.circle(warped,(1280//2-118,700),8,(255,255,255),10)
    #10)Add green path over the detected lane
    green_img = draw_green_path(warped,ploty,left_fit,right_fit)
    #11) Un-warp the frame bakc to the original
    unwarped = unwarp(green_img)
    warped = cv2.cvtColor(warped,cv2.COLOR_GRAY2RGB)
    unwarped, center = center_of_car(unwarped,left_fit,right_fit)
    final_img = cv2.addWeighted(u_img,1,unwarped,0.3, 0 )
    #12)Add curvature and center of the car with respect to the lane
    add_information(final_img,curvature,center)
    return final_img,left_fit, right_fit,curvature


cap = cv2.VideoCapture('project_video.mp4')
# Check if camera opened successfully
left_fit = []
right_fit = []
curvature  = 0
video_frames = []
if (cap.isOpened()== False): 
  print("Error opening video  file")
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    processed,left_fit,right_fit,curvature = process_images(frame,left_fit,right_fit,curvature)
    #show the video real time
    #cv2.imshow('Frame', processed)
    video_frames.append(processed)
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
  else:
    break
cap.release()

size = (1280,720)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('advance_laneline_detection.mov',fourcc, 15, size,True)

for i in range(len(video_frames)):
    out.write(video_frames[i])
out.release()
#cv2.destroyAllWindows()


