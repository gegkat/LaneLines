import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
#plt.switch_backend('Qt4Agg')  

def read_img(filename):
  img = cv2.imread(filename)
  return img

def img2gray(img):
  return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def img2RGB(img):
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def get_calibration(images, nx, ny):

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = read_img(fname)
        gray = img2gray(img)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            write_name = './cal_check/corners_found'+str(idx)+'.jpg'
            cv2.imwrite(write_name, img)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    cv2.destroyAllWindows()
    return mtx, dist

def run_calibration(directory, nx, ny):

  # Make a list of calibration images
  images = glob.glob(directory)
  mtx, dist = get_calibration(images, nx, ny)

  dist_pickle = {}
  dist_pickle["mtx"] = mtx
  dist_pickle["dist"] = dist
  pickle.dump( dist_pickle, open( "camera_cal/cal.p", "wb" ) )

def check_calibration(filename, nx, ny, mtx, dist):
  undist, warped = corners_unwarp(filename, nx, ny, mtx, dist)

  height = int(img.shape[0]/2)
  width = int(img.shape[1]/2)
  cv2.imshow('orig', cv2.resize(img, (width, height)))
  cv2.imshow('undist', cv2.resize(undist, (width, height)))
  try:
    cv2.imshow('warped', cv2.resize(warped, (width, height)))
  except:
    pass

  cv2.waitKey(100000)
  cv2.destroyAllWindows()

def load_calibration():
  dist_pickle = pickle.load( open( "camera_cal/cal.p", "rb" ) )
  mtx = dist_pickle["mtx"]
  dist = dist_pickle["dist"]
  return mtx, dist

# Define a function that takes an image, number of x and y points, 
# camera matrix and distortion coefficients
def corners_unwarp(filename, nx, ny, mtx, dist):
    img, gray = undistort_img(filename, mtx, dist)
    # Search for corners in the grayscaled image
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    warped = []
    M = []

    if ret == True:
        # If we found corners, draw them! (just for fun)
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        # Choose offset from image corners to plot detected corners
        # This should be chosen to present the result at the proper aspect ratio
        # My choice of 100 pixels is not exact, but close enough for our purpose here
        offset = 500 # offset for dst points
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])

        # For source points I'm grabbing the outer four detected corners
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result 
        # again, not exact, but close enough for our purposes
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                     [img_size[0]-offset, img_size[1]-offset], 
                                     [offset, img_size[1]-offset]])
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist, M, img_size)

    # Return the resulting image and matrix
    return undist, warped

def sobel(gray, orient='x', sobel_kernel=3):
    
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient is 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient is 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    else:
        print('Error: unexpected orient')

    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    return scaled_sobel

def binary_thresh(value, thresh):
    
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(value)
    binary_output[(value >= thresh[0]) & (value <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output


# Define a function to threshold an image for a given range and Sobel kernel
def absgraddir(sobelx, sobely):
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    return np.arctan2(sobely, sobelx)*180/3.1415926

def combine_binary(binary1, binary2, type):
  # Stack each channel to view their individual contributions in green and blue respectively
  # This returns a stack of the two binary images, whose components you can see as different colors
  color_binary = np.dstack(( np.zeros_like(binary1), binary1, binary2))

  # Combine the two binary thresholds
  combined_binary = np.zeros_like(binary1)
  if type == '&':
    combined_binary[(binary1 == 1) & (binary2 == 1)] = 1
  elif type == '|':
    combined_binary[(binary1 == 1) | (binary2 == 1)] = 1
  else:
    print('error')
     

  return combined_binary, color_binary

def undistort(img, mtx, dist):
  return cv2.undistort(img, mtx, dist, None, mtx)

def undistort_file(filename, mtx, dist):
  img = read_img(filename)
  undist = undistort(img, mtx, dist)
  return undist

def thresh(img, sx_thresh=(20,100), sy_thresh=(20,100), dir_thresh=(100,200), s_thresh=(170,255)):
  
  gray = img2gray(img)

  # Sobel
  sobelx = sobel(gray, 'x')
  sobely = sobel(gray, 'y')
  direction = absgraddir(sobelx, sobely)
 # print(np.max(direction))
 # print(np.min(direction))

  # Convert to HLS color space and separate the S channel
  # Note: img is the undistorted image
  hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
  s_channel = hls[:,:,2]

  # Threshold
  sx_binary = binary_thresh(sobelx, sx_thresh)
  sy_binary = binary_thresh(sobely, sy_thresh)
  dir_binary = binary_thresh(direction, dir_thresh)
  s_binary = binary_thresh(s_channel, s_thresh)
  return sx_binary, sy_binary, dir_binary, s_binary

def plot_binary(img, combined_binary, color_binary):
  # Plotting thresholded images
 f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
 ax1.set_title('Stacked thresholds')
 ax1.imshow(color_binary*255)

 ax2.set_title('Combined S channel and gradient thresholds')
 ax2.imshow(combined_binary, cmap='gray')

 ax3.set_title('Original image')
 ax3.imshow(img2gray(img), cmap='gray')

 #plt.savefig('thresh.png')

 plt.show()

def combine_thresh(sx_binary, sy_binary, dir_binary, s_binary):
  combined_binary1, color_binary1 = combine_binary(sx_binary, s_binary, '&')
  combined_binary2, color_binary2 = combine_binary(combined_binary1, dir_binary, '&')

  combined_binary3, color_binary3 = combine_binary(sy_binary, s_binary, '&')
  combined_binary4, color_binary4 = combine_binary(combined_binary3, dir_binary, '&')

  combined_binary, color_binary = combine_binary(combined_binary2, combined_binary4, '|')
  return combined_binary, color_binary



def warp(img, src, offset):

  offsetx = offset[0] # offset for dst points
  offsety = offset[1]
  # Grab the image shape
  img_size = (img.shape[1], img.shape[0])


  # For destination points, I'm arbitrarily choosing some points to be
  # a nice fit for displaying our warped result 
  # again, not exact, but close enough for our purposes
  dst = np.float32([[offsetx,               offsety ] , 
                    [img_size[0] - offsetx, offsety] , 
                    [img_size[0] - offsetx, img_size[1] - offsety] , 
                    [offsetx,               img_size[1] - offsety] ])
  # Given src and dst points, calculate the perspective transform matrix
  M = cv2.getPerspectiveTransform(src, dst)
  return M, img_size

def warp_and_thresh(filename, mtx, dist):

  undist = undistort_file(filename, mtx, dist)

  # For source points I'm grabbing the outer four detected corners
  src = np.float32([(595, 450), (680, 450), (1080, 720), (230, 720), ])
  offset = [400, 0] # offset for dst points
  M, img_size = warp(undist, src, offset)

  warped = cv2.warpPerspective(undist, M, img_size)

  #sx_thresh=(10,255)
  #sy_thresh=(10,255)
  #dir_thresh=(0, 180)
  #s_thresh=(100,255)

  sx_thresh=(10, 255)
  sy_thresh=(5,255)
  dir_thresh=(0, 60)
  s_thresh=(70,255)
  sx_binary, sy_binary, dir_binary, s_binary = thresh(warped, sx_thresh, sy_thresh, dir_thresh, s_thresh )
  combined_binary, color_binary = combine_thresh(sx_binary, sy_binary, dir_binary, s_binary)

  return undist, warped, combined_binary

def plot_warped_binary(axs, i, warped, warped_binary):
    # Warp the image using OpenCV warpPerspective()
    #warped = cv2.warpPerspective(undist, M, img_size)
    #warped_binary = cv2.warpPerspective(combined_binary, M, img_size)
    #axs[i,0].imshow(img2RGB(undist))
    axs[i,0].imshow(img2RGB(warped))
    axs[i,1].imshow(warped_binary, cmap='gray')

def fitpoly(fname, mtx, dist):
  undist, warped, binary_warped = warp_and_thresh(fname, mtx, dist)

  # Assuming you have created a warped binary image called "binary_warped"
  # Take a histogram of the bottom half of the image
  histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
  # Create an output image to draw on and  visualize the result
  out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
  # Find the peak of the left and right halves of the histogram
  # These will be the starting point for the left and right lines
  midpoint = np.int(histogram.shape[0]/2)
  leftx_base = np.argmax(histogram[:midpoint])
  rightx_base = np.argmax(histogram[midpoint:]) + midpoint

  # Choose the number of sliding windows
  nwindows = 9
  # Set height of windows
  window_height = np.int(binary_warped.shape[0]/nwindows)
  # Identify the x and y positions of all nonzero pixels in the image
  nonzero = binary_warped.nonzero()
  nonzeroy = np.array(nonzero[0])
  nonzerox = np.array(nonzero[1])
  # Current positions to be updated for each window
  leftx_current = leftx_base
  rightx_current = rightx_base
  # Set the width of the windows +/- margin
  margin = 100
  # Set minimum number of pixels found to recenter window
  minpix = 50
  # Create empty lists to receive left and right lane pixel indices
  left_lane_inds = []
  right_lane_inds = []

  # Step through the windows one by one
  for window in range(nwindows):
      # Identify window boundaries in x and y (and right and left)
      win_y_low = binary_warped.shape[0] - (window+1)*window_height
      win_y_high = binary_warped.shape[0] - window*window_height
      win_xleft_low = leftx_current - margin
      win_xleft_high = leftx_current + margin
      win_xright_low = rightx_current - margin
      win_xright_high = rightx_current + margin
      # Draw the windows on the visualization image
      cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
      cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
      # Identify the nonzero pixels in x and y within the window
      good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
      good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
      # Append these indices to the lists
      left_lane_inds.append(good_left_inds)
      right_lane_inds.append(good_right_inds)
      # If you found > minpix pixels, recenter next window on their mean position
      if len(good_left_inds) > minpix:
          leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
      if len(good_right_inds) > minpix:        
          rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

  # Concatenate the arrays of indices
  left_lane_inds = np.concatenate(left_lane_inds)
  right_lane_inds = np.concatenate(right_lane_inds)

  # Extract left and right line pixel positions
  leftx = nonzerox[left_lane_inds]
  lefty = nonzeroy[left_lane_inds] 
  rightx = nonzerox[right_lane_inds]
  righty = nonzeroy[right_lane_inds] 

  # Fit a second order polynomial to each
  left_fit = np.polyfit(lefty, leftx, 2)
  right_fit = np.polyfit(righty, rightx, 2)

  ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
  left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
  right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

  out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
  out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
  f,axs = plt.subplots(1,3)
  axs[0].imshow(binary_warped, cmap='gray')
  axs[1].imshow(out_img)
  axs[1].plot(left_fitx, ploty, color='yellow')
  axs[1].plot(right_fitx, ploty, color='yellow')
  plt.xlim(0, 1280)
  plt.ylim(720, 0)
  axs[2].imshow(img2RGB(warped))
  #plt.show()
  plt.savefig(fname + '_out.jpg')
  plt.close()

  return left_fit, right_fit

def do_stuff(savename):
  do_cal = False
  check_cal = False
  do_plot_binary = False
  do_plot_warped_binary = True

  nx = 9
  ny = 6

  if do_cal:
    run_calibration('camera_cal/calibration*.jpg', nx, ny)

  mtx, dist = load_calibration()

  if check_cal:
    check_calibration('test_images/test1.jpg', nx, ny, mtx, dist)

  #filenames = glob.glob('test_images/*.jpg')
  filenames = ['test_images/test5.jpg']

  if do_plot_warped_binary:
    f, axs = plt.subplots(len(filenames), 2, figsize=(20,10))
  count = 0
  for i in range(len(filenames)):
    print(filenames[i])
    if do_plot_warped_binary:
      plot_warped_binary
    left_fit, right_fit = fitpoly(binary_warped, warped)


  if do_plot_warped_binary:
    #plt.show()
    plt.savefig(savename)
    plt.close()


  return 1


