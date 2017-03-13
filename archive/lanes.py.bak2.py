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

def do_stuff(savename):
  do_cal = False
  check_cal = False
  do_plot_binary = False


  nx = 9
  ny = 6

  if do_cal:
    run_calibration('camera_cal/calibration*.jpg', nx, ny)

  mtx, dist = load_calibration()

  if check_cal:
    check_calibration('test_images/test1.jpg', nx, ny, mtx, dist)


  #undist = undistort_file('test_images/straight_lines1.jpg', mtx, dist)
  #undist = undistort_file('test_images/straight_lines2.jpg', mtx, dist)
  #undist = undistort_file('test_images/test1.jpg', mtx, dist)
  #undist = undistort_file('test_images/test5.jpg', mtx, dist)

  filenames = glob.glob('test_images/*.jpg')
  filenames = ['imgaes/test1.jpg']
  f, axs = plt.subplots(len(filenames), 2, figsize=(20,10))
  count = 0
  for i in range(len(filenames)):
    print(filenames[i])
    undist, warped, warped_binary = warp_and_thresh(filenames[i], mtx, dist)


    # Warp the image using OpenCV warpPerspective()
    #warped = cv2.warpPerspective(undist, M, img_size)
    #warped_binary = cv2.warpPerspective(combined_binary, M, img_size)
    #axs[i,0].imshow(img2RGB(undist))
    axs[i,0].imshow(img2RGB(warped))
    axs[i,1].imshow(warped_binary, cmap='gray')
  #plt.show()
  plt.savefig(savename)
  plt.close()

  # Assuming you have created a warped binary image called "binary_warped"
  # Take a histogram of the bottom half of the image
  histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)

  return 1

