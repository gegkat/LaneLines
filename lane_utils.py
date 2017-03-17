import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
from os.path import basename, splitext

def read_img(filename):
  img = cv2.imread(filename)
  return img

def img2gray(img):
  return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def img2RGB(img):
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def img2BGR(img):
  return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

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

  # Get calibration matrices from images
  mtx, dist = get_calibration(images, nx, ny)

  # Use first image for calculating warp matrices
  M, Minv = get_warp_matrices(images[0])

  # Save calibration and Warp matrices to pickle
  dist_pickle = {}
  dist_pickle["mtx"] = mtx
  dist_pickle["dist"] = dist
  dist_pickle["M"] = M
  dist_pickle["Minv"] = Minv
  pickle.dump( dist_pickle, open( "camera_cal/cal.p", "wb" ) )

def get_warp_matrices(fname):

  # Read image
  img = read_img(fname)

  # Undistort
  undist = cv2.undistort(img, mtx, dist, None, mtx)

  # Source points around edges of lane lines for warp matrices
  src = np.float32([(595, 450), (680, 450), (1080, 720), (230, 720), ])
  offset = [300, 0] # offset for dst points

  # Get warp  matrices
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
  Minv = cv2.getPerspectiveTransform(dst, src)
  return M, Minv

def check_calibration(filename, mtx, dist):
  img = read_img(filename)
  undist = cv2.undistort(img, mtx, dist, None, mtx)

  plt.figure()
  plt.imshow(img)
  plt.figure()
  plt.imshow(undist)
  plt.show()
  plt.close()

def load_calibration():
  dist_pickle = pickle.load( open( "camera_cal/cal.p", "rb" ) )
  mtx = dist_pickle["mtx"]
  dist = dist_pickle["dist"]
  M = dist_pickle["M"]
  Minv = dist_pickle["Minv"]
  return mtx, dist, M, Minv

def sobel(gray, orient='x', sobel_kernel=3):
    
    # Take the derivative in x or y given orient = 'x' or 'y'
    if orient is 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient is 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    else:
        print('Error: unexpected orient')

    # Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)

    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    return scaled_sobel

def binary_thresh(value, thresh):
    
    # Create a mask of 1's where the scaled gradient magnitude 
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(value)
    binary_output[(value >= thresh[0]) & (value <= thresh[1])] = 1

    # Return this mask as your binary_output image
    return binary_output


# Define a function to threshold an image for a given range and Sobel kernel
def absgraddir(sobelx, sobely):
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    return np.arctan2(sobely, sobelx)*180/3.1415926

def combine_binary(binary1, binary2, type):
  # Combine the two binary thresholds
  combined_binary = np.zeros_like(binary1)
  if type == '&':
    combined_binary[(binary1 == 1) & (binary2 == 1)] = 1
  elif type == '|':
    combined_binary[(binary1 == 1) | (binary2 == 1)] = 1
  else:
    print('Error: expected type to be & or |')
     
  return combined_binary
