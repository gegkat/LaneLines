import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
from os.path import basename, splitext

class limg(object):
  def __init__(self, img, M, Minv, mtx, dist, name='curr', show_plot=False, save_plot=True):
    self.img = img
    self.img_shape = (img.shape[1], img.shape[0])
    self.M = M
    self.Minv = Minv
    self.mtx = mtx
    self.dist = dist
    self.name = splitext(basename(name))[0]
    self.savename = self.name + '_out.png'
    self.show_plot = show_plot
    self.save_plot = save_plot

    self.calc_undistort()
    self.get_warp()

    self.sx_thresh=(10, 255)
    self.sy_thresh=(5,255)
    self.dir_thresh=(0, 10)
    self.gray_thresh=(30,255)

    # Define conversions in x and y from pixels space to meters
    self.ym_per_pix = 30/720 # meters per pixel in y dimension
    self.xm_per_pix = 3.7/700 # meters per pixel in x dimension

    self.mixing_factor = 0.5
    self.get_gray()

    self.do_thresh()
    self.get_combined_binary()
    self.fitpoly()
    self.get_rect_plots()
    self.get_curvature()
    self.plot_process()

    # Now our radius of curvature is in meters
    print("Curvature. Left: {:.0f} m Right: {:.0f} m".format(self.left_curverad, self.right_curverad))


  def calc_undistort(self):
    self.undist = cv2.undistort(self.img, self.mtx, self.dist, None, self.mtx)

  def get_warp(self):
    self.warped = cv2.warpPerspective(self.undist, self.M, self.img_shape)

  def get_gray(self):
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(self.warped, cv2.COLOR_BGR2HLS)
    self.gray = hls[:,:,1]*(1-self.mixing_factor) + hls[:,:,2]*self.mixing_factor

  def do_thresh(self):
    # Sobel
    self.sobelx = sobel(self.gray, 'x')
    self.sobely = sobel(self.gray, 'y')
    self.sobeldir = absgraddir(self.sobelx, self.sobely)

    # Threshold
    self.sx_binary =   binary_thresh(self.sobelx,   self.sx_thresh)
    self.sy_binary =   binary_thresh(self.sobely,   self.sy_thresh)
    self.dir_binary =  binary_thresh(self.sobeldir, self.dir_thresh)
    self.gray_binary = binary_thresh(self.gray,     self.gray_thresh)

  def get_combined_binary(self):
    combined_binary1 = combine_binary(self.sx_binary, self.gray_binary, '&')
    combined_binary2 = combine_binary(combined_binary1, self.dir_binary, '&')

    combined_binary3 = combine_binary(self.sy_binary, self.gray_binary, '&')
    combined_binary4 = combine_binary(combined_binary3, self.dir_binary, '&')

    self.binary_warped = combine_binary(combined_binary2, combined_binary4, '|')

  def fitpoly(self):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(self.binary_warped[self.binary_warped.shape[0]/2:,:], axis=0)

    # Create an output image to draw on and  visualize the result
    self.out_img = np.dstack((self.binary_warped, self.binary_warped, self.binary_warped))*255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9

    # Set height of windows
    window_height = np.int(self.binary_warped.shape[0]/nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = self.binary_warped.nonzero()
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
        win_y_low = self.binary_warped.shape[0] - (window+1)*window_height
        win_y_high = self.binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(self.out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(self.out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
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

    self.out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    self.out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Extract left and right line pixel positions
    self.leftx = nonzerox[left_lane_inds]
    self.lefty = nonzeroy[left_lane_inds] 
    self.rightx = nonzerox[right_lane_inds]
    self.righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    self.left_fit = np.polyfit(self.lefty, self.leftx, 2)
    self.right_fit = np.polyfit(self.righty, self.rightx, 2)


  def get_curvature(self):
    y_eval = np.max(self.ploty)

    # Fit new polynomials to x,y in world space
    left_fit_cr =  np.polyfit(self.ploty*self.ym_per_pix, self.left_fitx*self.xm_per_pix, 2)
    right_fit_cr = np.polyfit(self.ploty*self.ym_per_pix, self.right_fitx*self.xm_per_pix, 2)
    # Calculate the new radii of curvature
    self.left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*self.ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    self.right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*self.ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

  def plot_process(self):
    f,axs = plt.subplots(2,3, figsize=(28,16))
    axs[0,0].imshow(img2RGB(self.undist))
    axs[0,1].imshow(self.gray, cmap='gray')
    axs[0,2].imshow(self.binary_warped, cmap='gray')


    self.plot_rect_method(axs[1,0])
    self.plot_2d_lane(axs[1,1])
    self.plot_lane(axs[1,2])
    if self.save_plot:
      plt.savefig(self.savename)
    if self.show_plot:
      plt.show()
    plt.close()

  def get_rect_plots(self):
    self.ploty = np.linspace(0, self.binary_warped.shape[0]-1, self.binary_warped.shape[0] )
    self.left_fitx = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
    self.right_fitx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]

  def plot_rect_method(self, ax):
    ax.imshow(self.out_img)
    self.plot_2d_lane_fits(ax, color='yellow')
    ax.set_xlim(0, self.img_shape[0])
    ax.set_ylim(self.img_shape[1], 0)

  def plot_2d_lane(self, ax):
    ax.imshow(img2RGB(self.warped))
    self.plot_2d_lane_fits(ax, color='green')
    ax.set_xlim(0, self.img_shape[0])
    ax.set_ylim(self.img_shape[1], 0)

  def plot_2d_lane_fits(self, ax, color='yellow'):
    ax.plot(self.left_fitx,  self.ploty, color=color)
    ax.plot(self.right_fitx, self.ploty, color=color)

  def plot_lane(self, ax):
    # Create an image to draw the lines on
    color_warp = np.zeros_like(self.undist).astype(np.uint8)
    #color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, self.Minv, self.img_shape) 
    # Combine the result with the original image
    result = cv2.addWeighted(self.undist, 1, newwarp, 0.3, 0)
    ax.imshow(img2RGB(result))

def read_img(filename):
  img = cv2.imread(filename)
  print('Read image: {} Size: {}'.format(filename, img.shape))
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

  undist = undistort_file(images[0], mtx, dist)

  # For source points I'm grabbing the outer four detected corners
  src = np.float32([(595, 450), (680, 450), (1080, 720), (230, 720), ])
  offset = [300, 0] # offset for dst points
  M, Minv, img_size = warp(undist, src, offset)

  dist_pickle = {}
  dist_pickle["mtx"] = mtx
  dist_pickle["dist"] = dist
  dist_pickle["M"] = M
  dist_pickle["Minv"] = Minv
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
  M = dist_pickle["M"]
  Minv = dist_pickle["Minv"]
  return mtx, dist, M, Minv

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
  # Combine the two binary thresholds
  combined_binary = np.zeros_like(binary1)
  if type == '&':
    combined_binary[(binary1 == 1) & (binary2 == 1)] = 1
  elif type == '|':
    combined_binary[(binary1 == 1) | (binary2 == 1)] = 1
  else:
    print('Error: expected type to be & or |')
     
  return combined_binary

def undistort(img, mtx, dist):
  return cv2.undistort(img, mtx, dist, None, mtx)

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
  Minv = cv2.getPerspectiveTransform(dst, src)
  return M, Minv, img_size

def plot_warped_binary(axs, i, warped, warped_binary):
    # Warp the image using OpenCV warpPerspective()
    #warped = cv2.warpPerspective(undist, M, img_size)
    #warped_binary = cv2.warpPerspective(combined_binary, M, img_size)
    #axs[i,0].imshow(img2RGB(undist))
    axs[i,0].imshow(img2RGB(warped))
    axs[i,1].imshow(warped_binary, cmap='gray')



def do_stuff(show_plot=True, save_plot=True):
  do_cal = False
  check_cal = False
  do_plot_binary = False
  do_plot_warped_binary = True

  nx = 9
  ny = 6

  if do_cal:
    run_calibration('camera_cal/calibration*.jpg', nx, ny)

  mtx, dist, M, Minv = load_calibration()

  if check_cal:
    check_calibration('test_images/test1.jpg', nx, ny, mtx, dist)

  filenames = glob.glob('test_images/*.jpg')
  #filenames = ['test_images/test5.jpg']

  for i in range(len(filenames)):
    img = read_img(filenames[i])
    ll = limg(img, M, Minv, mtx, dist, filenames[i], show_plot=show_plot, save_plot=save_plot)

  return 1


