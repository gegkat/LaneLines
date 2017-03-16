import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
from os.path import basename, splitext
from moviepy.editor import VideoFileClip, ImageSequenceClip

class Line():
  def __init__(self):
    # x values of the last n fits of the line
    self.recent_xfitted = [] 

    # average x values of the fitted line over the last n iterations
    self.bestx = None     

    # polynomial coefficients averaged over the last n iterations
    self.best_fit = None  

    # polynomial coefficients for the most recent fit
    self.current_fit = [np.array([False])]  

    # radius of curvature of the line in some units
    self.radius_of_curvature = None 

    # distance in meters of vehicle center from the line
    self.line_base_pos = None 

    # difference in fit coefficients between last and new fits
    self.diff = None

    # Define conversions in x and y from pixels space to meters
    self.ym_per_pix = 30/720 # meters per pixel in y dimension
    self.xm_per_pix = 3.7/700 # meters per pixel in x dimension

  def preprocess(self, fitx, ploty, img_shape):
    self.current_fit = np.polyfit(fitx, ploty, 2)  
    self.curr_curvature, self.curr_distance = self.get_geometry(fitx, ploty, img_shape) 

    if self.best_fit is None:
      self.diff = 0
    else:
      self.diff = sum(abs((self.current_fit - self.best_fit)/self.best_fit))

  def update(self, detected, reset, fitx, ploty, img_shape):
    if detected:
      self.undetected_counter = 0
      self.recent_xfitted.append(fitx)
      if len(self.recent_xfitted) > 5:
        self.recent_xfitted.pop(0)

      self.bestx = sum(self.recent_xfitted)/len(self.recent_xfitted)     
      self.best_fit = np.polyfit(self.bestx, ploty, 2)  
      self.radius_of_curvature, self.distance = self.get_geometry(self.bestx, ploty, img_shape) 
    
    if reset:
        self.best_fit = None
        self.recent_xfitted = []

  def get_geometry(self, x, y, img_shape):
    y_eval = np.max(y)

    # Fit new polynomials to x,y in world space
    fit_cr =  np.polyfit(y*self.ym_per_pix, x*self.xm_per_pix, 2)
    # Calculate the new radii of curvature
    curvature = ((1 + (2*fit_cr[0]*y_eval*self.ym_per_pix  + fit_cr[1])**2)**1.5)  / np.absolute(2*fit_cr[0])

    distance = np.absolute(fit_cr[2] - self.xm_per_pix*img_shape[0]/2)
    return curvature, distance



class limg(object):
  def __init__(self, M, Minv, mtx, dist, show_plot=False, save_plot=True):

    self.M = M
    self.Minv = Minv
    self.mtx = mtx
    self.dist = dist

    self.show_plot = show_plot
    self.save_plot = save_plot

    self.sx_thresh=(10, 255)
    self.sy_thresh=(5,255)
    self.dir_thresh=(0, 10)
    self.gray_thresh=(30,255)

    self.mixing_factor = 0.5

    self.detected = True
    self.reset = True
    self.diff_thresh = 1000
    self.undetected_counter = 0

    self.undected_counter_thresh = 3

    self.width_expected = 3.7
    self.width_thresh = 1.0
    self.distance_from_center_thresh = 2

    # Choose the number of sliding windows
    self.nwindows = 9

    # Set the width of the windows +/- margin
    self.margin = 100

    # Set minimum number of pixels found to recenter window
    self.minpix = 50

    self.img_shape = None
    self.ploty = None

    self.left_line = Line();
    self.right_line = Line();

    self.frame_count = 0

    self.initialized = False

  def process_img(self, img, name='curr'):
    self.img = img
    self.frame_count = self.frame_count + 1

    if self.ploty is None:
      self.img_shape = (img.shape[1], img.shape[0])
      self.ploty = np.linspace(0, self.img_shape[0]-1, self.img_shape[0] )

    self.name = splitext(basename(name))[0]
    self.savename = 'out_imgs/' + self.name + '_out.png'

    self.calc_undistort()
    self.get_warp()
    self.get_gray()

    self.do_thresh()
    self.get_combined_binary()

    # Create an output image to draw on and  visualize the result
    self.out_img = np.dstack((self.binary_warped, self.binary_warped, self.binary_warped))*255

    if self.reset:
      print('window method')
      self.get_lane_inds_windows()
      self.used_windows = True
      self.reset = False
      self.initialized = False
    else:
      self.get_lane_inds_margin()
      self.used_windows = False

    self.out_img[self.nonzeroy[self.left_lane_inds],  self.nonzerox[self.left_lane_inds]] = [255, 0, 0]
    self.out_img[self.nonzeroy[self.right_lane_inds], self.nonzerox[self.right_lane_inds]] = [0, 255, 0]
    if not self.used_windows:
      self.margin_plot_calcs()


    self.fitpoly()

    self.get_rect_plot_points()
    #self.get_curvature()
    self.preprocess_lines()

    self.is_good_line()
    if not self.detected:
      self.undetected_counter = self.undetected_counter + 1
      if self.undetected_counter > self.undected_counter_thresh:
        print("Line reset")
        self.reset = True
        self.undetected_counter = 0
    else: 
      self.undetected_counter = 0

    self.update_lines()

    self.get_lane_img()
    self.curvature = (self.left_line.radius_of_curvature + self.right_line.radius_of_curvature)/2
    self.distance_from_center = (self.left_line.distance - self.right_line.distance)/2
    self.width = self.left_line.distance+self.right_line.distance
    self.add_text_to_lane_img()
    if self.show_plot or self.save_plot:
      self.plot_process()
          # Now our radius of curvature is in meters
    # print("Curvature. Left: {:.0f} m Right: {:.0f} m".format(self.left_curverad, self.right_curverad))

  def is_good_line(self):
    self.detected = True
    
    if self.left_line.diff + self.right_line.diff > self.diff_thresh:
      print("Line rejected on bad fit")
      self.detected = False

    self.curr_width = self.left_line.curr_distance + self.right_line.curr_distance
    if np.absolute(self.curr_width - self.width_expected) > self.width_thresh:
      print("Line rejected on bad width")
      self.detected = False

    self.curr_distance_from_center = (self.left_line.curr_distance - self.right_line.curr_distance)/2
    if np.absolute(self.curr_distance_from_center) > self.distance_from_center_thresh:
      print("Line rejected on bad distance from center")
      self.detected = False

    if self.initialized is False:
      self.detected = True
      self.initialized = True

  def preprocess_lines(self):
    self.left_line.preprocess( self.left_fitx,  self.ploty, self.img_shape)
    self.right_line.preprocess(self.right_fitx, self.ploty, self.img_shape)

  def update_lines(self):
    self.left_line.update( self.detected, self.reset, self.left_fitx,  self.ploty, self.img_shape)
    self.right_line.update(self.detected, self.reset, self.right_fitx, self.ploty, self.img_shape)

  def calc_undistort(self):
    self.undist = cv2.undistort(self.img, self.mtx, self.dist, None, self.mtx)

  def get_warp(self):
    self.warped = cv2.warpPerspective(self.undist, self.M, self.img_shape)

  def get_gray(self):
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(self.warped, cv2.COLOR_RGB2HLS)
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

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = self.binary_warped.nonzero()
    self.nonzeroy = np.array(nonzero[0])
    self.nonzerox = np.array(nonzero[1])

  def get_lane_inds_windows(self):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(self.binary_warped[self.binary_warped.shape[0]/2:,:], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(self.binary_warped.shape[0]/self.nwindows)

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(self.nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = self.binary_warped.shape[0] - (window+1)*window_height
        win_y_high = self.binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - self.margin
        win_xleft_high = leftx_current + self.margin
        win_xright_low = rightx_current - self.margin
        win_xright_high = rightx_current + self.margin
        # Draw the windows on the visualization image
        cv2.rectangle(self.out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(255,255,0), 2) 
        cv2.rectangle(self.out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(255,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds =  ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) & (self.nonzerox >= win_xleft_low)  & (self.nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) & (self.nonzerox >= win_xright_low) & (self.nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > self.minpix:
            leftx_current = np.int(np.mean(self.nonzerox[good_left_inds]))
        if len(good_right_inds) > self.minpix:        
            rightx_current = np.int(np.mean(self.nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    self.left_lane_inds = np.concatenate(left_lane_inds)
    self.right_lane_inds = np.concatenate(right_lane_inds)
 
  def get_lane_inds_margin(self):
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!

    self.left_lane_inds =  ((self.nonzerox > (self.left_fit[0]*(self.nonzeroy**2)  + self.left_fit[1]*self.nonzeroy  + self.left_fit[2] - self.margin))  & (self.nonzerox < (self.left_fit[0]*(self.nonzeroy**2)  + self.left_fit[1]*self.nonzeroy  + self.left_fit[2]  + self.margin))) 
    self.right_lane_inds = ((self.nonzerox > (self.right_fit[0]*(self.nonzeroy**2) + self.right_fit[1]*self.nonzeroy + self.right_fit[2] - self.margin)) & (self.nonzerox < (self.right_fit[0]*(self.nonzeroy**2) + self.right_fit[1]*self.nonzeroy + self.right_fit[2] + self.margin)))  

  def fitpoly(self):
    # Fit a second order polynomial to each
    self.left_fit =  np.polyfit(self.nonzeroy[self.left_lane_inds] , self.nonzerox[self.left_lane_inds],  2)
    self.right_fit = np.polyfit(self.nonzeroy[self.right_lane_inds], self.nonzerox[self.right_lane_inds], 2)

  def plot_process(self):
    f,axs = plt.subplots(2,3, figsize=(28,16))
    axs[0,0].imshow(self.undist)
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

  def get_rect_plot_points(self):
    self.left_fitx = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
    self.right_fitx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]

  def margin_plot_calcs(self):
      # Create an image to draw on and an image to show the selection window
      window_img = np.zeros_like(self.out_img)

      # Generate a polygon to illustrate the search window area
      # And recast the x and y points into usable format for cv2.fillPoly()
      left_line_window1 = np.array([np.transpose(np.vstack([self.left_fitx-self.margin, self.ploty]))])
      left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.left_fitx+self.margin, self.ploty])))])
      left_line_pts = np.hstack((left_line_window1, left_line_window2))
      right_line_window1 = np.array([np.transpose(np.vstack([self.right_fitx-self.margin, self.ploty]))])
      right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx+self.margin, self.ploty])))])
      right_line_pts = np.hstack((right_line_window1, right_line_window2))

      # Draw the lane onto the warped blank image
      cv2.fillPoly(window_img, np.int_([left_line_pts]), (255,255,0))
      cv2.fillPoly(window_img, np.int_([right_line_pts]), (255,255,0))
      self.out_img = cv2.addWeighted(self.out_img, 1, window_img, 0.0, 0)

  def plot_rect_method(self, ax):
    ax.imshow(self.out_img)
    self.plot_2d_lane_fits(ax, color='yellow')
    ax.set_xlim(0, self.img_shape[0])
    ax.set_ylim(self.img_shape[1], 0)

  def plot_2d_lane(self, ax):
    ax.imshow(self.warped)
    self.plot_2d_lane_fits(ax, color='green')
    ax.set_xlim(0, self.img_shape[0])
    ax.set_ylim(self.img_shape[1], 0)

  def plot_2d_lane_fits(self, ax, color='yellow'):
    ax.plot(self.left_fitx,  self.ploty, color=color)
    ax.plot(self.right_fitx, self.ploty, color=color)

  def get_lane_img(self):
    # Create an image to draw the lines on
    color_warp = np.zeros_like(self.undist).astype(np.uint8)
    #color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    #print(self.left_line.bestx.shape)
    #print(self.ploty.shape)
    pts_left = np.array([np.transpose(np.vstack([self.left_line.bestx, self.ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_line.bestx, self.ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), [0,0,255])

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, self.Minv, self.img_shape) 
    newwarp2 = cv2.warpPerspective(self.out_img, self.Minv, self.img_shape) 
    # Combine the result with the original image
    self.lane_img = cv2.addWeighted(self.undist, 1.0, newwarp, 0.3, 0)
    self.lane_img = cv2.addWeighted(self.lane_img, 1, newwarp2, 1.0, 0)

  def plot_lane(self, ax):
    ax.imshow(self.lane_img)

  def add_text_to_lane_img(self):
    s = "Radius of Curvature = {:.0f}(m)".format(self.curvature)
    s2 = "Vehicle is {:.02f}m from center. Width: {:.01f}m".format(self.distance_from_center, self.width)
    #s4 = "Vehicle is {:.02f}m from center. Width: {:.01f}m".format(self.curr_distance_from_center, self.curr_width)
    s3 = "Video Frame: {:04}: Line Detected: {}  Reset: {}".format(self.frame_count, self.detected, self.reset)
    #s = "Diff: {:.01f}, {:.01f}".format(self.diff[0], self.diff[1])
    #s = "{}: {}, {}".format(self.frame_count, self.detected, self.reset)
    cv2.putText(self.lane_img, s, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    cv2.putText(self.lane_img, s2, (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    cv2.putText(self.lane_img, s3, (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, 255) 
    #cv2.putText(self.lane_img, s4, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, 255) 

def read_img(filename):
  img = cv2.imread(filename)
  #print('Read image: {} Size: {}'.format(filename, img.shape))
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
  mtx, dist = get_calibration(images, nx, ny)

  img = read_img(images[0])
  undist = cv2.undistort(img, mtx, dist, None, mtx)

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


def do_stuff(show_plot=True, save_plot=True, fname='project_video.mp4', MAX_FRAMES=10000, n_mod=1, fps=16):
  do_cal = False
  check_cal = False

  nx = 9
  ny = 6

  if do_cal:
    run_calibration('camera_cal/calibration*.jpg', nx, ny)

  mtx, dist, M, Minv = load_calibration()

  if check_cal:
    check_calibration('camera_cal/calibration1.jpg', mtx, dist)

  filenames = glob.glob('test_images/*.jpg')
  #filenames = ['test_images/test5.jpg']

  ll = limg(M, Minv, mtx, dist, show_plot=show_plot, save_plot=save_plot)
  #for i in range(len(filenames)):
  #  img = read_img(filenames[i])
  #  img = img2RGB(img)
  #  ll.process_img(img, filenames[i])


  clip = VideoFileClip(fname)
  n_frames = int(clip.fps * clip.duration)
  n_frames = min(MAX_FRAMES, n_frames) 
  count = 0
  images_list = []
  for frame in clip.iter_frames():
    count = count+1
    if count % n_mod == 0:
      print("{} of {}".format(count, n_frames))
      ll.process_img(frame, str(count))
      images_list.append(ll.lane_img)
    if count >= MAX_FRAMES:
      break

  clip = ImageSequenceClip(images_list, fps=fps)

  savename = splitext(basename(fname))[0]
  savename = 'out_imgs/' + savename + '_out.mp4'
  clip.write_videofile(savename) # default codec: 'libx264', 24 fps


  return ll, clip




