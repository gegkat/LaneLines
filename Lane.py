import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
from os.path import basename, splitext
import Line
from lane_utils import *

# Lane class contains two line objects which together form a lane
# Processes images to determine line pixels that form basis of
# lane lines
class Lane(object):
  def __init__(self, M, Minv, mtx, dist, show_plot=False, save_plot=True):

    # Initialize warp and inverse warp matrices for going between
    # original camera image and top down view of lane lines
    self.M = M
    self.Minv = Minv

    # Initialize distortion matrices
    self.mtx = mtx
    self.dist = dist

    # Flags for controlling plotting behavior
    self.show_plot = show_plot
    self.save_plot = save_plot



    # Thresholds to form binary image in lane detection
    # self.sx_thresh=(10, 255)
    # self.sy_thresh=(5,255)
    # self.dir_thresh=(0, 10)
    # self.gray_thresh=(30,255)

    self.sx_thresh=(6, 255)
    self.sy_thresh=(3,255)
    self.dir_thresh=(0, 30)
    self.gray_thresh=(15,255)

    # Mixing factor for averaging together L and S channels of
    # HLS color space
    self.mixing_factor = 0.5

    # Initialize logic states and lane detection
    self.detected = True
    self.reset = True

    # Initialize undetected counter. This increments on 
    # consecutive lane detection failures
    self.undetected_counter = 0

    # Threshold to reset lane lines on consecutive failed
    # lane detections. On reset the filter for averaging
    # n lines is reset and the window method to locate the 
    # line is used
    self.undected_counter_thresh = 3

    # Threshold to invalidate line based on the difference of
    # the polynomial fit compared to the best fit
    self.diff_thresh = 1000

    # Expected width between lane lines in meter
    self.width_expected = 3.7

    # Threshold of deviation from expected width for invalidating
    # lane line detection in eters
    self.width_thresh = 1.0

    # Threshold for deviation from center for invalidating 
    # lane line detection in meters
    self.distance_from_center_thresh = 2

    # The number of sliding windows
    self.nwindows = 9

    # Set the width of the windows +/- margin
    self.margin = 100

    # Set minimum number of pixels found to recenter window
    self.minpix = 50

    # Size of image in pixels 
    self.img_shape = None

    # y points for fitted lane line points
    self.ploty = None

    # Line objects that make up a lane
    self.left_line = Line.Line();
    self.right_line = Line.Line();

    # Count of frames processed
    self.frame_count = 0

    # image matrix
    self.img = None
    self.savename = None

  # Main function of Lane object that ingests and processes a new image
  def process_img(self, img, name='curr'):

    # Record img matrix to object
    self.img = img

    # Increment frame count
    self.frame_count = self.frame_count + 1

    # On the first time we have seen an image, record the shape and establish the ploty
    if self.ploty is None:
      self.img_shape = (img.shape[1], img.shape[0])
      self.ploty = np.linspace(0, self.img_shape[0]-1, self.img_shape[0] )

    # name and savename used for saving to file
    name = splitext(basename(name))[0] # gets name without directory or extension
    self.savename = 'out_imgs/' + name + '_out.png'

    #### The pipeline ###

    # 1. Undistort the img
    self.undist = cv2.undistort(self.img, self.mtx, self.dist, None, self.mtx)

    # 2. Warp the undistorted img to be a top down view of the lane line area
    self.warped = cv2.warpPerspective(self.undist, self.M, self.img_shape)

    # 3. Convert color space to 1 dimensional "gray" image. In this case gray
    #    actually means a weighted averaging of the L and S channels in the
    #    HLS color space
    hls = cv2.cvtColor(self.warped, cv2.COLOR_RGB2HLS)
    self.gray = hls[:,:,1]*(1-self.mixing_factor) + hls[:,:,2]*self.mixing_factor

    # 4. Calculate gradients and apply thresholds obtain a set of binary images
    self.sobelx = sobel(self.gray, 'x')
    self.sobely = sobel(self.gray, 'y')
    self.sobeldir = absgraddir(self.sobelx, self.sobely)

    self.sx_binary =   binary_thresh(self.sobelx,   self.sx_thresh)
    self.sy_binary =   binary_thresh(self.sobely,   self.sy_thresh)
    self.dir_binary =  binary_thresh(self.sobeldir, self.dir_thresh)
    self.gray_binary = binary_thresh(self.gray,     self.gray_thresh)

    # 5. Logical combination of binary image set to get a single binary image
        
    # Get AND of sx_binary, gray_binary, dir_binary
    combined_binary1 = combine_binary(self.sx_binary, self.gray_binary, '&')
    combined_binary1 = combine_binary(combined_binary1, self.dir_binary, '&')

    # Get AND of sy_binary, gray_binary, dir_binary
    combined_binary2 = combine_binary(self.sy_binary, self.gray_binary, '&')
    combined_binary2 = combine_binary(combined_binary2, self.dir_binary, '&')

    # Get OR of combined_binary 1 and 2
    self.binary_warped = combine_binary(combined_binary1, combined_binary2, '|')

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = self.binary_warped.nonzero()
    self.nonzeroy = np.array(nonzero[0])
    self.nonzerox = np.array(nonzero[1])

    # Create an output image to draw on and  visualize the result
    self.out_img = np.dstack((self.binary_warped, self.binary_warped, self.binary_warped))*255

    # 6. Use either Window method or margin method to find indices of lane lines
    if self.reset:
      print('window method')
      self.get_lane_inds_windows()
      self.add_lane_inds_on_out_img()
      self.reset = False
    else:
      self.get_lane_inds_margin()
      self.add_lane_inds_on_out_img()
      self.margin_plot_calcs()      

    # 7. Fit lane indices to 2nd degree polynomial
    self.fitpoly()

    # 8. Use fit polynomial to get X points corresponding to evenly spaced ploty
    self.get_rect_plot_points()

    # 9. Get curvature, distance from center, and fit error of current line for 
    #    use in fit detection 
    self.left_line.preprocess( self.left_fitx,  self.ploty, self.img_shape)
    self.right_line.preprocess(self.right_fitx, self.ploty, self.img_shape)

    # 10. Determine if the current lane is valid by comparing to expected lane 
    #     width, distance from center, and difference in fit from averaged best fit
    self.is_valid_lane()

    # Handle detection logic
    if self.detected:
      # reset undetected counter on detection
      self.undetected_counter = 0
    else:
      # increment undetected counter on invalid line
      self.undetected_counter = self.undetected_counter + 1

      # Check for reset flag
      if self.undetected_counter > self.undected_counter_thresh:
        print("Line reset")
        self.reset = True
        self.undetected_counter = 0

    # 11. Now that we have all of the data for the lane lines and have decided
    #     if they are valid we can go ahead and call the update on the line
    #     objects. If the lane was detection was valid this will calculate the 
    #     averaged best estimate lane line, curvature, and distance from center
    self.left_line.update( self.detected, self.reset, self.left_fitx,  self.ploty, self.img_shape)
    self.right_line.update(self.detected, self.reset, self.right_fitx, self.ploty, self.img_shape)

    # 12. Calculate curvature, distance from center of lane, and lane width from 
    #     using best fit lines
    self.combined_line_geometry()

    # 13. Create output image which composites the original camera image with 
    #     pixels used for lane detection higlighted and a polyfill for the 
    #     lane itself. Also add text for curvature and distance from center
    self.get_lane_img()
    self.add_text_to_lane_img()

    # Optional plots of the steps involved
    if self.show_plot or self.save_plot:
      self.plot_process()


  # Get summary geometry values for the lane overall based on geometry of the 
  # of the two line objects
  def combined_line_geometry(self):

    # Lane curvature is average of curvature of each line
    self.curvature = (self.left_line.radius_of_curvature + self.right_line.radius_of_curvature)/2

    # Distance of camera from center of lane
    self.distance_from_center = (self.left_line.distance - self.right_line.distance)/2

    # Width of lane
    self.width = self.left_line.distance + self.right_line.distance


  # Highlight the pixels identified as part of each lane line
  def add_lane_inds_on_out_img(self):
    self.out_img[self.nonzeroy[self.left_lane_inds],  self.nonzerox[self.left_lane_inds]] = [0, 0, 255]
    self.out_img[self.nonzeroy[self.right_lane_inds], self.nonzerox[self.right_lane_inds]] = [255, 0, 0]


  # Determine if current lane is reasonable
  def is_valid_lane(self):

    # Initialize to True
    self.detected = True
    
    # If difference in polynomial fit is too large reject the lane
    if self.left_line.diff + self.right_line.diff > self.diff_thresh:
      print("Line rejected on bad fit")
      self.detected = False

    # If width is too far from expected value, reject the lane
    self.curr_width = self.left_line.curr_distance + self.right_line.curr_distance
    if np.absolute(self.curr_width - self.width_expected) > self.width_thresh:
      print("Line rejected on bad width")
      self.detected = False

    # If distance from center is too large, reject the lane
    self.curr_distance_from_center = (self.left_line.curr_distance - self.right_line.curr_distance)/2
    if np.absolute(self.curr_distance_from_center) > self.distance_from_center_thresh:
      print("Line rejected on bad distance from center")
      self.detected = False


  # Do window method for finding indices of lane lines
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
        cv2.rectangle(self.out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(self.out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
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
 

  # Do margin method for finding indices of lane lines
  def get_lane_inds_margin(self):
    self.left_lane_inds =  ((self.nonzerox > (self.left_fit[0]*(self.nonzeroy**2)  + self.left_fit[1]*self.nonzeroy  + self.left_fit[2] - self.margin))  
                          & (self.nonzerox < (self.left_fit[0]*(self.nonzeroy**2)  + self.left_fit[1]*self.nonzeroy  + self.left_fit[2]  + self.margin))) 
    self.right_lane_inds = ((self.nonzerox > (self.right_fit[0]*(self.nonzeroy**2) + self.right_fit[1]*self.nonzeroy + self.right_fit[2] - self.margin)) 
                          & (self.nonzerox < (self.right_fit[0]*(self.nonzeroy**2) + self.right_fit[1]*self.nonzeroy + self.right_fit[2] + self.margin)))  


  # Fit a second order polynomial to each line
  def fitpoly(self):
    self.left_fit =  np.polyfit(self.nonzeroy[self.left_lane_inds] , self.nonzerox[self.left_lane_inds],  2)
    self.right_fit = np.polyfit(self.nonzeroy[self.right_lane_inds], self.nonzerox[self.right_lane_inds], 2)

  # Make plots of the data pipeline
  def plot_process(self):
    # Make a 2x3 subplot
    f,axs = plt.subplots(2,3, figsize=(28,16))

    # Plot undistorted image
    axs[0,0].imshow(self.undist)

    # Plot gray and warped image
    axs[0,1].imshow(self.gray, cmap='gray')

    # Plot binary image
    axs[0,2].imshow(self.binary_warped, cmap='gray')

    self.plot_lane_detection(axs[1,0])
    self.plot_2d_lane(axs[1,1])
    axs[1,2].imshow(self.lane_img)

    # Save based on flag
    if self.save_plot:
      plt.savefig(self.savename)
    if self.show_plot:
      plt.show()
    plt.close()

  # Get x points correspoinding to ploty based on fit polynomial
  def get_rect_plot_points(self):
    self.left_fitx =  self.left_fit[0]*self.ploty**2 +  self.left_fit[1]*self.ploty +  self.left_fit[2]
    self.right_fitx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]

  # Form plot img for plotting margin method
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
      cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255,0))
      cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255,0))
      self.out_img = cv2.addWeighted(self.out_img, 1, window_img, 0.0, 0)


  # Do plot showing lane detection
  def plot_lane_detection(self, ax):
    # plot out_img
    ax.imshow(self.out_img)

    # Add polyfit lane line
    self.plot_2d_lane_fits(ax, color='yellow')

    # Axis limits
    ax.set_xlim(0, self.img_shape[0])
    ax.set_ylim(self.img_shape[1], 0)


  # Plot warped image and lane fit polynomials
  def plot_2d_lane(self, ax):
    # Plot warped image
    ax.imshow(self.warped)

    # Add polyfit lane line
    self.plot_2d_lane_fits(ax, color='green')

    # Axis limits
    ax.set_xlim(0, self.img_shape[0])
    ax.set_ylim(self.img_shape[1], 0)

  # Plot polynomial fit lines
  def plot_2d_lane_fits(self, ax, color='yellow'):
    ax.plot(self.left_fitx,  self.ploty, color=color)
    ax.plot(self.right_fitx, self.ploty, color=color)

  # Calculations for creaing lane image
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
    cv2.fillPoly(color_warp, np.int_([pts]), [0,255,0])

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, self.Minv, self.img_shape) 

    # Also warp the out_img 
    newwarp2 = cv2.warpPerspective(self.out_img, self.Minv, self.img_shape) 

    # Combine the result with the original image
    self.lane_img = cv2.addWeighted(self.undist, 1.0, newwarp, 0.3, 0)

    # Also combine in the out_img
    self.lane_img = cv2.addWeighted(self.lane_img, 1, newwarp2, 1.0, 0)


  # Add text to image
  def add_text_to_lane_img(self):
    s = "Radius of Curvature = {:.0f}(m)".format(self.curvature)
    s2 = "Vehicle is {:.02f}m from center. Width: {:.01f}m".format(self.distance_from_center, self.width)
    s3 = "Video Frame: {:04}: Line Detected: {}  Reset: {}".format(self.frame_count, self.detected, self.reset)
    cv2.putText(self.lane_img, s,  (50, 80),  cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    cv2.putText(self.lane_img, s2, (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    cv2.putText(self.lane_img, s3, (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, 255) 

# End of Lane Class