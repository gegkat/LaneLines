import numpy as np

# Line class is used to contain and track the fit data and geometry
# of a single lane line. This Class also takes care of filtering
# the last N good line detections
class Line():
  def __init__(self):
    # x values of the last n fits of the line
    self.recent_xfitted = [] 

    # average x values of the fitted line over the last n iterations
    self.bestx = None     

    # polynomial coefficients averaged over the last n iterations
    self.best_fit = None  

    # polynomial coefficients for the most recent fit
    self.current_fit = None

    # radius of curvature of the line in some units
    self.radius_of_curvature = None 

    # distance in meters of vehicle center from the line
    self.line_base_pos = None 

    # difference in fit coefficients between best fit and new fit
    self.diff = None

    # Define conversions in x and y from pixels space to meters
    self.ym_per_pix = 30/720 # meters per pixel in y dimension
    self.xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Number of lines to keep for filter
    self.n_lines_filter = 5

  def preprocess(self, fitx, ploty, img_shape):
  # Preprocess is done on every frame before it has been determined
  # if the current line detection is valid

    # Calculate the the best fit 2nd degree polynomial
    self.current_fit = np.polyfit(fitx, ploty, 2)  

    # Calculate the curvature and distance for the current line
    # These are used to assess the validity of the lane line, the 
    # best estimate will come later with an averaging step after
    # validity has been determined
    self.curr_curvature, self.curr_distance = self.get_geometry(fitx, ploty, img_shape) 

    # Get the % difference of the fit polynomial coefficients between
    # the current line and the averaged best fit line. If the best fit
    # line coefficients are None then we have nothing to compare to and 
    # just set the difference to 0
    if self.best_fit is None:
      self.diff = 0
    else:
      # Diff is the sum of the absolute value of the difference between
      # the current fit and the averaged best fit, normalized by the best fit
      self.diff = sum(abs((self.current_fit - self.best_fit)/self.best_fit))

  def update(self, detected, reset, fitx, ploty, img_shape):
    # If the line was determined to be valid then detected will be
    # True and the recent_xfitted, bestx, best_fit, radius_of_curvature
    # and distance from center can all be updated
    # 
    # If the reset flag is high then clear out the best_fit and recent_xfitted
    # So that we can start fresh and search for a lane line from scratch

    if detected:
      # Reset the undetected_counter. When this gets high enough it triggers
      # a reset, but since we have a valid line this should be reset to 0
      self.undetected_counter = 0

      # Append fitx to the recent_xfitted 
      self.recent_xfitted.append(fitx)

      # Only want to keep a maximum of n_lines_filter in the recent_xfitted
      # list. Pop off the first element if we have exceeded
      if len(self.recent_xfitted) > self.n_lines_filter:
        self.recent_xfitted.pop(0)

      # Best x is the average of the values in recent_xfitted 
      self.bestx = sum(self.recent_xfitted)/len(self.recent_xfitted) 

      # Fit a line with the averarged x values    
      self.best_fit = np.polyfit(self.bestx, ploty, 2)  

      # Calculate curvature and distance from lane center with bestx
      self.radius_of_curvature, self.distance = self.get_geometry(self.bestx, ploty, img_shape) 
    
    if reset:
        # Too many invalid lines in a row, time to start fresh
        self.best_fit = None
        self.recent_xfitted = []

  def get_geometry(self, x, y, img_shape):
    # get_geometry returns curvature and distance from center of the image for 
    # a given set of x and y points in an image of shape img_shape

    # Flip the y values in the image so that y=0 is at the bottom of the image
    # This makes calculating the distance from camera center easier
    y = img_shape[1] - y
    y_eval = 0

    # Convert x and y from pixels to world coordinates
    y_world_coordinates = y*self.ym_per_pix
    x_world_coordinates = x*self.xm_per_pix

    # Fit 2nd degree polynomial f(y) = Ay^2 + By + C
    fit_cr =  np.polyfit(y_world_coordinates, x_world_coordinates, 2)

    # Calculate radius of curvature
    # curvature = [(1 + (2*A*y + B)^2)^1.5] / abs(2*C)
    # Since y = 0, simplifies to:
    # curvature = (1 + B^2)^1.5/ abs(2*C)
    curvature = ((1 + fit_cr[1]**2)**1.5)  / np.absolute(2*fit_cr[0])
    distance = np.absolute(fit_cr[2] - self.xm_per_pix*img_shape[0]/2)

    return curvature, distance