**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted_output.png "Undistorted Calibration"
[image2]: ./output_images/undistorted_test_image.png "Undistorted Test Image"
[image3]: ./output_images/warped_and_gray_scale_test_image.png "Warp Example"
[image4]: ./output_images/binary_threshold_test_image.png "Binary Example"
[image5]: ./output_images/window_method_lane_line_id_test_image.png "Fit Window Visual"
[image6]: ./output_images/margin_method_test_image.png "Fit Margin Visual"
[image7]: ./output_images/lane_estimate_test_image.png "Output"
[video1]: ./output_images/project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the function get_calibration(), lines 22-54 of lane_utils.py.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
The code for my pipeline is executed by the funciton process_img in the file Lane.py. The distortion correction is the first step of the pipeline and is performed in line 123 of Lane.py. This step uses the mtx and dist matrices obtained from the calibration pre-processing step performed on checkerboard images and saved with a pickle file. 

This is an example of an undistorted test image:

![alt text][image2]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
In my pipeline I choose to do the binary thresholding after the perspective transform because this converts the lane lines to be more vertical in the resulting image. I used a combination of color and gradient thresholds to generate a binary image as steps 3 and 4 in my pipeline which can be found in lines 131-141 of Lane.py. These steps make use of the functions sobel, absgraddir and binary_thresh found in lines 125-158 of lane_utils.py. To get a single dimension of color for my thresholding that didn't miss any lane lines I used an averaging of the L and S channels in the HLS color space. 

Here's an example of my output for this step.

![alt text][image4]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective transform is performed as step 2 of my pipeline on line 126 of Lane.py. The matrix for the transform is calculated in a preprocessing step at the same time as camera calibration and saved in a pickle file. This is performed by the function get_warp_matrices() on lines 75 to 105 in lane_utils.py.

I chose the hardcode the source and destination points in the following manner:

```
  # Source points around edges of lane lines for warp matrices
  src = np.float32([(595, 450), (680, 450), (1080, 720), (230, 720), ])

  # Offset for dst points. Choosen to get resulting image that
  # includes only relevant portion of road
  offsetx = 300 
  offsety = 0

  dst = np.float32([[offsetx,               offsety ] , 
                    [img_size[0] - offsetx, offsety] , 
                    [img_size[0] - offsetx, img_size[1] - offsety] , 
                    [offsetx,               img_size[1] - offsety] ])
```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 595,  450     | 300, 0        | 
| 680,  450     | 980, 0        |
| 1080, 720     | 980, 720      |
| 230,  720     | 300, 720      |

I verified that my perspective transform was working as expected by testing on an image with straight parallel lane lines and verifying that the lines appear parallel in the warped image. Here is an example of a warped test image with parallel vertical lines. This image as also been converted to gray scale. 

![alt text][image3]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Identifying lane-line pixels is step 6 in my pipeline. This is done with either the function get_lane_inds_window() or the function get_line_inds_margin(). The margin method is used if in the video stream valid lane lines were identified recently. In this method indices within 100 pixels of the previous lane line are considered to be lane-line pixels. This is executed in lines 325 to 328 of Lanes.py. 



On the first video image or after the algorithm has been reset due to consecutive invalid lane lines, then a window method is used to find the lane-line pixels. This is executed in lines 272 to 320 of Lanes.py. In this method the first step is to estimate the starting point for where the lane lines intersect the bottom of the frame. This is done by making a histogram of the number of active pixels in the bottom half of the binary threshold image as a function of X position in the image. Then the bin with the most active pixels from the left and right halves of the image are selected as the base of the right and left lane lines. Next, we will look for active pixels in a window around this base location. The window is a rectangle of width 200 pixels and height 1/9th of the image height. Any active pixels in this window are identified as lane-line pixels. Then the window is moved up to the next 1/9th of the image vertically and recentered horizontally to the centroid of the previous window in X. This is repeated 9 times until the top of the image is reached. 

Fitting the lane-line pixels to a 2nd order polynomial is step 7 of the pipeline. This is achieved by feeding the positions of the lane-line pixels for each line to the numpy polyfit function. The code is found on lines 332 to 334 of Lane.py.

Here is visual demonstration of the lane indices detection and polynomial fit using the margin method. In this plot the green area is the search area within the margin of the previous lane line detection. Blue and Red pixels are left and right lane line detections. White pixels met the thresholding criteria but did not fall within the margin. The fitted polynomials are shown in yellow. 

![alt text][image6]

Here is visual demonstration of the lane indices detection and polynomial fit using the window method. In this plot the green rectangles show the windows of the search area. Blue and Red pixels are left and right lane line detections. White pixels met the thresholding criteria but did not fall within the margin. The fitted polynomials are shown in yellow. 

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The curvature of each lane line is calculated in step 11 of the pipeline. The code for this is in the function get_geometry() found at lines 95 to 118 of Line.py. The method is to first scale the pixels of the lane lines to convert to meters, then make a new 2nd degree polynomial fit with the scaled lane line indices. Estimating curvature at the current driving position is done by evaluating the curvature of this polynomial at the point where the polynomial intersects the base of the image. The equation for the curvature of the polynomial of the form 
f(y) = A*y^2 + B*y + C 
at a point y is 
curvature = [(1 + (2*A*y + B)^2)^1.5] / abs(2*C). 
To make the calculation even simpler, I flipped the y values of the image before fitting the polynomial so that y value at the base of the image is 0. This simplifies the curvature equation to 
curvature = (1 + B^2)^1.5/ abs(2*C)

The radius of curvature of the entire lane is calculated by averaging the curvature of each lane line in step 12 of my pipeline. The code is line 232 of Lane.py. 

The lane position of the vehicle with respect to center is done by first estimating the distance of each lane line from the center of the image. With this information the position of the vehicle in the lane is calculated as the difference between these two distances divied by 2. 

The distance of each lane line from the center of the image is calculated in step 11 of the pipeline and on lines 95 to 118 of Line.py. The same polynomial from the curvature calculation is used to calculate the difference between the camera center position converted to meters and the fit polynomial evaluated at the base of the image (y=0 in this case, so f(0) = C). 

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this as step 13 of my pipeline as the function get_lane_img() in Lane.py, lines 417 to 442. When I plot my result I include a shaded green area for the lane estimate, as well as a blue highlight of pixels identified for the left lane line, red pixels for the right lane line, and white for pixels that passed the binary threshold but were not matched to a lane line. Additionally, if the window method was used then green rectanges of the windows is also included. 

Here is an example of my result on a test image:

![alt text][image7]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/project_video_output.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In my process, I spent quite a bit of time on fine tuning the specific threshold values for converting the image to a binary image of candidate lane lines. I found that my results were sensitive to these thresholds. Although it was not too hard to find values that worked well for the project video, the same thresholds did not work as well for the challenge videos. One way that I could improve my pipeline would be to manually label lane lines and then use a learning algorithm to search the parameter space for thresholds that gave the best accuracy for detecting pixels that are part of lane lines. 

The biggest problem with identifying lane lines was in finding false positives. By using a combination of gradient and the L and S components of the HLS color space, my algorithm is still susceptible to falsely identifying features like shadows or certain discolorations in the road pavement as lane lines. This could be improved by spending more time fine-tuning the binary thresholds on gradient and color space. In particular since lanes lines are always white or yellow I could have possibly used the RGB color space as an additionaly threshold. 

Another way to perhaps improve the robustness would be to use more information about known characteristics of lane lines to improve the step that determines lane lines from the binary threshold image. Instead of just taking the max bin of a histogram and fitting each lane line separately to a polynomial, I could have done a simultaneous fitting of two parallel polynomials with a fixed width between them and choose the best fit for the whole lane instead of each lane line. 