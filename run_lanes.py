import lanes

do_cal = False
check_cal = False
do_plot_binary = True


nx = 9
ny = 6

if do_cal:
  lanes.run_calibration('camera_cal/calibration*.jpg', nx, ny)

mtx, dist = lanes.load_calibration()

if check_cal:
  lanes.check_calibration('test_images/test1.jpg', nx, ny, mtx, dist)

undist = lanes.undistort_file('test_images/test1.jpg', mtx, dist)
sx_binary, sy_binary, dir_binary, s_binary = lanes.thresh(undist)
combined_binary, color_binary = lanes.combine_binary(sx_binary, s_binary)

if do_plot_binary:
  lanes.plot_binary(combined_binary, color_binary)

