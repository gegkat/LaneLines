import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
from os.path import basename, splitext
from moviepy.editor import VideoFileClip, ImageSequenceClip
import Lane
from lane_utils import *

def run_lane_lines(show_plot=True, save_plot=True, fname='project_video.mp4', MAX_FRAMES=10000, n_mod=1, fps=16):
  do_cal = False
  check_cal = False

  nx = 9
  ny = 6

  if do_cal:
    run_calibration('camera_cal/calibration*.jpg', nx, ny)

  mtx, dist, M, Minv = load_calibration()

  if check_cal:
    check_calibration('camera_cal/calibration1.jpg', mtx, dist)

  #filenames = glob.glob('test_images/*.jpg')
  filenames = ['test_images/test1.jpg']

  ll = Lane.Lane(M, Minv, mtx, dist, show_plot=show_plot, save_plot=save_plot)
  # for i in range(len(filenames)):
  #  img = read_img(filenames[i])
  #  img = img2RGB(img)
  #  ll.process_img(img, filenames[i])

  # return ll

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




