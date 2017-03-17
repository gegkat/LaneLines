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


def do_cal():
  run_calibration('camera_cal/calibration*.jpg', nx=9, ny=6)


def check_cal():
  mtx, dist, M, Minv = load_calibration()
  check_calibration('camera_cal/calibration1.jpg', mtx, dist)


def test_lane_lines(show_plot=True, save_plot=True):
  mtx, dist, M, Minv = load_calibration()

  filenames = glob.glob('test_images/*.jpg')
  #filenames = ['test_images/test1.jpg']

  lane = Lane.Lane(M, Minv, mtx, dist, show_plot=show_plot, save_plot=save_plot)

  for i in range(len(filenames)):
    img = read_img(filenames[i])
    img = img2RGB(img)
    lane.process_img(img, filenames[i])

  return lane


def run_lane_lines(show_plot=True, save_plot=True, fname='project_video.mp4', MAX_FRAMES=10000, n_mod=1, fps=16):
  mtx, dist, M, Minv = load_calibration()

  lane = Lane.Lane(M, Minv, mtx, dist, show_plot=show_plot, save_plot=save_plot)

  clip = VideoFileClip(fname)
  n_frames = int(clip.fps * clip.duration)
  n_frames = min(MAX_FRAMES, n_frames) 
  count = 0
  images_list = []
  for frame in clip.iter_frames():
    count = count+1
    if count % n_mod == 0:
      print("{} of {}".format(count, n_frames))
      lane.process_img(frame, str(count))
      images_list.append(lane.lane_img)
    if count >= MAX_FRAMES:
      break

  clip = ImageSequenceClip(images_list, fps=fps)

  savename = splitext(basename(fname))[0]
  savename = 'out_imgs/' + savename + '_out.mp4'
  clip.write_videofile(savename) 

  return lane, clip




