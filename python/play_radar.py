################################################################################
#
# Copyright (c) 2017 University of Oxford
# Authors:
#  Dan Barnes (dbarnes@robots.ox.ac.uk)
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
################################################################################

import argparse
import os
from radar import load_radar, radar_polar_to_cartesian
import numpy as np
import cv2

parser = argparse.ArgumentParser(description='Play back radar data from a given directory')

parser.add_argument('dir', type=str, help='Directory containing radar data.')

args = parser.parse_args()

timestamps_path = os.path.join(os.path.join(args.dir, os.pardir, 'radar.timestamps'))
if not os.path.isfile(timestamps_path):
    raise IOError("Could not find timestamps file")

# Cartesian Visualsation Setup
# Resolution of the cartesian form of the radar scan in metres per pixel
cart_resolution = .25
# Cartesian visualisation size (used for both height and width)
cart_pixel_width = 501  # pixels
interpolate_crossover = True

title = "Radar Visualisation Example"
autoplay = False

radar_timestamps = np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)
for i, radar_timestamp in enumerate(radar_timestamps):
    filename = os.path.join(args.dir, str(radar_timestamp) + '.png')

    if not os.path.isfile(filename):
        raise FileNotFoundError("Could not find radar example: {}".format(filename))

    timestamps, azimuths, valid, fft_data, radar_resolution = load_radar(filename)
    cart_img = radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, cart_resolution, cart_pixel_width,
                                        interpolate_crossover)

    # Combine polar and cartesian for visualisation
    # The raw polar data is resized to the height of the cartesian representation
    downsample_rate = 4
    fft_data_vis = fft_data[:, ::downsample_rate]
    resize_factor = float(cart_img.shape[0]) / float(fft_data_vis.shape[0])
    fft_data_vis = cv2.resize(fft_data_vis, (0, 0), None, resize_factor, resize_factor)
    vis = cv2.hconcat((fft_data_vis, fft_data_vis[:, :10] * 0 + 1, cart_img))

    vis = cv2.putText(vis, "%d: %d" % (i, radar_timestamp), org=(10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                      fontScale=0.5, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    vis = cv2.putText(vis, "Q to [Q]uit, P to toggle auto[p]lay, Any key to step forward", org=(
        10, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    cv2.imshow(title, vis * 2.)  # The data is doubled to improve visualisation
    
    if autoplay:
        _k = cv2.waitKey(1) & 0xFF        
    else: 
        _k = cv2.waitKey(0) & 0xFF

    if _k == ord('q'):
        break
    elif _k == ord('p'):
        autoplay ^= True
