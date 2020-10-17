import cv2;
import numpy as np;
                     # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
import math
print("Environment Ready")

# Setup:
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device_from_file("correct_hole.bag")
profile = pipe.start(cfg)
# Create colorizer object
colorizer = rs.colorizer()

# Skip 5 first frames to give the Auto-Exposure time to adjust
for x in range(5):
  pipe.wait_for_frames()
  
# Store next frameset for later processing:
frameset = pipe.wait_for_frames()
# colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
# create align object
align = rs.align(rs.stream.color)
# alig frameset of depth
aligned_frameset = align.process(frameset)

# align depth frames,colorized depth map 
aligned_depth_frame = aligned_frameset.get_depth_frame()
color_frame = frameset.get_color_frame()
depth_frame = frameset.get_depth_frame()


intrin = depth_frame.profile.as_video_stream_profile().intrinsics
print(intrin)
fx=intrin.fx
fy=intrin.fy
fl=math.sqrt((fx*fx)+(fy*fy))
print(fl)

# Cleanup:
pipe.stop()
print("Frames Captured")
depth=np.asanyarray(depth_frame.get_data())
aligned_depth=np.asanyarray(aligned_depth_frame.get_data())
color = np.asanyarray(color_frame.get_data())
aligned_depth_colorized=np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
aligned_depth = (aligned_depth/256).astype('uint8')
plt.imshow(aligned_depth_colorized)
plt.show()
hole_filling = rs.hole_filling_filter()
filled_depth = hole_filling.process(aligned_depth_frame)
colorized_depth = np.asanyarray(colorizer.colorize(filled_depth).get_data())
plt.imshow(colorized_depth)
plt.show()

'''
# Threshold.
# Set values equal to or above 220 to 0.
# Set values below 220 to 255.

th, im_th = cv2.threshold(aligned_depth_colorized, 290, 300, cv2.THRESH_BINARY_INV)
plt.imshow(im_th)
plt.show()

# Copy the thresholded image.
im_floodfill = im_th.copy()

# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = im_th.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)

# Floodfill from point (0, 0)
#cv2.floodFill(im_floodfill, mask, (0,0))
cv2.floodFill(im_floodfill, mask, (0,0), newVal=(0, 0, 255), loDiff=(5, 5, 5, 5), upDiff=(5, 5, 5, 5))

# Invert floodfilled image
im_floodfill_inv = cv2.bitwise_not(im_floodfill)

# Combine the two images to get the foreground.
im_out = im_th | im_floodfill_inv

# Display images.

cv2.imshow("Thresholded Image", im_th)
cv2.imshow("Floodfilled Image", im_floodfill)
cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
cv2.imshow("Foreground", im_out)
cv2.waitKey(0)

plt.imshow(im_out)
plt.show()
'''