## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

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

[image1]: ./result/test_undist.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  



### Camera Calibration
- Distortion changes the shape and size of the objects, we need to calibrate for the changes.
- Take pictures of known shape and use them to detect and correct the errors.
- Will be using a chess board to calibrate in this case, the high contrast pattern makes automatic detection easier.
- Find the Chess board corners of the distorted images using `cv2.findChessboardCorners`.
- Calculate the actual corners of the undistorted chess board image, we can find this since the size if each chess board
  square is 1 x 1 and the total dimension of the board is 9 x 6.
- Using these points calibrate the camera to obtain the camera matrix and distortion co-efficients.
- Use this information to Undistort the images taken from the camera using the  `cv2.undistort` function.
- `result/chess_board_corners` contains the images with corners of the chess board drawn.
- [Here is the code](https://gist.github.com/hackintoshrao/5b40dd4a1ba814c7fb26569f50510e23) for calibrating the camera and       undistorting the image.
- The folder `result/chess_undistort` contains all the undistorted chess board images.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

- The folder `./result/test_undist/` contains undistorted test images.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
- [Here is the code snippet](https://gist.github.com/hackintoshrao/db8e5438b3f41850f4a5a4131ac60acb) I've used to create binary thresholded image .
- The challenge here was to get the right combination of thresholding values.
- Have used gradx, grady, gradient magnitude, direction and saturation thresholding, the threshold values can be seen in the code snippet above.
- [Here is the commit](https://github.com/hackintoshrao/advance-lane-finding/commit/88ebb21db5f377f82165453d782c3e83224f4035) corresponding to the pipeline addition.
- The folder `result/binary_images` contains the binary images of the test images.



#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Chose hardcoded source and destination points to perform perspective transform of the image.

```python
top_left_source = [560, 470]
top_right_source = [730, 470]
bottom_right_source = [1080, 720]
bottom_left_source = [200, 720]

source = np.array([bottom_left_source,bottom_right_source,top_right_source,top_left_source])
source_points = np.float32(pts.tolist())


top_left_dst = [200,0]
top_right_dst = [1100,0]
bottom_right_dst = [1100,720]
bottom_left_dst = [200,720]


destination = np.array([bottom_left_dst,bottom_right_dst,top_right_dst,top_left_dst])
destination = np.float32(destination.tolist())
```


- I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

- The warped images are saved in `result/perspective_transform`.
- [Here is the commit](https://github.com/hackintoshrao/advance-lane-finding/commit/1081fe2690232a01c8e127846f1faec8c74c9107) containing the code for the corresponding changes. The code can be pulled and can be run to obtain warped images for all the test images.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

- Identified the starting x coordinates for lane line scanning by using histogram peaks for both side of the lanes.
- Using sliding window for continuous y coordinates, the peak points around a margin are found.
- `np.polyfit` is used to fit a polynomial to the identified points.
- x coordinate points are identified using coefficients of the polynomial found.
- The folder `./result/polyfit` contains images of the polynomial fit.
- [Here is the link](https://gist.github.com/hackintoshrao/fd17a36ef7415fa942db2ac5262c81d4) for the code snippet used to achieve the polynomial fit.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

- Conversion from pixels to meters is done using the following scale
```
  ym_per_pix = 30/720 # meters per pixel in y dimension
  xm_per_pix = 3.7/700
```
- New polynomial is fit using measurements in meters.

```
# Fit new polynomials to x,y in world space
  A = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
  right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
```

- Once the coefficients (A, B) of the polynomial fit is obtained the radius of curvature is calculated using the following formula

  ```
  left_curverad = ((1 + (2 * A *y_eval * ym_per_pix + B)**2)**1.5) / np.absolute(2 * A)
  right_curverad = ((1 + (2 * A * y_eval * ym_per_pix + B)**2)**1.5) / np.absolute( 2 * A)
  ```
- The distance from center is the difference between center of width of the warped image and the actual center of the width of the image from the camera.

- Here is the code snippet used to find the position of the car w.r.t center of the road.

  ```
  center_of_warped = (right_max + left_min)/2
  actual_center = (1280/2)
  pixel_from_center =  center_of_warped - actual_center
  print('pix dist from center', pixel_from_center)

  distance_from_center = xm_per_pix * pixel_from_center
  print('meter dist from center', distance_from_center)
  ```  

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
- Use `cv2.fillPoly` to fill the detected lane with green. Use left and right x-coordinates obtained from the polynomial fit to draw the lane line.
- Then use inverse perspective transform for draw the lane back onto the original image.
- Here is the code snippet I've used to plot back the detected lane onto the original image.

```
# Create an image to draw the lines on
  warp_zero = np.zeros_like(warped).astype(np.uint8)
  color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

  # Recast the x and y points into usable format for cv2.fillPoly()
  pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
  pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
  pts = np.hstack((pts_left, pts_right))

  # Draw the lane onto the warped blank image
  cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

  # Warp the blank back to original image space using inverse perspective matrix (Minv)
  newwarp = cv2.warpPerspective(color_warp, test_M_inverse, (undist_img.shape[1], undist_img.shape[0]))
  # Combine the result with the original image

  orig_img = np.copy(undist_img)
  result = cv2.addWeighted(orig_img, 1, newwarp, 0.3, 0)

```

- The final results are saved in `./result/final_result/` folder.


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/Qw1LTmVvWyQ)

The video is also saved in `./result/video/result.mp4`

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

- The first challenge was to get the right combination of threshold values for obtaining the binary image. Had to experiment a lot
  around this and this consumed lot of time.
- The next issue was related to choosing parameters for doing sanity check on the polynomial fit, it took a while to choose the
  right parameter to perform the sanity check.
- The last challenge I was faced to get the moving average of the polynomial fits, choosing deque for data structure made the job  easy.
- We have made assumption that the road is flat, but under conditions where the road is not flat the pipeline would fail. To overcome this one needs to account for the change in flatness of the road before performing perspective transform.
- Since we have adjusted the threshold parameters for the conditions in the test image, the pipeline would fail under drastic lighting conditions. Probably automatic adjusting of the threshold and channel paramters, making it more dynamic could help detect lines in varying drastic environments.
