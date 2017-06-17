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

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
