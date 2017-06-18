import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from collections import deque
from moviepy.editor import VideoFileClip

ksize = 7
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')


# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    chess_img = cv2.imread(fname)
    gray = cv2.cvtColor(chess_img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(chess_img, (9,6), corners, ret)
        write_name = 'board'+str(idx)+'.jpg'
        cv2.imwrite('result/chess_board_corners/' + write_name, chess_img)

def undistort_image(img):
    img_size = (img.shape[1], img.shape[0])
    # Do camera calibration given object points and image points.
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    # undistort the image using camera matrix and distortion co-efficients.
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

# Saturation color channel binary threshold.
def s_threshold(img, s_thresh=(0, 255)):
    s_channel = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:,:,2]
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    return s_binary

# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # But the binary image just pitch black with no edges in it.
    # Return the result
    return binary_output

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def pipeline(img):
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(10, 255))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(60, 255))
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(40, 255))
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(.65, 1.05))
    saturation_binary = s_threshold(img, s_thresh=(160, 255))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (saturation_binary == 1)] = 1

    return combined


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)
    ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# images = glob.glob('./test_images/test*.jpg')

top_left = [560, 470]
top_right = [730, 470]
bottom_right = [1080, 720]
bottom_left = [200, 720]

pts = np.array([bottom_left,bottom_right,top_right,top_left])
pts = np.float32(pts.tolist())


top_left_dst = [200,0]
top_right_dst = [1100,0]
bottom_right_dst = [1100,720]
bottom_left_dst = [200,720]

test_src = pts
test_dst = np.array([bottom_left_dst,bottom_right_dst,top_right_dst,top_left_dst])
test_dst = np.float32(test_dst.tolist())



class Line():
    def __init__(self):
        # x values of the last n fits of the line
        self.left_x_polyfits  = deque(maxlen=5)

        self.right_x_polyfits  = deque(maxlen=5)
        # average of the polyfits from the deque.
        self.mean_left_polyfits = 0
        self.mean_right_polyfits = 0

        # maximum allowed gap between x coordinates of
        # 2 consecutive frames to pass sanity check.
        self.max_next_frame_gap = 100
        # last values of gap between min and max values of each lane line.
        self.last_left_gap = 0
        self.last_right_gap = 0

    def add_mean_fit_left(self, left_fitx):
        """
        Adds the left lane polynomial fits into the dequeue,
        calculates the mean of last 5 fits which'll be used for
        smoother trasistion across frames.
        """
        self.left_x_polyfits.append(left_fitx)
        self.mean_left_polyfits = np.mean(self.left_x_polyfits, axis=0)

    def add_mean_fit_right(self, right_fitx):
        """
        Adds the right lane polynomial fits into the dequeue,
        calculates the mean of last 5 fits which'll be used for
        smoother trasistion across frames.
        """
        self.right_x_polyfits.append(right_fitx)

        self.mean_right_polyfits = np.mean(self.right_x_polyfits, axis=0)

    def get_mean_x_fits(self):
        """
        return left and right polyfit values.
        """
        return self.mean_left_polyfits, self.mean_right_polyfits

    def sanity_check_return_mean_fit(self, left_fitx, right_fitx):
        """
        Sanity check for detected polyfit is based on the difference between the detected lane gap and
        the previously detected lane gap.
        If the sanity check passes the the coefficients of the polynomial fit then the values are added to deque,
        and the mean of the coefficients calcualted.
        If the sanity check fails the
        """
        min_leftfitx = np.min(left_fitx)
        max_leftfitx = np.max(left_fitx)
        min_rightfitx = np.min(right_fitx)
        max_rightfitx = np.max(right_fitx)

        gap_left = max_leftfitx - min_leftfitx
        gap_right = max_rightfitx - min_rightfitx

        left_sanity = False
        right_sanity = False

        if self.last_left_gap == 0 and self.last_right_gap == 0:
            self.last_left_gap = gap_left
            self.last_right_gap = gap_right
            self.add_mean_fit_left(left_fitx)
            self.add_mean_fit_right(right_fitx)

            left_sanity = True
            right_sanity = True
            mean_left_fits, mean_right_fits = self.get_mean_x_fits()
            return left_sanity,right_sanity, mean_left_fits, mean_right_fits
        else:
            if np.abs( gap_left - self.last_left_gap) <= self.max_next_frame_gap :
                self.last_left_gap = gap_left
                self.add_mean_fit_left(left_fitx)
                left_sanity = True

            if np.abs( gap_right - self.last_right_gap) <= self.max_next_frame_gap :
                self.last_right_gap = gap_right
                self.add_mean_fit_right(right_fitx)
                right_sanity = True

            mean_left_fits, mean_right_fits = self.get_mean_x_fits()
            return left_sanity,right_sanity, mean_left_fits, mean_right_fits


Left_line = Line()
Right_line = Line()

def process_image(image):
    # undistorted original image.
    undist_img = undistort_image(image)
    combined_binary = pipeline(image)
    # process the test images through the pipeline and save the binary images in /result/binary_images/.
    imshape = combined_binary.shape

    vertices = np.array([[(0 + 150, imshape[0]), (575, 430), (725, 430), (imshape[1]-100,imshape[0])]], dtype=np.int32)

    #masked_image = region_of_interest(combined_binary, vertices)
    masked_image = region_of_interest(combined_binary, vertices)

    #Compute the perspective transform, M, given source and destination points:
    #M = cv2.getPerspectiveTransform(src, dst)
    test_M = cv2.getPerspectiveTransform(test_src, test_dst)

    # Compute the inverse perspective transform.
    test_M_inverse = cv2.getPerspectiveTransform(test_dst, test_src)

    #Warp an image using the perspective transform, M.
    warped = cv2.warpPerspective(masked_image, test_M, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)
    binary_warped = np.copy(warped)
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)


    # Use histogram on both side of the lane to obtain starting
    # point for lane line scanning.
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices.
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each.
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # generate continuous y coordinates from (0, image-height).
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

    # find x coordinates using the polyfit coefficients.
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # If sanity check passes  then the current values are added to deque and mean is calculated.
    left_lane_sanity, right_lane_sanity, mean_left_fitx, mean_right_fitx = Left_line.sanity_check_return_mean_fit(left_fitx, right_fitx)

    left_fitx = mean_left_fitx
    right_fitx = mean_right_fitx
    # Generate x and y values for plotting
    y_eval = np.max(ploty)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])


    # distance of car from center.
    # The distance from center is the difference between center of width of the warped
    # image and the actual center of the width of the image from the camera.
    left_min = np.amin(leftx, axis=0)

    right_max = np.amax(rightx, axis=0)


    center_of_warped = (right_max + left_min)/2
    actual_center = (1280/2)
    pixel_from_center =  center_of_warped - actual_center


    distance_from_center = xm_per_pix * pixel_from_center


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

    font = cv2.FONT_HERSHEY_SIMPLEX
    # by default its assumed that car is to the left of center of lane.
    side = "left"
    # side set to right if distance from center is negative.
    if distance_from_center < 0:
        side = "right"

    cv2.putText(result, 'Vehicle is %.2fm %s of center' % (np.abs(distance_from_center), side), (50, 140), font, 1,(255, 255, 255), 2)

    cv2.putText(result, 'Radius of left Curvature = %d(m)' % left_curverad, (50, 50), font, 1, (255, 255, 255), 2)
    return result

video_output = './result/result.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(video_output, audio=False)
