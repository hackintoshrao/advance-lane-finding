import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

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

def undistort_image(image):
    img = cv2.imread(image)
    img_size = (img.shape[1], img.shape[0])
    # Do camera calibration given object points and image points.
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    # undistort the image using camera matrix and distortion co-efficients.
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

# save undistorted chess board images.
for idx, fname in enumerate(images):

# Undistort a chess board image.
    undist_img = undistort_image(fname)
# save the undistorted image.
    write_name = 'board_undistort'+str(idx)+'.jpg'
    cv2.imwrite('./result/chess_undistort/' + write_name, undist_img)


ksize = 7 # Choose a larger odd number to smooth gradient measurements

# Saturation color channel binary threshold.
def s_threshold(img, s_thresh=(0, 255)):
    s_channel = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:,:,2]
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    return s_binary


# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
    plt.imshow(combined, cmap='gray')
    plt.show()
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

images = glob.glob('./test_images/test*.jpg')


# undistort the test images and save them in /result/test_undist/.
for idx, fname in enumerate(images):

    undist_img = undistort_image(fname)
    write_name = "test_" + str(idx+1) + "_undistort.jpg"
# save the undistorted image.
    cv2.imwrite('./result/test_undist/' + write_name,undist_img)

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

for idx, fname in enumerate(images):
    test_image = cv2.imread(fname)
    combined_binary = pipeline(test_image)
    # process the test images through the pipeline and save the binary images in /result/binary_images/.
    imshape = combined_binary.shape

    vertices = np.array([[(0 + 150,imshape[0]),(575, 430), (725, 430), (imshape[1]-100,imshape[0])]], dtype=np.int32)


    write_name = "test_" + str(idx+1) + "_binary.jpg"
    plt.imsave('./result/binary_images/' + write_name, combined_binary, cmap='gray')
    #masked_image = region_of_interest(combined_binary, vertices)
    masked_image = region_of_interest(combined_binary, vertices)

    #Compute the perspective transform, M, given source and destination points:
    #M = cv2.getPerspectiveTransform(src, dst)
    test_M = cv2.getPerspectiveTransform(test_src, test_dst)

    #Warp an image using the perspective transform, M:
    test_warped = cv2.warpPerspective(masked_image, test_M, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)
    #warped = cv2.warpPerspective(combined_binary, M, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)
    write_name = "test_" + str(idx+1) + "_warped.jpg"
    plt.imsave('./result/perspective_transform/'+write_name,test_warped, cmap='gray')