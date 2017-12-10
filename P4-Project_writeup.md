## P4 Advanced Lane Lines

### Udacity Self-Driving Car Nanodegree

#### David Peabody

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

[image1]: ./camera_cal/calibration4.jpg "original"
[image1.5]: ./camera_cal/undist_calibration4.jpg "Undistorted"
[image2]: ./test_images/test5.jpg "Road original"
[image2.5]: ./camera_cal_out/undist_test5.jpg "Road Undistorted"
[image3]: ./output_images/initial "initial"
[image3.1]: ./output_images/hls_l.jpg "hls_l"
[image3.2]: ./output_images/hls_s "hls_s"
[image3.3]: ./output_images/abs_sobel "absolute Sobel"
[image3.4]: ./output_images/combined "combined"
[image3.5]: ./output_images/combined_bine "combined binary"
[image4]: ./output_images/warp "Warp Example"
[image5]: ./output_images/find_lines "Fit Visual"
[image6]: ./output_images/final "Output"
[video1]: ./output_images/project_video_out.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected image
The code for this step is contained in the second and third code cells of the Jupyter notebook located in `P4-Advanced_Lane_Lines.ipynb`.

To calibrate the camera I used the chessboard method which involves using a series of pictures of a chessboard taken at various angles. This includes preparing the object points based on a 9 by 6 grid and then using the OpenCV function `cv2.findChessboardCorners()` to find the corners in the chessboard images.

With the object points and image points found I used the OpenCV function `cv2.calibrateCamera()` to calculate the camera calibration points.

This allows me to use the OpenCV function `cv2.undistort(img, mtx, dist, None, mtx)` to undistort all future images taken by the original camera. See below of example chessboard images, note how the verticle line on the right is markedly less curved after it has been undistorted.

Original             |  Undistorted
:-------------------------:|:-------------------------:
![alt text][image1]        |  ![alt text][image1.5]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

As described above here is an example of one of the test road photos undistorted. The differences are subtle but if you look closely you can see a change in the car bonnet and in the position of the white car.

Original             |  Undistorted
:-------------------------:|:-------------------------:
![alt text][image2]        |  ![alt text][image2.5]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and the absolute sobel thresholds to create the final binary image. I created the pipeline and functions so it is fairly easy to try out new combinations and visualize the results.

The final process I used involved making 2 copies of the original undistorted image. The first I converted from RGB to HLS and then took the L (Lightness) channel and passed it into an absolute sobel threshold filter with a threshold set at (20, 200).

The second copy I also converted to HLS but instead took the S (saturation) channel with a threshold of (145, 255). Here I selected a binary output.

I then combined these two output images. The result can be seen below in both the color version which shows the contribution by each image and the final binary used to find the lane lines.

Undistorted             |  HLS - Lightness
:-------------------------:|:-------------------------:
![alt text][image3]        |  ![alt text][image3.1]

HLS - Saturation            |  Absolute Sobel threshold
:-------------------------:|:-------------------------:
![alt text][image3.2]        |  ![alt text][image3.3]

Colored Combined Image             |  Combined Binary Image
:-------------------------:|:-------------------------:
![alt text][image3.4]        |  ![alt text][image3.5]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears cell 9 of the jupyter notebook.
The `warp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
def warp(img):
    img_size = (img.shape[1], img.shape[0])

    src = np.float32(
        [[575, 450],
         [705, 450],
         [1080, 690],
         [200, 690]])

    dst = np.float32(
        [[200, 0],
         [1080, 0],
         [1080, 690],
         [200, 690]])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return warped, Minv
```
Using the above function on our image generated above, transforms it into a birds eye view of the lane lines.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used the histogram method to find the lane lines. This works by generating a histogram for each slice in the image and using the histogram peaks to find the area most likely to be lane lines. This is then repeated sliding up the image until the lane lines have been found as shown in the image below.

![alt text][image5]

once the line lanes have been found I fit a second order polynomial to the left and right lane pixel positions using the following code.

```python
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
```

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius was found by calculating the curvature of the lane lines in pixels and then converting that to meters.

```python
def radius(ploty, left_fitx, right_fitx):
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    y_eval = np.max(ploty)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')
    return left_curverad, right_curverad
```
The offset calculation is shown as:

```python
# FIND CENTER Plot center
camera_position = result.shape[1]/2
lane_center = (right_fitx[719] + left_fitx[719])/2
center_offset_pixels = abs(camera_position - lane_center)
vehicle_position = center_offset_pixels / 12800 * 3.7
left_or_right = "left" if vehicle_position > 0 else "right"
result = cv2.putText(result, 'Vehicle is %.2fm %s of center' % (np.abs(vehicle_position), left_or_right), (50, 100), font, 1, (255, 255, 255), 2)
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in cell 16 of the jupyter notebook in the function `draw_img()`. This was done drawing the lanes onto a warped blank image and then using the warpPerspective() function to transform the lane line image back into the original perspective. This image is then combined with an original undistorted image.

Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I have produced what I would call a MVP, minimally viable product. It achieves the required result for the test video however it does suffer from a number of issues:

I did not fully implement the faster lane find based on the previous lane position and so on my current computer it takes about 2 minutes to process a 50 second clip. Thus it would not work in real time.

I have not implemented any anomaly filter, so strong changes in brightness or distracting lines on the road would fool the current system.
