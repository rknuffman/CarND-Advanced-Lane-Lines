# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
writeup_template.md
[//]: # (Image References)

[image1]: ./output_images/undistorted1.png "Undistorted Checkerboard"
[image2]: ./output_images/undistorted2.png "Undistorted Road"
[image3]: ./output_images/binary.png "Binary Example"
[image4]: ./output_images/warped_straight.png "Warp Example"
[image5]: ./output_images/histogram.png "Histogram"
[image6]: ./output_images/margin.png "Margin"
[image7]: ./output_images/output.png "Output"
[video1]: ./project_video_out.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the utilities file located in [alf_utils.py](https://github.com/rknuffman/CarND-Advanced-Lane-Lines/blob/master/alf_utils.py) (function "undistort" in lines 5 through 25).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 27 through 124 in [alf_utils.py](https://github.com/rknuffman/CarND-Advanced-Lane-Lines/blob/master/alf_utils.py).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform appears in the 3rd code cell of the IPython notebook, [Pipeline.ipynb](https://github.com/rknuffman/CarND-Advanced-Lane-Lines/blob/master/Pipeline.ipynb)).  The code uses source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points as follows:

| Source         | Destination   | 
|:--------------:|:-------------:| 
| 180, 720       | 100, 720      | 
| 550, 475       | 100, 0        |
| 740, 475       | 1180, 0       |
| 1160, 720      | 1180, 720     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I utilized histograms of pixel intensities across horizontal slices of the warped binary lane image to do initial lane discovery.  Candidate pixels for left and right lane lines were then fit with a polynomial of degree 2 to allow for an appropriate level of curve.  

![alt text][image5]

Additional search was limited to a margin of 50 pixels around previously discovered lanes.  Each additional lane pair was validated against a running average of historical widths calculated from lane line pairs.  Any pairs with a lane width deviation +/- 10% were ignored.  

As driving continues, a history of polynomials fit to left and right lanes are maintained and averaged over a running sequence of 4 frames to help smooth predictions.    

![alt text][image6]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 249 through 269 in my code in [alf_utils.py](https://github.com/rknuffman/CarND-Advanced-Lane-Lines/blob/master/alf_utils.py)

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 230 through 247 in my code in [alf_utils.py](https://github.com/rknuffman/CarND-Advanced-Lane-Lines/blob/master/alf_utils.py) in the function `fill_lane()`.  Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I faced several challenges in this build.

* Tuning usage of color spaces and image gradients for detecting lane pixels
* Building in functionality to catch mis-identified lanes
* Incorporating a history of detected lines to smooth 'jitter'

Despite being more robust to shadows and visual differences in the paving surface, there are still many opportunities for my model to lose track of the lines.  If a vehicle merged into your lane in front of you, the model would lose one line, but could continue leveraging past predictions.  If, however, the curve changes at the same time, the past predictions won't be appropriate for the car's current state.

A potential solution could be to infer the second line based on a confident detection of the first lane line, and a known lane width.  This way the car can still update the polynomial fit based on the single line being detected, rather than being forced to abandon both due to a merging vehicle obscuring only one line.