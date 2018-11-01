# **Finding Lane Lines on the Road** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<img src="examples/laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

## Project 1
## Date: 9/17/2018

Overview
---

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

In this project you will detect lane lines in images using Python and OpenCV.  OpenCV means "Open-Source Computer Vision", which is a package that has many useful tools for analyzing images.  
---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[image1]: ./test_images/solidWhiteCurve.jpg 

[image2]: ./test_images_output/solidWhiteCurve_output.jpg


### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. 
* I converted the images to grayscale
* I defined a kernal size and apply Gaussian smoothing
* I defined parameters and apply Canny edge detection
* I defined a four sides polygon to mask
* I defined the Hough transform parameter and apply Hough detected image

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by:
* I created global variable lists for left lane slope, right lane slope, left lane middle point, right lane middle point
* When a image is processed, the left and right lane are identified by their slopes, and saved seperately to their according global list.
* The middle point and left and right lane are also calculated, by taking the average of all coordinates on the lane, and saved to their global list.
* I take average value over multiple images for a stable vaue of slope and middle point, in order to smooth out any eccentric values.
* Lastly I use the slope and the middle point, using the function of y=A*x+B, and calculate value of B.
* Since I know the height of the masked image (from the polygon masked step), I can obtain the two ends of each lane.



### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen if there is any vehicle driving right in front of my vehicle, and it blocks part of my lane, my algorithm is not implemented to handle that.

Another shortcoming could be if the lane is heavily curve in front of the vehicle, my algorithm will not able to handle that. Since it is only able to draw straight lines.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to instead of taking the average of slope and middle point of lane, I can calculate the normal distribution over a period of time and filter out the extremely large and small values.

Another potential improvement could be to tune the parameter values for Hough and Canny functions to find the best detection result.
