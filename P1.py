#!/usr/bin/env python
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# 
# ## Project: **Finding Lane Lines on the Road** 
# ***
# In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 
# 
# Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.
# 
# In addition to implementing code, there is a brief writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) that can be used to guide the writing process. Completing both the code in the Ipython notebook and the writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/322/view) for this project.
# 
# ---
# Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.
# 
# **Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".**
# 
# ---

# **The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**
# 
# ---
# 
# <figure>
#  <img src="examples/line-segments-example.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> 
#  </figcaption>
# </figure>
#  <p></p> 
# <figure>
#  <img src="examples/laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p> 
#  </figcaption>
# </figure>

# **Run the cell below to import some packages.  If you get an `import error` for a package you've already installed, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt.  Also, consult the forums for more troubleshooting tips.**  

# ## Import Packages

# In[376]:


#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')

left_slope_list = list()
right_slope_list = list()
x_left_mid_previous = list()
y_left_mid_previous = list()
x_right_mid_previous = list()
y_right_mid_previous = list()
    


# ## Read in an Image

# In[377]:


#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')


# ## Ideas for Lane Detection Pipeline

# **Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**
# 
# `cv2.inRange()` for color selection  
# `cv2.fillPoly()` for regions selection  
# `cv2.line()` to draw lines on an image given endpoints  
# `cv2.addWeighted()` to coadd / overlay two images
# `cv2.cvtColor()` to grayscale or change color
# `cv2.imwrite()` to output images to file  
# `cv2.bitwise_and()` to apply a mask to an image
# 
# **Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

# ## Helper Functions

# Below are some helper functions to help get you started. They should look familiar from the lesson!

# In[378]:


import math


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image



def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    # Take dimension of the image
    imshape = img.shape
    height = imshape[0]
    width = imshape[1]
    
    
    # Create lists to hold left and right lane (x, y)s and slope
    left_lane = list()
    right_lane = list()
    left_slope = list()
    right_slope = list()
    
    # Calculate slope
    for line in lines:
        for x1,y1,x2,y2 in line:
            if x2 == x1:
                x2=x2+1
            slope = ((y2-y1)/(x2-x1)) 
            if slope > 0:
                right_slope.append(slope)
                right_lane.append((x1, y1))
                right_lane.append((x2, y2))
            else:
                left_slope.append(slope)
                left_lane.append((x1, y1))
                left_lane.append((x2, y2))
                
    # Take average of the slope
    left_slope_ave = sum(left_slope) / len(left_slope)
    right_slope_ave = sum(right_slope) / len(right_slope)
    
    # Take the middle point of each lane
    x_left_sum= 0
    y_left_sum = 0
    for xy in left_lane:
        x_left_sum += xy[0]
        y_left_sum += xy[1]

    x_left_mid = x_left_sum / len(left_lane)
    y_left_mid = y_left_sum / len(left_lane)
    
    
    x_right_sum = 0
    y_right_sum = 0
    for xy in right_lane:
        x_right_sum += xy[0]
        y_right_sum += xy[1]

    x_right_mid = x_right_sum / len(right_lane)
    y_right_mid = y_right_sum / len(right_lane)
    
    
    # y = Ax + B. Calculate B using the average slope and middle point for both lane
    B= y_left_mid - left_slope_ave * x_left_mid
    x_left_bottom = (height - B) / left_slope_ave
    x_left_top = (height/2 + 45 - B) / left_slope_ave
    
    B= y_right_mid - right_slope_ave * x_right_mid
    x_right_bottom = (height - B) / right_slope_ave
    x_right_top = (height/2 + 45 - B) / right_slope_ave
  
    # Draw lines
    cv2.line(img, (int(x_left_bottom), height), (int(x_left_top), int(height/2 + 45)) , color, 4)
    cv2.line(img, (int(x_right_bottom), height), (int(x_right_top), int(height/2 + 45)) , color, 4)
    
# Second draw line function for video processing
def draw_lines2(img, lines, color=[255, 0, 0], thickness=2):
    # Take dimension of the image
    imshape = img.shape
    height = imshape[0]
    width = imshape[1]
    
    # Create global variable for multiple image processing
    global left_slope_list 
    global right_slope_list 
    global x_left_mid_previous
    global y_left_mid_previous 
    global x_right_mid_previous 
    global y_right_mid_previous

    
    # Create lists to hold left and right lane (x, y)s and slope
    left_lane = list()
    right_lane = list()
    left_slope = list()
    right_slope = list()
    
    # Calculate slope
    for line in lines:
        for x1,y1,x2,y2 in line:
            if x2 == x1:
                x2=x2+1
            slope = ((y2-y1)/(x2-x1)) 
            if slope > 0:
                if len(right_slope_list) < 40:
                    right_slope_list.append(slope)
                else:
                    del right_slope_list[0]
                    right_slope_list.append(slope)
                #right_slope.append(slope)
                right_lane.append((x1, y1))
                right_lane.append((x2, y2))
            else:
                if len(left_slope_list) < 40:
                    left_slope_list.append(slope)
                else:
                    del left_slope_list[0]
                    left_slope_list.append(slope)
                #left_slope.append(slope)
                left_lane.append((x1, y1))
                left_lane.append((x2, y2))

                
    # Take average of the slope
    left_slope_ave = sum(left_slope_list) / len(left_slope_list)
    right_slope_ave = sum(right_slope_list) / len(right_slope_list)
    

    # Take average over last 20 images for a stable middle point on left lane
    x_left_sum= 0
    y_left_sum = 0
    for xy in left_lane:
        x_left_sum += xy[0]
        y_left_sum += xy[1]
    
    if(len(left_lane) != 0):
        if (len(x_left_mid_previous) < 20):
            x_left_mid_previous.append(x_left_sum / len(left_lane))
        else:
            del x_left_mid_previous[0]
            x_left_mid_previous.append(x_left_sum / len(left_lane))
        if(len(y_left_mid_previous) < 20):
            y_left_mid_previous.append(y_left_sum / len(left_lane))
        else:
            del y_left_mid_previous[0]
            y_left_mid_previous.append(y_left_sum / len(left_lane))        
    
    x_left_mid = sum(x_left_mid_previous) / len(x_left_mid_previous)
    y_left_mid = sum(y_left_mid_previous) / len(y_left_mid_previous)

    # Take average over last 20 images for a stable middle point on right lane
    x_right_sum = 0
    y_right_sum = 0
    for xy in right_lane:
        x_right_sum += xy[0]
        y_right_sum += xy[1]
        
    if(len(right_lane) != 0):
        if(len(x_right_mid_previous) < 20):
            x_right_mid_previous.append(x_right_sum / len(right_lane))
        else:
            del x_right_mid_previous[0]
            x_right_mid_previous.append(x_right_sum / len(right_lane))
        if(len(y_right_mid_previous) < 20):
            y_right_mid_previous.append(y_right_sum / len(right_lane))
        else:
            del y_right_mid_previous[0]
            y_right_mid_previous.append(y_right_sum / len(right_lane))  
        
    x_right_mid = sum(x_right_mid_previous) / len(x_right_mid_previous)
    y_right_mid = sum(y_right_mid_previous) / len(y_right_mid_previous)

    
    
    # y = Ax + B. Calculate B using the average slope and middle point for both lane
    B= y_left_mid - left_slope_ave * x_left_mid
    x_left_bottom = (height - B) / left_slope_ave
    x_left_top = (height/2 + 55 - B) / left_slope_ave
    
    B= y_right_mid - right_slope_ave * x_right_mid
    x_right_bottom = (height - B) / right_slope_ave
    x_right_top = (height/2 + 55 - B) / right_slope_ave
  
    # Draw lines
    cv2.line(img, (int(x_left_bottom), height), (int(x_left_top), int(height/2 + 55)) , color, 4)
    cv2.line(img, (int(x_right_bottom), height), (int(x_right_top), int(height/2 + 55)) , color, 4)
    

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Second Hough function for video processing
def hough_lines2(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines2(line_img, lines)
    return line_img
# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


# ## Test Images
# 
# Build your pipeline to work on the images in the directory "test_images"  
# **You should make sure your pipeline works well on these images before you try the videos.**

# In[379]:


import os
os.listdir("test_images/")

# Create output folder for test images as instructed below
path = "test_images_output/"
os.mkdir(path)


# ## Build a Lane Finding Pipeline
# 
# 

# Build the pipeline and run your solution on all test_images. Make copies into the `test_images_output` directory, and you can use the images in your writeup report.
# 
# Try tuning the various parameters, especially the low and high Canny thresholds as well as the Hough lines parameters.

# In[1]:


# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.
#os.remove("test_images_output/")

for test_image in os.listdir("test_images/"):
    path = os.path.realpath(test_image)
    
    filename_w_ext = os.path.basename(path)
    filename, file_extension = os.path.splitext(filename_w_ext)
    #print(filename_w_ext)
    # Read in and grayscale the image
    image = mpimg.imread("test_images/"+filename_w_ext)
    gray = grayscale(image)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray= gaussian_blur(gray, kernel_size)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

      # Defining a four sided polygon to mask
    imshape = image.shape
    top_left = (imshape[1]/2 - 5, imshape[0]/2 + 45)
    top_right = (imshape[1]/2 + 5, imshape[0]/2 + 45)
    vertices = np.array([[(0,imshape[0]),top_left, top_right, (imshape[1],imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    #plt.imshow(masked_edges)
    
    # Define the Hough transform parameters
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40 #minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    plt.imshow(lines)

    # Draw the lines on the edge image
    lines_edges = weighted_img(lines, image, 0.8, 1, 0.5)
    #plt.imshow(lines_edges)

    
    filename_new = 'test_images_output'+'/'+filename + '_output'+file_extension
    mpimg.imsave(filename_new, lines_edges)

    #cv2.imwrite(path, filename_new)


# ## Test on Videos
# 
# You know what's cooler than drawing lanes over images? Drawing lanes over video!
# 
# We can test our solution on two provided videos:
# 
# `solidWhiteRight.mp4`
# 
# `solidYellowLeft.mp4`
# 
# **Note: if you get an import error when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel). Still have problems? Try relaunching Jupyter Notebook from the terminal prompt. Also, consult the forums for more troubleshooting tips.**
# 
# **If you get an error that looks like this:**
# ```
# NeedDownloadError: Need ffmpeg exe. 
# You can download it by calling: 
# imageio.plugins.ffmpeg.download()
# ```
# **Follow the instructions in the error message and check out [this forum post](https://discussions.udacity.com/t/project-error-of-test-on-videos/274082) for more troubleshooting tips across operating systems.**

# In[381]:


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[382]:




def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    gray = grayscale(image)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray= gaussian_blur(gray, kernel_size)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

      # Defining a four sided polygon to mask
    imshape = image.shape
    top_left = (imshape[1]/2 - 10, imshape[0]/2 + 55)
    top_right = (imshape[1]/2 + 10, imshape[0]/2 + 55)
    vertices = np.array([[(0,imshape[0]),top_left, top_right, (imshape[1],imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    #plt.imshow(masked_edges)
    
    # Define the Hough transform parameters
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40 #minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = hough_lines2(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    #plt.imshow(lines)

  

    # Draw the lines on the edge image
    result = weighted_img(lines, image, 0.8, 1, 0.5)

    return result


# Let's try the one with the solid white lane on the right first ...

# In[383]:


white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
#clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'white_clip.write_videofile(white_output, audio=False)')


# Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.

# In[384]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))


# ## Improve the draw_lines() function
# 
# **At this point, if you were successful with making the pipeline and tuning parameters, you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform. As mentioned previously, try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines. You can see an example of the result you're going for in the video "P1_example.mp4".**
# 
# **Go back and modify your draw_lines function accordingly and try re-running your pipeline. The new output should draw a single, solid line over the left lane line and a single, solid line over the right lane line. The lines should start from the bottom of the image and extend out to the top of the region of interest.**

# Now for the one with the solid yellow lane on the left. This one's more tricky!

# In[385]:


yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
#clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,10)
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
get_ipython().run_line_magic('time', 'yellow_clip.write_videofile(yellow_output, audio=False)')


# In[386]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))


# ## Writeup and Submission
# 
# If you're satisfied with your video outputs, it's time to make the report writeup in a pdf or markdown file. Once you have this Ipython notebook ready along with the writeup, it's time to submit for review! Here is a [link](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) to the writeup template file.
# 

# ## Optional Challenge
# 
# Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!

# In[ ]:


challenge_output = 'test_videos_output/challenge.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
get_ipython().run_line_magic('time', 'challenge_clip.write_videofile(challenge_output, audio=False)')


# In[ ]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))

