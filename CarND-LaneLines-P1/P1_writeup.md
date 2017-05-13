**Finding Lane Lines on the Road**

[//]: # (Image References)

[image1]: ./images/grayscale.jpg "Grayscale"
[image2]: ./images/gaussian.jpg "Gaussian"
[image3]: ./images/canny.jpg "Canny"
[image4]: ./images/mask.jpg "Mask"
[image5]: ./images/houghline.jpg "HoughLine"

---

### Reflection

###1. Describe the pipeline

My pipeline consisted of 5 steps:

Step 1: Convert the images to grayscale

![alt text][image1]

Step 2: Apply Gaussian smoothing to suppress noise and spurious gradients

![alt text][image2]

Step 3: Apply Canny transform

![alt text][image3]

Step 4: Apply image region masking

![alt text][image4]

Step 5: Apply HoughLine transform to find lines from Canny Edges and return hough lines drawn

![alt text][image5]

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by 4 steps and used some global variables to backup value of the average left/right line:

Step 1: Calculate slope to decide which segments are part of the left line vs. the right line and suppress noise. In this step, I put all left line/right line to their list

Step 2: Get the average line from the list of left/right line

Step 3: I just updated global variables when I have a small value change between 2 frames. I compared the slope value of them

Step 4: I just drew the average left/right line when I have both of them

###2. Identify any shortcomings

In 1 frame when all lines have negative slope value or all lines have positive slope value, I cannot decide which segments are part of the left line vs. the right line.

###3. Suggest possible improvements

A possible improvement would be to optimize algorithm to draw a single line on the left and right lanes correctly in all case.

Another potential improvement could be to optimize algorithm to draw a single line on the left and right lanes smoothy base on 1 series frame, not only 1 frame.
