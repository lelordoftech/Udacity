**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/training_visualization.png "Training examples Visualization"
[image2]: ./examples/validation_visualization.png "Training validation Visualization"
[image3]: ./examples/testing_visualization.png "Testing examples Visualization"
[image4]: ./examples/grayscale.png "Grayscaling"
[image5]: ./examples/normalization.png "Normalization"
[image6]: ./examples/rotateLeft.png "Increase training data set - Rotate Left"
[image7]: ./examples/rotateRight.png "Increase training data set - Rotate Right"
[image8]: ./traffic-signs-data/16.png "Traffic Sign 1"
[image9]: ./traffic-signs-data/18.png "Traffic Sign 2"
[image10]: ./traffic-signs-data/33.png "Traffic Sign 3"
[image11]: ./traffic-signs-data/25.png "Traffic Sign 4"
[image12]: ./traffic-signs-data/11.png "Traffic Sign 5"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
## Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

## Data Set Summary & Exploration

**1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.**

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

**2. Include an exploratory visualization of the dataset.**

Here is an exploratory visualization of the data set. It is a bar chart showing how the data was distributed.

![alt text][image1]
![alt text][image2]
![alt text][image3]

## Design and Test a Model Architecture

**1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)**

As a first step, I decided to convert the images to grayscale because when I tested with grayscale, it always make the result is better than RGB.
I think this tecniques will help me decrease the size of the output 3 times (change the depth from 3 to 1).
My calculation will be faster and I can prevent overfitting.

Here is an example of a traffic sign image before and after grayscaling.
![alt text][image4]

As a last step, I normalized the image data because I want keep values roughly around a mean of zero.
So it make it much easier for the optimizaton to proceed numerically.

The formular is **(pixel - 128)/ 128**

Here is an example of a traffic sign image before and after normalization
![alt text][image5]

I decided to generate additional data because it will help me improve model performance.

To add more data to the the data set, I used the rotation techniques with value in range +-15 degree.
With this techniques, my model will learn more cases and it will improve the performance.

Here is an example of an original image and an augmented image:

Rotation to the **LEFT**

![alt text][image6]

Rotation to the **RIGHT**

![alt text][image7]

So I get an augmented data set with the size is increase 3 times: 34799*3=104397

**2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.**

I used LeNet model and my final model consisted of the following layers:

| Layer                 |     Description                               |
|:---------------------:|:---------------------------------------------:|
| Input                 | 32x32x3 RGB image                             |
| Convolution 5x5       | 2x2 stride, same padding, outputs 32x32x32    |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, outputs 16x16x32                  |
| Convolution 5x5       | 2x2 stride, same padding, outputs 16x16x64    |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, outputs 8x8x64                    |
| Convolution 5x5       | 2x2 stride, same padding, outputs 8x8x128     |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, outputs 4x4x128                   |
| Concat 3 convolution  |               outputs 14336                   |
| layer                 |                                               |
| Dropout               | keep 50%                                      |
| Fully connected       | output 400                                    |
| RELU                  |                                               |
| Fully connected       | output 120                                    |
| RELU                  |                                               |
| Fully connected       | output 84                                     |
| RELU                  |                                               |
| Fully connected       | output 43                                     |

**3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.**

To train the model, I used:
* Batch size: 128
* Epochs: 10
* Learning rate: 0.001

At this time, I used these values because my laptop is weak for calculation.
It does not have GPU for model training.
And I have some problem to contact and use AWS services for training.
So I will try to increase EPOCHS, decrease LEARNING RATE and optimize BATCH SIZE later when I have changes.

To reduce sample data, I used Max pooling for 3 layer Convolution.
To help model can access to all lower layer, I merged all output of 3 layer Convolution.
By this way, I see that my model is better.
To prevent overfitting, I used Dropout with keep probability is 50%

**4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.**

The results on the training, validation and test sets were calculated in code 25th cell of the Ipython notebook.
To get the validation set accuracy to be at least 0.93, I try some process and compare the result:
* Convert RGB to gray scale
* Augmented data: rotation technique
* Increase number of convolution layer
* Increase number of full connected layer
* Increase/Decrease batch size:
* Increase epochs
* Decrease learning rate
* Apply Dropout technique

My final model results were:
* Training set accuracy of 0.996
* Validation set accuracy of 0.960
* Test set accuracy of 0.936

## Test a Model on New Images

**1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.**

Here are five German traffic signs that I found on the web:

Link: http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip

![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]

The first image might be easy to classify because it look like a standard traffic sign.
The sencond, third and fifth image might be difficult to classify because it have noise behind the traffic sign.
The fourth image might be difficult to classify because it is not clearly and inclined to the right.

**2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).**

Here are the results of the prediction:

| Image                                     |     Prediction                            |
|:-----------------------------------------:|:-----------------------------------------:|
| Vehicles over 3.5 metric tons prohibited  | Vehicles over 3.5 metric tons prohibited  |
| General caution                           | General caution                           |
| Turn right ahead                          | Turn right ahead                          |
| Road work                                 | Road work                                 |
| Right-of-way at the next intersection     | Right-of-way at the next intersection     |

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.
This compares favorably to the accuracy on the test set of 93.6%

Maybe these samples is too easily for my model.
I will try another samples later.

**3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)**

The code for making predictions on my final model is located in the 31th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.999), and the image does contain a "Vehicles over 3.5 metric tons prohibited" sign. The top five soft max probabilities were as below:

| Probability           |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| .999                  | Vehicles over 3.5 metric tons prohibited      |
| .0001                 | Roundabout mandatory                          |
| .0000001              | Speed limit (100km/h)                         |
| .00000000000001       | Speed limit (120km/h)                         |
| .000000000000008      | No passing                                    |

For the second image, the model is relatively sure that this is a stop sign (probability of 1.0), and the image does contain a "General caution" sign. The top five soft max probabilities were as below:

| Probability           |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| 1.0                   | General caution                               |
| .0000000000017        | Traffic signals                               |
| .00000000000004       | Pedestrians                                   |
| .000000000000002      | Road work                                     |
| .00000000000000002    | Right-of-way at the next intersection         |

For the third image, the model is relatively sure that this is a stop sign (probability of 1.0), and the image does contain a "Turn right ahead" sign. The top five soft max probabilities were as below:

| Probability           |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| 1.0                   | Turn right ahead                              |
| .0000000000012        | Go straight or left                           |
| .00000000000012       | Stop                                          |
| .0000000000001        | Keep left                                     |
| .00000000000007       | No passing for vehicles over 3.5 metric tons  |

For the fourth image, the model is relatively sure that this is a stop sign (probability of 1.0), and the image does contain a "Road work" sign. The top five soft max probabilities were as below:

| Probability           |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| 1.0                   | Road work                                     |
| .000000011            | Bicycles crossing                             |
| .00000000019          | Road narrows on the right                     |
| .0000000000599        | Children crossing                             |
| .000000000025         | Slippery road                                 |

For the fifth image, the model is relatively sure that this is a stop sign (probability of 1.0), and the image does contain a "Right-of-way at the next intersection" sign. The top five soft max probabilities were as below:

| Probability           |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| 1.0                   | Right-of-way at the next intersection         |
| .0000000000001        | Beware of ice/snow                            |
| .00000000000000008    | Double curve                                  |
| .00000000000000003    | General caution                               |
| .0000000000000000075  | Road work                                     |

When I have free time I will come back and try other techniques to get higher validation accuracy, maybe more than 98%.

**Reference:**
* https://chatbotslife.com/german-sign-classification-using-deep-learning-neural-networks-98-8-solution-d05656bf51ad
* http://navoshta.com/traffic-signs-classification/
