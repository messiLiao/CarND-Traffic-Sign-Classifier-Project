# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image_train]: ./examples/hist_train.png "Visualization"
[image_valid]: ./examples/hist_validation.png "Visualization"
[image_test]: ./examples/hist_test.png "Visualization"
[image2]: ./examples/traffic_sign_4_gray.png "Color image"
[image2_gray]: ./examples/traffic_sign_4_gray.png "Grayscaling"
[image2_hist]: ./examples/traffic_sign_4_hist.png "Histogram equalization"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/traffic_sign_0.png "Traffic Sign 1"
[image5]: ./examples/traffic_sign_1.png "Traffic Sign 2"
[image6]: ./examples/traffic_sign_2.png "Traffic Sign 3"
[image7]: ./examples/traffic_sign_3.png "Traffic Sign 4"
[image8]: ./examples/traffic_sign_5.png "Traffic Sign 5"
[image_epoch_20]: ./examples/ValidationAccuracy_20.png "Validation Accuracy EPOCH=20"
[image_epoch_200]: ./examples/ValidationAccuracy_200.png "Validation Accuracy EPOCH=200"
[image_batch_size_64_128_512]: ./examples/ValidationAccuracy_bs_64_128_512.png "Validation Accuracy BATCH_SIZE=[64, 128, 512]"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/messiliao/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic signs data set:
Number of training examples:
n_train = 34799

Number of validation examples:
n_validation = 4410

Number of testing examples.
n_test = 12630

The shape of an traffic sign image:
image_shape = (32, 32, 3)

The number of unique classes/labels in the dataset is
n_classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing number of each class in 3 datasets.

![alt text][image_train]
![alt text][image_valid]
![alt text][image_test]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I convert the images to grayscale because image need histogram equalization. Some images too dark to recognize.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]
![alt text][image2_gray]

As a second step. I use cv2.equalizeHist to equalize the grayscale image's histogram. This would enhanced the features. Here is an example of a traffic sign image before and after histogram equalization.

![alt text][image2_gray]
![alt text][image2_hist]

As a last step, I normalized the image data because 

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

|         Layer         |                   Description                 | 
|:---------------------:|:---------------------------------------------:| 
| Input                 |          32x32x1 GRAY Scale image             | 
| Convolution 3x3       | 5x5 kernel size, 1x1 stride, VALID padding    |
| RELU                  |                                               |
| Max pooling           |        2x2 stride,  VALID padding             |
| Convolution 3x3       | 5x5 kernel size, 1x1 stride, VALID padding    |
| RELU                  |                                               |
| Max pooling           |        2x2 stride,  VALID padding             |
| Fully connected       | Input 400, Output 120.                        |
| RELU                  |                                               |
| Fully connected       | Input 120, Output 84.                         |
| RELU                  |                                               |
| Fully connected       | Input 84, Output 43.                          |
| RELU                  |                                               |
| Softmax               | Output 43.                                    |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an batch_size=128, epochs=20, learning_rate=0.001, here is the curve.

![alt text][image_epoch_20]


Obviously. epoch is too small. so I set EPOCH=200. then i get a new result.

![alt text][image_epoch_200]

so. EPOCH should be greate than 100.

Batch size also was a import hyperparameters. Here give a figure to show difference. learning_rate=0.001, EPOCH=200
![alt text][image_batch_size_64_128_512]

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
1. training set accuracy is 0.958
2. validation set accuracy is 0.952
3. test set accuracy was 0.931

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image                 |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign             | Stop sign                                     | 
| U-turn                | U-turn                                        |
| Yield                 | Yield                                         |
| 100 km/h              | Bumpy Road                                    |
| Slippery Road         | Slippery Road                                 |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| .60                   | Stop sign                                     | 
| .20                   | U-turn                                        |
| .05                   | Yield                                         |
| .04                   | Bumpy Road                                    |
| .01                   | Slippery Road                                 |


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

