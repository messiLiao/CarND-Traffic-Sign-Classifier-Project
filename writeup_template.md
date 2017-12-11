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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[origin_image]: ./examples/origin.bmp "Origin image"
[gray_image]: ./examples/gray.bmp "Gray image"
[hist_image]: ./examples/hist.bmp "histogram equalization"
[noisy_image]: ./examples/noisy.bmp "image with random noisy"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/messiLiao/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 35289
* The size of the validation set is 3920
* The size of test set is 0
* The shape of a traffic sign image is 50x50
* The number of unique classes/labels in the data set is  ?


#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Preprocessing the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


As a first step, I decided to convert the images to grayscale because it was also easy to identify and fewer computations.
```python
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```


Here is an example of a traffic sign image before and after grayscaling.
![alt text][origin_image]
![alt text][gray_image]

As a second step. I resize the images to 50x50.
```python
image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
```



As a third step, I normalized the image data because it was easy to convergence when train the model.

As a last step. I add random noisy on the images. It was a effective method to prevent overfitting.
```python
for ii in range(200):
    _x = int(random.uniform(0, IMAGE_SIZE))
    _y = int(random.uniform(0, IMAGE_SIZE))
    noisy_pix = int(random.uniform(0, 255))
    image[_x, _y] = noisy_pix
```
Here is an example of a traffic sign image after adding noisy and resize.
![alt text][noisy_image]

Here is an example of an original image and an augmented image:

![alt text][origin_image]
![alt text][noisy_image]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 50x50x1 GRAY Scale image   							| 
| Convolution 3x3     	| 1x1 stride, VALID padding, outputs 46x46x6 	|
| RELU					|						|
| Max pooling	      	| 2x2 stride,  outputs 23x23x6 				|
| Convolution 3x3	    | 1x1 stride, VALID padding, outputs 19x19x16  |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  SAME padding, outputs 10x10x16              |
| Convolution 3x3       | 1x1 stride, VALID padding, outputs 5x5x32  |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 3x3x32              |
| Fully connected       | Input 288, Output 160.         |
| RELU                  |                                               |
| Fully connected		| Input 160, Output 80.       	|
| RELU                  |                                               |
| Fully connected       | Input 80, Output 43.         |
| RELU                  |                                               |
| Softmax				| Output 43.      				|							|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Adam optimization method. At start, I only resize the picture to 32x32, no nomalizing, no adding noisy, no equalization. The validation set accuracy was 0.15. And then adding the nomalized preprocessing. The validation set accuracy was rised to 0.92. Then I changed the input shape of Lenet model to 50x50. The accuracy was rised to 0.95. Then I analised the wrong classified samples and class, I found that image is too dark and noisy, so I added histogram equalization step and in the image preprocessing, and adding random noisy pixes after equalization. After this steps, accurary was raise to 0.990. The wrong samples is hard to classified by my eyes, so I stoped.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.995
* validation set accuracy of 0.990

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

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


