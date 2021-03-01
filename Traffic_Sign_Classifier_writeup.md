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

[image1]: ./examples/grayscale.jpg "Grayscaled Image"
[image2]: ./examples/original.jpg "Original Image"
[image3]: ./examples/normalized.jpg "Normalized Image"
[image4]: ./traffic_signs/sign_50speed.jpg "50 Kph Speed"
[image5]: ./traffic_signs/sign_leftarrow.jpeg "Turn Left Ahead"
[image6]: ./traffic_signs/sign_warning.png "General Caution"
[image7]: ./traffic_signs/sign_forwardArrow.png "Ahead Only"
[image8]: ./traffic_signs/sign_priority.png "Priority Road"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

This is the writeup and the project code can be found in Traffic_Sign_Classifier.ipynb and a HTML version is also available.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used standard python and numpy library functions to calculate the summary statistics for the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

For the exploratory visualization I output 30 random signs from the test set with their corresponding labels. The images will be different everytime, but this provides good insight into what the computer is seeing and what measures should be taken to pre-process the data.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

First, I converted all of the images to grayscale. The color is not needed to classify each sign and adds unnecessary complexity to the training. By removing the color channels, the computer is able to focus on the defining features of each sign.  

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]![alt text][image1]

I also normalized the data so that it would have equal variance.

Here is an example of an original image and an normalized image:

![alt text][image2]![alt text][image3]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

For my model I started with the LeNet Architecture and added dropout layers to prevent overfitting.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image  						| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Fully connected		| input 400, output 120 						|
| RELU					|												|
| Dropout				|												|
| Fully connected		| input 120, output 84  						|
| RELU					|												|
| Dropout				|												|
| Fully connected		| input 84, output 43   						|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The model is run to obtain the logits and these are passed into tf.nn.softmax_cross_entropy_with_logits along with the one-hot encoded labels. Using the result of softmax_cross_entropy_with_logits the average cross entropy is calculated. An Adam optimizer is used to help minimize the average cross entropy and improve the training. 

The following values were found to give me the best performance:
Epochs: 20
Batch Size: 128
Learning Rate: 0.001
Training Keep Rate: 0.05

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.992
* validation set accuracy of 0.949
* test set accuracy of 0.936

To get these results, multiple parameter combinations were tried. I started with the standard LeNet5 architecture first, which resulted in validation set accuracy over 90% when the parameters were tuned correctly. To increase the performance the dropout layers were added to ensure that the model was not overfitting. After adding the additional layers the performance increased to around 95% (slight variations depending on the run). The LeNet was a suitable architecture for this problem because it a simple and well known architecture and provided good results on the MNIST data set. The accuracy in the training, validation and testing show that the model is working well. With additional modifications the accuracy likely could be increased.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

In general the images should not be overly difficult to classify, with a couple exceptions. 

The "priority road" sign will be difficult because it also has a forward arrow in it which the system could confuse as a straight only sign. Also, the 50 Kph speed sign is very similar to other speed signs which could make it difficult to differentiate.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 50 km/h       		| 30 km/h   									|
| Turn Left Ahead		| Turn Left Ahead								|
| General Caution		| General Caution     							|
| Ahead Only			| Ahead Only 									|
| Priority Road    		| Ahead Only					 				|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This is much lower accuracy than the test set achieved. It was unexpected that it would incorrectly classify the 50 Kph speed sign. It also incorrectly classified the priority road sign. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 18th cell of the Ipython notebook.

For each image, the corresponding top_k predictions are also output as images with their respective softmax probabilities. 

For the first image, the model is relatively sure that this is a 30 km/h sign (probability of 0.9989), but the image does not contain a 30 km/h sign it actually contains a 50 km/h sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9989        			| 30 km/h   									| 
| .0011    				| 70 km/h 										|
| .00					| 20 km/h										|
| .00	      			| 50 km/h						 				|
| .00					| 120 km/h      								|


For the second image, the model is relatively sure that this is a Turn Left Ahead sign (probability of 0.9918), and the image does contain a Turn Left Ahead sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9918         		| Turn Left Ahead   							| 
| .0063     			| Keep Right 									|
| .0009					| Beware of Ice/Snow							|
| .0009	      			| Right-of-way at next intersection				|
| .0001				    | Road Work      								|


For the third image, the model is certain that this is a General Caution sign (probability of 1), and the image does contain a General Caution sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| General Caution   							| 
| .00     				| Traffic Signals 								|
| .00					| Pedestrians									|
| .00	      			| Right-of-way at next intersection				|
| .00				    | Double Curve      							|


For the fourth image, the model is certain that this is an "Ahead Only" sign (probability of 1), and the image does contain an "Ahead Only" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Ahead Only   									| 
| .00     				| Dangerous curve to the left 					|
| .00					| 60 km/h										|
| .00	      			| Bicycles Crossing				 				|
| .00				    | Go Straight or right  						|


For the fifth image, the model is relatively sure that this is an Ahead Only sign (probability of 0.9966), but the image does not contain an Ahead Only sign, it actually contains a priority road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9966         		| Ahead Only   									| 
| .0014     			| 60 km/h 										|
| .0009					| Turn Left Ahead								|
| .0005	      			| Go Straight or Right			 				|
| .0003				    | Vehicles over 3.5 metric tons prohibited     	|


The model appears to be certain of the decision is chooses even when that decision is wrong. It was unexpected that the model would consider the 50 km/h sign but still mark it as a 30 km/h sign. In general performance is marginal, and I would expect performance on more images to be similar, possibly improving slightly.

