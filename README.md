# **Traffic Sign Recognition** 



**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1.2.1]: ./output_images/TrainData.png "Train Data"
[image1.2.2]: ./output_images/ValidData.png "Valid Data"
[image1.2.3]: ./output_images/TestData.png "Test Data"
[image2.4.1]: ./output_images/LeNet_accuracy.png "Test Data"
[image3.1.1]: ./new_traffic_signs/img1.jpg "Traffic Sign 1"
[image3.1.2]: ./new_traffic_signs/img2.jpg "Traffic Sign 2"
[image3.1.3]: ./new_traffic_signs/img3.jpg "Traffic Sign 3"
[image3.1.4]: ./new_traffic_signs/img4.jpg "Traffic Sign 4"
[image3.1.5]: ./new_traffic_signs/img5.jpg "Traffic Sign 5"

## 1. Data Set Summary & Exploration

### 1. Data Set Summary

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 42

### 2. Chart

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distribute over the training, validation, and test dataset

![alt text][image1.2.1]

![alt text][image1.2.1]

![alt text][image1.2.1]

The chart show that all class have equal ratio across the 3 class. However, in the training dataset, there are certain class which have significantly lower amount of data across the 3 dataset. I will increase those data through data augmentation in the data pre-processing phase.

## 2. Design and Test a Model Architecture

### 1. Data Pre-processing

I performed data augmentation to increase the number of data in a class which have relatively low data compare to other class. This will prevent the model from learning too much from class with more data and will learn more about the rare case. I used tensorflow `image` class to resize, random cropping, and adjust random brightness to the image. The augmentation is perform by `augmented` function.

Other pre-processing is perform inside of the model architecture such as normalizing.

### 2. Model Architecture

Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

|           Layer		            |     	     Description   			    | 
|:---------------------------------:|:-----------------------------------------:| 
| Input         		            | 32x32x3 RGB image   						  | 
| Normalize Image                   | Subtract each pixel by 128 then divide by 128| 
| Convolution 5x5                 	| 1x1 stride, valid padding, outputs 28x28x10|
| RELU			        		    |											|
| Max pooling	                   	| 2x2 stride,  outputs 14x14x20 			|
| Convolution 5x5                   | 1x1 stride, valid padding, outputs 10x10x16 |
| RELU			        		    |											|
| Max pooling	                   	| 2x2 stride,  outputs 5x5x16 		    	|
| Flatten                   		| outputs 400x1        						|
| Fully connected		            | outputs 120x1        						|
| Fully connected		            | outputs 84x1        						|
| Fully connected	            	| outputs 43x1        						|
| Softmax				            | Turn output to probabilities				|
 
### 3. Training Model

To train the model, I used an softmax cross entropy with Adam Optimizer. The learning rate is 0.001 with EPOCHS of 50 and Batch Size of 128.

### 4. Model Approach

Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1
* validation set accuracy of 0.941
* test set accuracy of 0.930

![alt text][image2.4.1]

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

I based my model on LeNet because it is a popular image detection CNN model. It could detected common feature on number drawing; therefore, I used it as a starting point for my model

Certain modification I made was adding the normalization, increasing the number of filters in the both convolution layers to detect more features, and modify the Fully Connected layer so it output the correct number of classes for my sign detection.

Some relevant information for detecting traffic signs are colors and line shape. Different sign have different colors which will help the model distinguish the sign, which is why I did not convert it to grayscale. The line shape also identify the sign shape and image on it to help determine the type of sign in the image.

The training, validation, and test accuracy are all above 90%, which is very high. The data could be a little overfit becauase for some EPOCHS, it predicts all the image correctly. I try to fix this problem with dropout on the convolution layer, but that decrease the accuracy of the validation test to about 80%. By performing data augmentation, it gives more data to the model and make it less overfit.


## 3. Test a Model on New Images

#### 1. New Sign Images

Here are five German traffic signs that I found on the web:

![alt text][image3.1.1] ![alt text][image3.1.2] ![alt text][image3.1.3]
![alt text][image3.1.4]![alt text][image3.1.5]

The first image might be difficult to classify because the image is not at the center unlike the image in the training dataset. The last image also have addition text sign to see if it will cause the model to give wrong prediction.

#### 2. New Sign Images Prediction

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority Road    		| Priority Road   							| 
| Road Work     		| Bicycles Crossing 						|
| No Entry				| No Entry									|
| Stop Sign	      		| Stop Sign					 				|
| General Caution		| No Passing     							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This is lower than the accuracy of the test dataset which is 93.3%. This might be because the image inside the dataset are well position so the region of interest cover the sign only. However, my new images include empty area and unrealated text sign which is not related to the sign and they are not position near the center, causing error predicting the result.

#### 3. Model Top 5 Prediction 

For the first image, the model is relatively sure that this is a Priority Road (probability of 1.00), and the image does contain a Priority Road Sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Priority Road   									| 
| 2.60E-14     				| Dangerous Curve to the Right 					|
| 5.00E-15					| Speed Limit (20km/h)							|
| 4.44E-15	      			| Traffic Signals					 			|
| 2.46E-15				    | Ahead Only      							|


For the second image, the model think it's a bicycles crossing sign. However, it actually is a road work sign. However, the second top prediction is the Road Work sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99E-1         			| Bicycles Crossing   						| 
| 2.83E-5    				| Road Work 								|
| 1.28E-11					| Bumpy Road								|
| 1.95E-12	      			| Road Narrows on the Right		 			|
| 4.23E-18				    | Stop     							|

For the third image, the model predict a No Entry sign, which is correct. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| No Entry   									| 
| 2.93E-9   				| Stop 										|
| 5.01E-16					| Speed Limit (30km/h)						|
| 2.33E-18	      			| Bumpy Road								|
| 4.03E-19				    | Traffic Signals     		        		|

For the fourth image, the model predict a Stop sign, which is correct. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 4.52E-1        			| Stop   									| 
| 2.85E-1   				| Speed Limit (80km/h) 						|
| 2.22E-1					| Road Work									|
| 4.12E-2	      			| Speed Limit (30km/h)						|
| 5.58E-6				    | Yield     							    |


For the fourth image, the model predict a No Passing sign. However, the sign is actually General caution sign. The top five prediction does not contain correct prediction. One reason could be because in the image, it contains addition text below which could misled the model. This prove that the model is prone to error external text and unrelate sign in the image. Another reason could be because the training data for that sign is relatively low compare to most classes. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99E-1        			| No Passing   								| 
| 8.11E-4   				| Eng Of No Passing 						|
| 7.67E-11					| Priority Road								|
| 2.26E-12	      			| No Passing for Vehicles over 3.5 Metric Tons	|
| 9.17E-13				    | Yield     							|

## 4. Discussion

### 1. Problem

The first and most common problem with machine learning is uneven number of data in each class. Therefore, I use data augmentation by changing the contrast of the image and reposition it randomly. This would take time to create thousands and thousands of image.

Another problem is that using dropout actually lower the validation accuracy. Therefore, I decide to remove it. It might be from how all the note is actually contribute to detect the sign. Next time I could lower the probability of the dropout so it happend less and test the accuracy.

### 2. Future Improvement

In the future, we could gather more data, especially on the class that has lower data than others. Next I could use technique like SIFT to improve detection regardless of the size of image. Currently, the sign have to cover majority of the area of the image. In real life, the camera will look at a wide range and the sign the be relatively small. There is a high chance it won't detect accurately.


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


