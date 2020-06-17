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

![alt text][image1.2.2]

![alt text][image1.2.3]

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

My final model results were:
* training set accuracy of 1
* validation set accuracy of 0.941
* test set accuracy of 0.930

![alt text][image2.4.1]


I based my model on LeNet because it is a popular image detection CNN model. It could detected common feature on number drawing; therefore, I used it as a starting point for my model

Certain modification I made was adding the normalization, increasing the number of filters in the both convolution layers to detect more features, and modify the Fully Connected layer so it output the correct number of classes for my sign detection.

Some relevant information for detecting traffic signs are colors and line shape. Different sign have different colors which will help the model distinguish the sign, which is why I did not convert it to grayscale. The line shape also identify the sign shape and image on it to help determine the type of sign in the image.

The training, validation, and test accuracy are all above 90%, which is very high. The data could be a little overfit becauase for some EPOCHS, it predicts all the image correctly. I try to fix this problem with dropout on the convolution layer, but that decrease the accuracy of the validation test to about 80%. By performing data augmentation, it gives more data to the model and make it less overfit.


## 3. Test a Model on New Images

### 1. New Sign Images

Here are five German traffic signs that I found on the web:

![alt text][image3.1.1] 

![alt text][image3.1.2] 

![alt text][image3.1.3]

![alt text][image3.1.4]

![alt text][image3.1.5]

The first image might be difficult to classify because the image is not at the center unlike the image in the training dataset. The last image also have addition text sign to see if it will cause the model to give wrong prediction.

### 2. New Sign Images Prediction

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority Road    		| Speed Limit (60km/h)   							| 
| Road Work     		| Road Work 						|
| No Entry				| No Entry									|
| Stop Sign	      		| Speed Limit (20km/h) 					 				|
| General Caution		| Dangerous curve to the left     							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This is significantly lower than the accuracy of the test dataset which is 92.5%. This might be because the image inside the dataset are well position so the region of interest cover the sign only. However, my new images include empty area and unrealated text sign which is not related to the sign and they are not position near the center, causing error predicting the result. Moreover, I did not perform data augmentation throughout the whole training dataset because it takes a lot of time. Also, having more images would improve the models.

### 3. Model Top 5 Prediction 

For the first image, the model is relatively sure that this is a Speed Limit of 60 km/h (probability of 1.00). However, the image does contain a Priority Road Sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Speed Limit (60km/h)   									| 
| 3.79E-5     				| Children Crossing 					|
| 6.95E-9					| End of all Speed and Passing Limits							|
| 1.42E-9	      			| Pedestrians					 			|
| 2.17E-11				    | Right-of-way at the Next Intersection     							|


For the second image, the model a road work sign, which is correct. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99E-1         			| Road Work   						| 
| 1.95E-4    				| Bicycles Crossing 								|
| 7.47E-6					| Bumpy road								|
| 5.54E-13	      			| Road Narrows on the Right		 			|
| 5.85E-14				    | Slippery road							|

For the third image, the model predict a No Entry sign, which is correct. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| No Entry   									| 
| 1.32E-22   				| Speed Limit (30km/h) 										|
| 4.62E-24					| Speed Limit (20km/h)						|
| 9.21E-25	      			| No passing								|
| 3.89E-25				    | No vehicles     		        		|

For the fourth image, the model predict a Speed Limit of 20km/h sign. However, it is a Stop sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99E-1      			| Speed Limit (20km/h)   									| 
| 6.98E-6   				| Speed Limit (30km/h) 						|
| 5.27E-9					| General caution								|
| 2.93E-9      			| Pedestrians						|
| 8.87E-10				    | Stop     							    |


For the fourth image, the model predict a Dangerous curve to the left. However, the sign is actually General caution sign. The top five prediction does not contain correct prediction. One reason could be because in the image, it contains addition text below which could misled the model. This prove that the model is prone to error external text and unrelate sign in the image. Another reason could be because the training data for that sign is relatively low compare to most classes. The top five soft max probabilities were:

| Probability             	|     Prediction	        					| 
|:-------------------------:|:---------------------------------------------:| 
| 6.32E-1        			| Dangerous curve to the left   				| 
| 3.68E-1  				    | No Passing 						            |
| 3.71E-6					| Right-of-way at the next intersection			|
| 6.66E-8	      			| End of no passing	                            |
| 7.31E-10				    | Vehicles over 3.5 metric tons prohibited     	|

## 4. Discussion

### 1. Problem

The first and most common problem with machine learning is uneven number of data in each class. Therefore, I use data augmentation by changing the contrast of the image and reposition it randomly. This would take time to create thousands and thousands of image.

Another problem is that using dropout actually lower the validation accuracy. Therefore, I decide to remove it. It might be from how all the note is actually contribute to detect the sign. Next time I could lower the probability of the dropout so it happend less and test the accuracy.

### 2. Future Improvement

In the future, we could gather more data, especially on the class that has lower data than others. Next I could use technique like SIFT to improve detection regardless of the size of image. Currently, the sign have to cover majority of the area of the image. In real life, the camera will look at a wide range and the sign the be relatively small. There is a high chance it won't detect accurately.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


