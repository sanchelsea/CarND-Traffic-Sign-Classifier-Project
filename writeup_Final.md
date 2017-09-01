#**Traffic Sign Recognition** 

##Writeup Template

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

[image1]: file:///./examples/Hist_DataSet.png "Visualization"
[image2]: file:///./examples/TrainImgs.png "Sample Train Images"
[image3]: file:///./examples/TestImgs.png "Sample Test Images"
[image4]: file:///./examples/NormalizedImgs.png "Sample Normalized Train Images"
[image5]: file:///./examples/AugmentedImgs.png "Sample Augmented Train Images"
[image6]: file:///./examples/SampleImgs.png "Sample Images"
[image7]: file:///./examples/softmax1.png "Softmax 1"
[image8]: file:///./examples/filter1.png "Filter 1"
[image9]: file:///./examples/filter2.png "Filter 2"
[image10]: file:///./examples/5.png "Traffic Sign 5"
[image11]: file:///./examples/6.png "Traffic Sign 6"
[image12]: file:///./examples/trloss1.png "Training Loss"
[image13]: file:///./examples/accuracy1.png "Accuracy"
[image14]: file:///./examples/ConfusionMatrix.png "Confusion Matrix"
[image15]: file:///./examples/softmax2.png "Softmax"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/sanchelsea/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

Number of training examples = 34799

Number of Validation examples = 4410

Number of testing examples = 12630

Image data shape = [32, 32]

Number of classes = 43


####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed in the test, validation and test sets.  ...

![Distribution of Train, Valid and Test Sets][image1]

Below are some sample images from Train data set.
![alt text][image2]

Below are some sample images from Test data set.
![alt text][image3]


###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

For Pre-processing, I used histogram equalization and normalization. 
Here are some sample images after histogram equalization and normalization.

![alt text][image4]

I also tried converting it to grayscale since i felt color had little significance in classifying traffic signs, but the test results were not as good as using color images. Didnt spend enough time to understand the reason. Hence I decided to proceed with color images.

I decided to generate additional data because the model I am using has X parameters, tuning which requires a vast data set. To add more data to the the data set, I used the imgaug library and applied the following techniques:

 1. Scale images to 80-120% of their size
 2. Translate by -10 to +10 percent 
 3. Rotate by -10 to +10 degrees
 4. Shear by -10 to +10 degrees

Here are some examples of augmented images:

![alt text][image5]

I have not changed the distribution of classes during augmentation. I did notice a bunch of people using augmenting the data to have approx equal samples from each class. I did try to make the training set have equal distribution across classes but couldn't conclude if it helped or not. 
After augmenting the dataset, I didn't create a new validation set and used the one provided earlier. Summary of data set after augmentation.
Number of training examples = 382789
Number of Validation examples = 4410
Number of testing examples = 12630


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         	|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x12 	|	 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x24 				| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 14x14x48 	|	 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x96 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x96 
|
| Fully connected		| input = 2400, output = 480								
|
| RELU  				|        									
|
| Dropout  				|  keep prob:0.5      									
|
|
| Fully connected		| input = 480, output = 84								
|
| RELU  				|        									
|
| Dropout  				|  keep prob:0.5      									
|
| Softmax        		| input = 84, output = 43								
|
| 						|												|
|						|												|
  


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used 
Optimizer: Adam optimizer
Batch size: 256
No of epochs:  15 
Keep Prob: 0.5

I experimented the following:

 1. Greyscale vs Color
 2. Different batch sizes
 3. Different learning rates
 4. Avg Pooling vs Max Pooling
 5. Keep Prob for drop outs
 6. Different architectures by tweaking no of layers and depth.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.994 
* test set accuracy of 0.983

![alt text][image12]

![Accuracy][image13]

I started with the Lenet architecture which was created as part of previous exercise. 
I was getting train accuracy of about 93% and validation accuracy of about 88% and understood it was over fitting. Added dropouts to fully connected layers and L2 regularization to avoid overfitting.

In order to improve the performance, I modified the architecture to add more convolution layers. Followed similar style to VGG16 where we have couple of convolution layers followed by max pool. This improved the test accuracy to around 95%. 
In order to improve it further, I increased the depth of the filters. (doubled the filters in each layer) 

I also experimented different keep probability for drop out and settled at 0.5 as it seemed to provide the best results. Drop out helped with avoiding overfitting.

Plotted the confusion matrix to understand that the model is working well. Also used the 5 random German traffic signs found on web to validate the model. 

![alt text][image14]
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] 

The second image (Bumpy road) might be difficult to classify because there are multiple signs like Cycling, which looks very similar to bumpy road if the img is noisy. 
Similarly, the fourth image (No vehicles) could also get tricky if there is any noise in the center of the image. 

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30 km/h       		| 30 km/h   									| 
| Bumpy road   			| Bumpy road 										|
| Ahead only			| Ahead only											|
| No Vehicles      		| Roundabout Mandatory					 				|
| Go straight or left	| Go straight or left      							|
| General caution   	| General caution      							|

The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83% even though the test set gives an accuracy of 97%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The model is predicting all the correct images with 100% certainty like the one below.

![alt text][image7]


For the fourth image, where is it predicting incorrectly, lets look at the top 5 probabilities.


![alt text][image15]

From the predictions, the first guess is a triangle which has a similar gradient. But the shape is different.
The second guess is the right one but has only 5% certainty.  

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

From the visual output, it looks like the network detects the gradients(edges) first. 
Then subsequent layers combines these filters to produce complicated ones. 
The fourth convolution layer seems more like the spatial one.

First Convolution Layer
![alt text][image8]

Last Convolution Layer
![alt text][image9]
