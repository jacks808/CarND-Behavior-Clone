# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

By reading Nvidia paper: `End to End Learning for Self-Driving Cars`, I impl a same convolution neural work model with 5 convolution layer followed by 4 fully connected layer like: 

![](https://ws4.sinaimg.cn/large/006tNc79ly1frr8lla9qmj312010iwlc.jpg)

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 64 (model.py lines 100~104) , which first 3 conv layer use strides `2,2` (Line 100~102) and other 2 conv layer use stripes `1,1`(Line 103~104). All the convolution layer use `ReLU` activation function (Line 100~104).

The data is normalized in the model using a Keras lambda layer (code line 98). 

Because the image capture by the front camera contains a lots of tree, sky, e.g. So I add a clip opreation to cut those object off from my model imput (Line 99).

#### 2. Attempts to reduce overfitting in the model

* The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 102). 
* The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.
* I use `Early Stop` technique to reduce overfitting, I choose `epoch` 7. 

#### 3. Model parameter tuning

* The model used an `Adam` optimizer, so the learning rate was not tuned manually (model.py line 110).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to make the neural networ learn  the mesurement from the imput images. 

1. Use a fully connected neural network to make sure all pipeline is work. 
   1. Near the input image, use `normalization` technique to convert all input value from 0~255 to near 0. As we are known this is good for neural work training.
2. Then I add convolution layer to the neural network. I thought this model might be appropriate because convolution neural network is good at image recognition and those image has clear lane line. 
3. After that, My car can run out from the start line. but still some problem there. such as faceing the first left turn, it always out off the road. 
4. Then I discover something in training data, there is a lot of tree and other useless object for this task. So I decide to cut those things off.code:  `model.add(Cropping2D(cropping=((70, 25), (0, 0))))`. 
5. And I add left and right camera data to train data. because the map is a cycle, so that I add mirror image too.
6. After that, I try a more powerful network: `LeNet`, but still got bad result. The car will off road at the right turn after the bridge. 
7. Then I read this paper: `End to End Learning for Self-Driving Cars`, change my network. And try again. Much batter than ever. but this car still drive to the side of road. 
8. There is a paremeter of handle left and right camera data: If in the left camera, the mesurement will add some value, This tell the network do not get close to the left. And do the same on right camera data.  Udacity tell me use `0.2` , But I turn this value to `0.3` , So that the car will keep away from the side of rode. After some turn about `epoch`,  my network works good, the car can drive by speed `18Mph`.

![](https://ws2.sinaimg.cn/large/006tNc79ly1frrerj1mp7j30zk0rw4gb.jpg)

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture 

![](https://s1.ax1x.com/2018/05/28/C4aqud.png)

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. (Sorry for my terrible drive skill, I use the dataset which provided by udacity)
Here is an example image of center lane driving:

![](https://ws3.sinaimg.cn/large/006tNc79ly1frretmzdd7j30kc0c8tci.jpg)

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive back to the center. These images show what a recovery looks like starting from a left turn :

![](https://ws1.sinaimg.cn/large/006tNc79ly1frrew58aroj30kc0c80x1.jpg)
![](https://ws3.sinaimg.cn/large/006tNc79ly1frrex7u2gjj30kc0c8tif.jpg)

To augment the data sat, I also flipped images and angles thinking that this would tell the network how to make a good right turn.  For example, here is an image that has then been flipped:

![](https://ws4.sinaimg.cn/large/006tNc79ly1frrfbqdqv1j30z6110h93.jpg)

After the collection process, I had `48216` number of data points. I finally randomly shuffled the data set and put 20% of the data into a validation set:

![](https://ws4.sinaimg.cn/large/006tNc79ly1frs6q19vgij312c01c0t1.jpg)

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was `7` as evidenced by validation loss begin increase after `7` epoch.  I used an adam optimizer so that manually training the learning rate wasn't necessary.



## Some skill to improve this system

* Change color channel from `3` to `1`: Because color is not much use in this case, and color informeation will bring 3 times train operation. If I reduce color channel,  so that I can train more data. 

* Add a off road distince value: From now on, this car is learn do not drive out of the lane. But, it don't know how to drive at the center. If add a value about off center distince, so the network will learn drive at the center of lane. The same plan was use at Nvidia system:![](https://ws2.sinaimg.cn/large/006tNc79ly1frs7hxd315j319204omz0.jpg)

  ![](https://ws3.sinaimg.cn/large/006tNc79ly1frs72rt06sj31940eywkw.jpg)

* Clip image higher or lower: Since the rode is not absolutely flat, some times car climbing or down the hill or just a jolt. So, clip the input image little bit up or down will make network more stronger. 

* Not only steering, more data: The collected data contains `steering`, ` throttle`, `brake` and `speed`, but I only use `steering` to train this network. use more data can tell network more about the road information. 

* Reinforcement learning: Make the car detect more information when it is driving or road. And feedback realtime. 

* More image data from bad case: add more bad case(drive back from the edge of a lane\ drive back to the center of a lane)

* Change `PID` code in `drive.py` make sure the car more "gentle". 

After Doing all this, I believe the car will successful pass the map 2.  