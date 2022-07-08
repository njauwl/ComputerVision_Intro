# ComputerVision_Intro
# 绪论
这里是计算机视觉技术的学习总结和简要介绍，包括内容图像处理，视觉应用如分类、分割，立体视觉等。
## 图像处理
参考书《Practical Python and OpenCV》2016 Authors: Adrian Rosebrock，通过基于python和opencv库的例子，简介了图像处理常用的基本方法，包括画线画圆，图像变换，图像统计，图像滤波，边缘检测，轮廓检测等基本内容。
blog Adrian Rosebrock 的技术博客 https://pyimagesearch.com/ 博客关于图像处理，和基于深度学习的图像基本应用。

## 视觉应用

大部分视觉应用都是基于深度学习技术的，可以学习的资料很多
### 深度学习
深度学习是多层的神经网络模型和传统的机器学习模型（如随机树，支持向量机等）训练过程一样，也和直线拟合也一样，目标都是通过训练数据训练模型，从而预测新的数据结果。训练过程是设计找到一个目标优化函数（模型输出结果与真实结果之间的误差），然后不断迭代优化这个目标函数，直到误差不断减小，最终得到最后模型训练结果。和直线（曲线）拟合一样，模型训练时也有过拟合问题，模型测试训练数据结果很多，但是预测测试数据结果很差，所以目标优化函数经常需要加入模型复杂度的权重，同奥卡姆剃刀定律类似，相同预测结果下，模型复杂度小的优于复杂度大的模型。

《Machine Learning》and 《Deep Learning》 -- Andrew Ng 的课程介绍了基础理论知识。
[《Neural Networks and Deep Learning》](http://neuralnetworksanddeeplearning.com/about.html),这本书详细介绍了神经网络模型设计和训练的细节原理。

blog https://machinelearningmastery.com/start-here/，博客有很多关于机器学习的例子。 

搭建深度学习模型和训练，设计优化目标函数，如何优化等问题，我们不用从scratch开始做起，有很多平台框架设计发布出来，经过竞争和选择，目前受欢迎的有Facebook公司的[PyTorch](https://pytorch.org/)，google的[tensorflow](https://tensorflow.google.cn/)，国内的百度[飞桨平台](https://www.paddlepaddle.org.cn/)，只需要像搭积木一样就可以设计完一个深度学习模型。另外各个平台的官网都有很好的教程入门学习深度学习相关知识。GPU显卡公司[NVIDIA官网](https://github.com/NVIDIA/DeepLearningExamples)也有很多基础的教程。
### Image classification
### Object Detection
### image segmentation
### image registration
### depth estimation


参考 <u>https://github.com/microsoft/computervision-recipes</u>

## 立体视觉
### 特征与匹配Feature Detection and Matching
### Autostitch
有了特征匹配点，就可以多张图片进行拼接，应用如全景相机
### Stereo
[6d-vision](http://www.6d-vision.com/),戴姆勒研究院的视觉感知方案

参考：康奈尔大学的课程Introduction to Computer Vision CS5670:https://www.cs.cornell.edu/courses/cs5670/








