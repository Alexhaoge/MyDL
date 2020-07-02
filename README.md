# MyDL
A lightweight deep learning framework entirely based on Java.  

![MyDL_logo.png](https://i.loli.net/2020/06/20/EV83WbDZXSqHzAe.png)

Note: **This project is still under construction and we welcome anyone who wants to participate in!**   

![Java CI with Maven](https://github.com/Alexhaoge/MyDL/workflows/Java%20CI%20with%20Maven/badge.svg)

## Introduction
MyDL (simply stands for "my deep learning") is a simple deep learning libary base on java. Its matrix operations are based on [EJML](https://github.com/lessthanoptimal/ejml)(Efficient Java Matrix Library) and its part of APIs design learns from Keras.
### Features
* **100% Java.** It is written in Java without any native methods and its dependency, EJML, is also fully written in Java. This provides a completely cross-platform compatibility. No matter what platform, as long as it support Java, you can use MyDL on it.
* **New implementation for tensor.** We build a set of tensor classes and operations based on EJML to establish a data type friendly to deep learning on Java.
* **A lightweight framework.** It is easy to deploy, all you need is a single JAR.
* **Minimalist design.** Components of this framework are easy and clear. Many API designs take reference from Keras, making it easy for users to get started.
### Compents implemented
* Model: abstract model, sequential  
* Layer: abstract layer 
    * Activation: abstract activation, ReLU, Sigmoid, Tanh, Softmax  
    * Fully-connected: Dense(for high dimension), Linear1D
    * Reshape layer
* Loss: abstract loss, SSE, MSE, binary cross-entropy, categorical cross-entropy  
* Optimizer: abstract optimizer, mini-batch GD/SGD
* Tensor:  abstract Tensor, Tensor1D, Tensor2D, Tensor3D
    * Tensor_size(record the shape of tensor)
* Dataset: APIs to load MNIST dataset(included in JAR)
* Utils: Some unclassified tools
    * Data(class for loading data into models, easy to convert)
    * IsAttribute(Determin if the object has such attribute)
## Getting Started
### Maven with GitHub Packages
MyDL has been published to GitHub Packages. If you have [configured Apache Maven for use with GitHub Packages](https://docs.github.com/en/packages/using-github-packages-with-your-projects-ecosystem/configuring-apache-maven-for-use-with-github-packages), all you need is to add this to pom.xml:  
```
<dependency>
  <groupId>io.github.alexhaoge.mydl</groupId>
  <artifactId>mydl</artifactId>
  <version>1.0-alpha</version>
</dependency>
```  

### JAR with depedencies
We also published a pre-release in this repository packing all the classes and dependencies in a single JAR.

## Support
The project website is still under construction.

**Javadoc**: https://apidoc.gitee.com/lda31415/MyDL
## What's next?
### Future works
* Although 100%-Java feature give MyDL a very good cross-platform compatibility, the performance of this framework sometimes is unsatisfying. In the future, we will find
* Add more components like convolution and pooling layers, regularizer, model checkpoint, visualization tools, etc. 
### Contribution
Due to limited time, this framework is actually far from complete, so the current release may be unstable. If you meet any problems, **please feel free to raise an issue or contact us** and we will fix it as soon as possible.  

The initial contributors are undergraduates in Nankai Univ and honesty their skills and time is limited, so we warmly welcome anyone who wants to contribute to this project. You can:  
* Raise an issue of suggestion for improvement.
* Create a pull request and provide description if you implement something new for this project.
* Contact [Alexhaoge](https://github.com/Alexhaoge) and become a collaborator of this repository for long term contribution.
## Acknowlegde
1. We wrote the initial version of this libary as a course project under the guidance of *Prof. Bing Xiang* in JAVA2760, School of Maths Sciences, NanKai University.
2. Special thanks to *[SaucerHi](https://github.com/Shiien)* who gave us inspiration on some key points in project desgin.
3. Part of APIs refer to *[joelnet](https://github.com/joelgrus/joelnet)* and *[Keras](https://github.com/keras-team/keras)*.
## License
MyDL is is GPL-3.0 licensed.
