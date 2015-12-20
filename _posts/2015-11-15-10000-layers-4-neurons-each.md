---
layout: post
comments: true
title: 10000 layers each 4 neurons wide
---

In this post you will learn that 
**1.** You can initialize deep net with shallow net;
**2.** For such initialization, similar to [this paper](http://arxiv.org/pdf/1412.0233.pdf) it can be shown that loss will tend to zero as network grows, without spin glass model;
**3.** It does not make sense to study local minimum quality alone;
**4.** You don't want layers of your network to be too "thin".

**What kind of networks do you consider here?**

Deep fully connected nets with single linear output neuron, where every layer has at least 4 neurons. Neurons of such network need not to be ReLU ones.

**1. Shallow net initialization**

Algorithm is given in [this post](http://iaroslav-ai.github.io/Local-minimum-is-not-a-problem-for-Deep-Learning/).
More specifically, you can use part of neurons for data transfer, part for processing, part for carrying the result to the output. See figure below for explanation.

![Putting shallow network into deep one.](/images/10000layers/Shallow_To_Deep.svg)

Above figure shows described ininitialization scheme for deep fully connected network. Here \\(w\_i\\) denote weights of i-th neuron in initialization shallow net, and \\(a\_i\\) it's output weight in final linear layer. 1s denote connections with weight selected such that input is passed to the output without change (coming up with such weights is an easy exercise for a reader), and connections not drawn are set to zero. Essentially shallow net is put into deep net. 

Notice that this kind of initialization is similar in spirit to [this paper](http://arxiv.org/abs/1505.00387), where in a sense deep networks are made more "shallow" by allowing some of the neurons in layer pass the information through them without a change. 

**2. Loss tends to 0 when network becomes deeper**

Firstly, you can almost always find a vector such that projection of training points gives unique values for every training point, see figure below. 

![Encoding every training point separately with a single number.](/images/10000layers/Projection_Example.svg)

This assumes that in your dataset you do not have training points which are exactly the same. If that is true, getting the same value of projection for two different points is almost impossible, at least in continuous space. This can be proven easily, if you assume that your data and projection axis is uniformly distributed, which I leave to the reader. 

Notice that in fully connected neural networks neurons perform projection + non - linearity on top. Thus you can use such neuron to uniquely identify every training point with a single number. Needless to say though, that such projection would not allow to generalize. 

![Encoding every training point separately with a single number.](/images/10000layers/Shallow_Example.svg)

You can replace every training point with such encoding and train shallow neural net with neurons whose number is equal to the number of training points. This will necessary give you zero training loss, [see here](http://iaroslav-ai.github.io/Local-minimum-is-not-a-problem-for-Deep-Learning/) for details.

Then you can reshape shallow network into the deep one with the same 0 loss value. 
To do that, it suffices to have 1 neuron for training point encoding (you can bias encoding values to be positive), one neuron for processing, two neurons to carry output, totalling 4 neurons on the layer; see figure below.

It can be shown for shallow network that as the number of its neruons grows,
the training loss will always improve.
See [this post](http://iaroslav-ai.github.io/Local-minimum-is-not-a-problem-for-Deep-Learning/) for details.

**2.1 Results on real dataset**

Result for 10000 MNIST images: supervised pretraining for 10000 layers network gives MSE loss of around 1e-3. In contrast to that, test MSE is around 10. As inputs to the network I provide the images, and as outputs the corresponding labels (numbers 0 - 9). You can see the code of experiment [here](https://github.com/iaroslav-ai/10000_layers_net).

**3. You are interested in generalization**

Above shows that generalization does not necessary depends on local minimum quality. Thus without any relation to generalization studying local minimum quality is useless.

**4. Layers can be too thin**

Typically, in neural networks neurons do projection on the vector with some non linearity on top. For a single layer, such neurons can be seen as axes of the space in which inputs to the layer are projected. If you necessary need m dimensions to generalize, you will loose some of these dimension with deep net n neurons wide, if m > n. Thus it would be not possible for such net to implement ground truth dependency between the inputs and outputs simply because information is lost, see one example in the figure below:

![Encoding every training point separately with a single number is not possible here.](/images/10000layers/Information_Loss.svg)

In above figure instances of classes are sampled from rectangles of respective color. Imagine you project all points on both rectangles on a single axis; In this case for every point on red square, there is a point in the blue square, that will have the same projection value for the axis, and thus for single projection value 2 different labels are assigned. Thus given only value of such projection, the best you can do is randomly guess one of two classes. 

Notice that due to the set of training points being finite set, probability of projection values to collide is usually almost zero and thus you do not experience this effect, which allows for your training loss to go to zero.

Above example demonstrates that number of neurons in the layer should be selected at least as the number of dimensions, needed to represent the data such that the information is not lost \[too much\].

How to know this number? One idea is to look at [manifold learning](http://scikit-learn.org/stable/modules/manifold.html). 
