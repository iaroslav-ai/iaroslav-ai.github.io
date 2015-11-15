---
layout: post
comments: true
title: 10000 layers each 4 neurons wide
---

In this post you will learn that 
**1.** You can initialize deep net with shallow net;
**2.** For such initialization, similar to [this paper](http://arxiv.org/pdf/1412.0233.pdf) it can be shown that loss will tend to zero, without spin glass model;
**3.** It does not make sense to study local minimum quality;
**4.** You don't want layers of your network to be too "thin".

**What kind of networks do you consider here?**

Deep fully connected nets with single outputs, where every layer has at least 4 neurons. 

**1. Shallow net initialization**

Algorithm is given in [this post](http://iaroslav-ai.github.io/Local-minimum-is-not-a-problem-for-Deep-Learning/).
More specifically, you can use part of neurons for data transfer, part for processing, part for carrying the result to the output. See figure below for explanation.

![Putting shallow network into deep one.](/images/10000layers/Shallow_To_Deep.svg)

Above figure shows described ininitialization scheme for deep fully connected networ. There \\(w\_i\\) denote weights of i-th neuron in shallow net, and \\(a\_i\\) it's output weight in final linear layer. 1s denote connections with weight one, and connections not drawn are set to zero. Essentially shallow net is put into deep net. 

Notice that this kind of initialization is similar in spirit to [this paper](http://arxiv.org/abs/1505.00387), where in a sense deep networks are made more "shallow" by allowing some of the neurons in layer pass the information through them without a change. 

**2. Loss tends to 0 when network becomes deeper**

Firstly, you can almost always find a neuron that gives different output values for different training points, see figure below. 

![Encoding every training point separately with a single number.](/images/10000layers/Projection_Example.svg)

This assumes that in your dataset you do not have training points which are exactly the same. If that is true, getting the same value of projection for two different points is almost impossible, at least in continuous space. Needless to say though, that such projection encoding would not generalize. 

Notice that in fully connected neural networks neurons perform projection + non - linearity on top. Thus you can use such neuron to uniquely identify every training point with a single number.

![Encoding every training point separately with a single number.](/images/10000layers/Shallow_Example.svg)

You can replace every training point with such encoding and train shallow neural net with neurons number equal to number of training points. This will necessary give you zero training loss, [see here](http://iaroslav-ai.github.io/Local-minimum-is-not-a-problem-for-Deep-Learning/) for details.

Then you can reshape shallow network into the deep one with the same 0 loss value. 
To do that, it suffices to have 1 neuron for training point encoding, one neuron for processing, two neurons to carry output, totalling 4 neurons on the layer; see figure below.

It can be shown for shallow network that as the number of its neruons grows,
the quality of local minimum as measured by training loss will always improve.
See [this post](http://iaroslav-ai.github.io/Local-minimum-is-not-a-problem-for-Deep-Learning/) for details.

**2.1 Results on real dataset**

Yes, here is the result for 10000 MNIST images: 10000 layers, each 4 neurons wide, gives training loss of 1e-5. As inputs to the network I provide the images, and as outputs the corresponding labels (numbers 0 - 9).

**3. You are interested in generalization**

Above results show that generalization does not necessary depends on local minimum quality. Thus without any relation to generalization studying local minimum quality is useless.

**4. What should be done to not overfit?**

In general, neuron does projection on the axis with some non linearity on top; If you necessary need m dimensions to generalize, you will loose some of these dimension with deep net n neurons wide, if m > n. Thus it would be not possible for such net to implement ground truth dependency between the inputs and outputs simply because information is lost, see one example in the figure below:

![Encoding every training point separately with a single number is not possible here.](/images/10000layers/Information_Loss.svg)

Here you cannot come up with projection on single vector (reduction to single dimension) such that you do not confuse training points and such that it would generalize to unseen data.

How to know such dimension? Take a look at [manifold learning](http://scikit-learn.org/stable/modules/manifold.html). 