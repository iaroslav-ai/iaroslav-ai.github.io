---
layout: post
comments: true
title: 10000 layers deep, 4 neurons wide network, and what you can learn from it
---

Article is in construction still; I put it temporary untill I find some decent offline .md file viewer.

I organize this post compactly in form of Q&A.

**What will I learn from this post?**
1. You can initialize deep net with shallow net. 
2. As network becomes deeper, with initialization 1) network loss on training set tends to 0.
3. For such initialization, conclusions similar to [this paper](http://arxiv.org/pdf/1412.0233.pdf) can be made without spin glass model.
4. It does not make sense to study local minimum quality.
5. You dont want your layers to be too "thin".

**What kind of networks do you consider here?**

Deep fully connected nets with single outputs, where every layer has at least 4 neurons. 
Arguments can be generalized for multiple outputs as well.

**How can I initialize deep net with shallow net?**

Algorithm is given in [this post](http://iaroslav-ai.github.io/Local-minimum-is-not-a-problem-for-Deep-Learning/).
More specifically, you can use part of neurons for data transfer, part for processing, part for carrying the result to the output. See figure below for explanation.

**So why does loss tends to 0 when network becomes deeper?**

Firstly, you can almost always find a neuron that gives different output values for different training points, see figure below. 

You can use such neuron to uniquely identify every training point with a single number.

You can replace every training point with such encoding and train shallow neural net with neurons number equal to number of training points. This will necessary give you zero training loss, [see here](http://iaroslav-ai.github.io/Local-minimum-is-not-a-problem-for-Deep-Learning/) for details.

Then you can reshape shallow network into the deep one with the same 0 loss value. 
To do that, it suffices to have 1 neuron for training point encoding, one neuron for processing, two neurons to carry output, totalling 4 neurons on the layer; see figure below.

**Can you really encode every image with single number?**

Theoretically you can. But this of course will not generalize.

**Did you try that on any real dataset?**

Yes, here is the result for 10000 MNIST images: 10000 layers, each 4 neurons wide, gives training loss of 1e-5, for 3 class prediction scheme, see figure below.

**How hard does this approach overfit?**

Very hard. Nevertheless, this shows that ultimatelly, as you increase number of layers and use pretraining procedure, training loss tends to zero.

This shows that generalization does not necessary depends on local minimum quality. Thus without any relation to generalization studying local minimum quality is useless.

**What should be done to not overfit?**

In general, neuron does projection on the axis with some non linearity on top; If you necessary need m dimensions to generalize, you will loose some of these dimension with deep net n neurons wide, if m > n. Thus it would be not possible for such net to implement ground truth dependency between the inputs and outputs simply because information is lost. 

How to know such dimension? Take a look at [manifold learning](http://scikit-learn.org/stable/modules/manifold.html). 

