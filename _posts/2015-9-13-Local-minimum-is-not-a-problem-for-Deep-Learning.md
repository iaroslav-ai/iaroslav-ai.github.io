---
layout: post
title: Local minimum, not a problem for Deep Learning
---

A local minimum problem, associated with training of neural networks, is frequently viewed as a their serous drawback. In this post I argue why with proper initialization and popular choices of neuron types this problem does not affect much quality of local minimum.

Recently there appear many works that show that local minimum is not a problem. One very recent work [arxiv Vidal] shows that given a certain type of nerual net it can be detected when global optimum is achieved and for sufficient number of neurons it is always possible to achieve global optimum with gradient descent. Similarly, algorithms for shallow networks were given in literature earlier [Montreal guys] that allow to achieve global optimum, when the number of neurons is not fixed. Moreover, for a network with rectified linear activations, a spin glass physics model was used in [2014 AIII] together with intense experiments to show that as the size of neural network is growing the quality of local minimum for such networks improves.

In this post I concentrate on supervised learning. I show that with proper initialization of neural network and for neurons which satisfy a certain criterion it becomes harder to arrive at bad local minimum. I show that local minimum of arbitrary quality can be achieved already at a stage of supervised initialization of neural net, given that the number of neurons or layers can be selected arbitrary. To do so, I use some simple derivations and in general avoid any complicated math. I support my findings with some experimental evaluation. The code of these experiments can be found at my gihub repository.
All of this findings imply that given a large amount of data and sufficiently large network initialized properly, supervised learning problem can always be solved efficiently.

### General philosopy

Intro: here I outline the more abstract view that explain why enriching model makes learning with it easier.

Poor model: global optimum by enumeration

Fine model: many local minima, large difference in quality

Rich model: huge number of local minima, however difference is not large. Learning becomes easier, as less mistakes can be done for proper initialization.

### Definition of learning problem

Some data X, Y is given

L2 regression, but results can be generalized

### Shallow networks

Define matrix M on data by fixing neuron weights. Resulting problem is convex and can be solved to global optimality! Solution is an upper bound on global optimum. Example plot with random initialization. Boosting.

Extreme case: M is not degenerate and square. Solve a linear system! therefore is a global optimum (one of them, at least). It is hard to describe how hard the resulting neural net would overfit. 

Extreme case: mean of all values Y.

Perturbation theory: something in between two extreme cases. Adding new neuron improves initialized network objective: condition when this happens.

How uniform improvements are? Experimental results for random neurons!

All values are upper bounds on any local optimum, which bounds how "bad" local minimum could be. This shows how important initialization can be. 

### Additional layers to correct errors

General philosophy: deeper layers fix errors of previous layers.

Transition of outputs from previous layer to next layer without change: example for different neuron types. Simple for rectified linear.

Experiments with 3 neurons in a layer, arbitrary layers. Caution: information loss! Reference google net.

Add input data for every neruon: expected experimental results.

### Conclusion

As the number of neurons grows, learning becomes easier. Supervised pretraining allows to explicitly avoid bad local minima. Caution: overfitting, thus good for big data.
