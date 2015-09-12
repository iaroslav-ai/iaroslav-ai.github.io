---
layout: post
title: Local minimum&#58; not a problem for Deep Learning
---

A local minimum problem, associated with training of neural networks, is frequently viewed as a their serous drawback. In this post I argue why with proper initialization and popular choices of neuron types this problem does not affect much quality of local minimum.

Recently there appear many works that show that local minimum is not a problem. One very recent work [arxiv Vidal] shows that given a certain type of nerual net it can be detected when global optimum is achieved and for sufficient number of neurons it is always possible to achieve global optimum with gradient descent. Similarly, algorithms for shallow networks were given in literature earlier [Montreal guys] that allow to achieve global optimum, when the number of neurons is not fixed. Moreover, for a network with rectified linear activations, a spin glass physics model was used in [2014 AIII] together with intense experiments to show that as the size of neural network is growing the quality of local minimum for such networks improves.

In this post I concentrate on supervised learning. I show that with proper initialization of neural network and for neurons which satisfy a certain criterion it becomes harder to arrive at bad local minimum. I show that local minimum of arbitrary quality can be achieved already at a stage of supervised initialization of neural net, given that the number of neurons or layers can be selected arbitrary. To do so, I use some simple derivations and in general avoid any complicated math. I support my findings with some experimental evaluation. The code of these experiments can be found at my gihub repository.
All of this findings imply that given a large amount of data and sufficiently large network initialized properly, supervised learning problem can always be solved efficiently.

### General philosopy

First I would like to demonstrate on abstract level that when modelling power of the model is increased learning with such model becomes easier. Here I assume that such modelling power can be expressed in terms of number of parameters of a model and that increasing such number leads to increase of modelling power. 

Consider for example the task of fitting data points shown on the following figure:

Consider using a model with relatively small modelling power - a single Gaussian. In our case such model has only 2 parameters: mean and deviation. It is computationally tractable to perform a grid over the two parameters and thus find the best possible placement of Gaussian:

Thus, when number of parameters of the model is small, it might be computationally tractable to find the best set of parameters by simple enumeration.

Now consider that instead of one Gaussian our model consists of four Gaussians. 
As the number of parameters of our model increases, simple approach of enumeration rapidly becomes intractable, and thus methods like gradient descent are used, which frequently lead to local minimum. For example, the following model is example of local minimum achieved with gradient descent:

while the globally optimal arrangement of Gaussians fits the data perfectly:

Above examples show that local minimum can be much worse than global optimum.

As the number of Gaussians further increases, so does the number of local minima that gradient descent can converge to. However, as you get more Gaussians to fit your data, it becomes harder to fit the data badly.
If you would have some extra Gaussians at your disposal you could "fix" the bad areas of local minimum in above example like this:

Furthermore, you could think of some simple strategies on how to train a model given unlimited Gaussians such that you are guaranteed that you will not arrive at "bad" model. For example, you can add Gaussian to the place in your data where model and data disagree most; You repeat this procedure until difference between data and model is below some threshold, or when you cross validation error starts to grow. 

An extreme example of such strategies is when every point in your dataset has a separate Gaussian. For every such Gaussian you set its support to be small and its height to be equal to the value of the data, and voila! You don't even need training for your model, and you fit your data perfectly. Needles to say though, how bad such model would overfit.

Yeah, that all is great, but how does this relate to Deep Learning?

Good question, curious voice in my head! Lets first specify the learning problem.

### Definition of learning problem

Here I assume that some data is given in the form of matrix \\(X \in R^{n \times m}\\) (location of data points) and vector \\(Y \in R^{n} \\) (values to fit at points \\(X\\)). I want to fit a certain model \\( f(X, W) \to R^{n} \\) with parameters \\( W \in R^{k} \\) to my data \\(X,Y\\). I can formulate this in vector form as the following optimization problem:
$$ x+1\over\sqrt{1-x^2}\label{ref1} $$
$$ 
\min\limits\_{W \in R^{k}} || f(X,W) - Y ||\_2^2 \tag{someref}
$$

When model \\( \ref{ref1} \\) d \\( f(X,W) \eqref{ref1} \\) is defined to be the neural network, above optimization problem is solved by gradient descent and using L2 objective.

### Shallow networks: in between two extremes

We start with shallow networks, properties of which we will use to show some interesting things for deep networks.

Shallow network consists of a single layer of hidden neurons. Let the output of neuron for some input \\( x \in R^{m} \\) and its parameters \\( w \in R^{m} \\) be denoted as a function \\( g(x,w) \to R \\). Then the output of shallow network for some input \\(x \in R^{m} \\)is defined as a linear combination of \\(u\\) neuron outputs:

$$ f(x,W) = \sum\_{i \in 1 ... u} g\_i(x,w\_i) s\_i \quad (2) $$

where values of \\( w\_i \\) and \\(s\_i\\) are stored in the vector \\(W\\).

Due to the way the output of the shallow neural network is computed its training is a non-convex optimizaiton problem. Due to this property, training of neural network suffer from local minimum problem.

For convenience, let \\(G \in R^{n \times u} \\) denote separate outputs of neurons for every training point in our dataset.

Imagine that I fix the parameters of every of \\(m\\) neurons of the shallow network. Then the training optimization problem specifies to:

$$
\min\limits\_{s \in R^{u}} || G s - Y ||\_2^2 \quad (3)
$$

All of a sudden, above problem is convex and thus can always be solved to global optimality with gradient descent over \\(s \in R^{u}\\)! Moreover, its solution is an upper bound on global optimum of training problem. This means that if we are able to give some guarantes on solution of above problem, they will hold for the non-convex one (initialized at fixed neuron parameters). 

To give you a taste of quality of solutions with fixed neurons, here is example fit of some wiggly function with 10 random neurons:

There are two extreme cases for shallow neural networks with fixed neurons, which define how bad / good such networks can fit the data.

The worst fit depends on the type of neurons used. In general, you can always set vector \\(s\\) to be all zeros, and then the worst objective value of (3)

Extreme case: M is not degenerate and square. Solve a linear system! therefore is a global optimum (one of them, at least). It is hard to describe how hard the resulting neural net would overfit. 

Extreme case: mean of all values Y.

Perturbation theory: something in between two extreme cases. Adding new neuron improves initialized network objective: condition when this happens.

How uniform improvements are? Experimental results for random neurons!

All values are upper bounds on any local optimum, which bounds how "bad" local minimum could be. This shows how important initialization can be. 

### Additional layers do error correction

General philosophy: deeper layers fix errors of previous layers.

Transition of outputs from previous layer to next layer without change: example for different neuron types. Simple for rectified linear.

Experiments with 3 neurons in a layer, arbitrary layers. Caution: information loss! Reference google net.

Add input data for every neruon: expected experimental results.

### Conclusion

As the number of neurons grows, learning becomes easier. Supervised pretraining allows to explicitly avoid bad local minima. Caution: overfitting, thus good for big data.
