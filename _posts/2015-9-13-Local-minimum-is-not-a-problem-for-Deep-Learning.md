---
layout: post
title: Local minimum&#58; not a problem for Deep Learning
---

A local minimum problem, associated with training of neural networks, is frequently viewed as a their serous drawback. In this post I argue why with proper initialization and popular choices of neuron types this problem does not affect much quality of local minimum.

Recently there appear many works that show that local minimum is not a problem. One [very recent work](http://arxiv.org/pdf/1506.07540.pdf) shows that given a certain type of nerual net it can be detected when global optimum is achieved and for sufficient number of neurons it is always possible to achieve global optimum with gradient descent. Similarly, algorithms for shallow networks [were given in literature](http://www.iro.umontreal.ca/~lisa/pointeurs/convex_nnet_nips2005.pdf) earlier that allow to achieve global optimum, when the number of neurons is not fixed. [Moreover](http://arxiv.org/pdf/1412.0233.pdf), for a network with rectified linear activations, a spin glass physics model was used together with intense experiments to show that as the size of neural network is growing the quality of local minimum for such networks improves.

In this post I concentrate on supervised learning. I show that with proper initialization of neural network and for neurons which satisfy a certain criterion it becomes harder to arrive at bad local minimum. I show that local minimum of arbitrary quality can be achieved already at a stage of supervised initialization of neural net, given that the number of neurons or layers can be selected arbitrary. To do so, I use some simple derivations and in general avoid any complicated math. I support my findings with some experimental evaluation. The code of these experiments can be found at [my gihub repository](https://github.com/iaroslav-ai/nn-local-minimum).
All of this findings imply that given a large amount of data and sufficiently large network initialized properly, supervised learning problem can always be solved efficiently.

### General approach

First I would like to demonstrate on abstract level that when modelling power of the model is increased learning with such model becomes easier. Here I assume that such modelling power can be expressed in terms of number of parameters of a model and that increasing such number leads to increase of modelling power. 

Consider for example the task of fitting data points shown on the following figure:

![Example data.](/images/localminimum/figure_0.png)

Consider using a model with relatively small modelling power - a single Gaussian. In our case such model has only 2 parameters: mean and deviation. It is computationally tractable to perform a grid over the two parameters and thus find the best possible placement of Gaussian:

![Example local minimum for data fitting with 1 Gaussian.](/images/localminimum/figure_0.5.png)

In above figure and in the next ones, red dot denotes mean of the Gaussian.

Now consider that instead of one Gaussian our model consists of four Gaussians. 
As the number of parameters of our model increases, simple approach of enumeration rapidly becomes intractable, and thus methods like gradient descent are used, which frequently lead to local minimum. For example, the following model is example of local minimum achieved with gradient descent:

![Example local minimum for data fitting with 4 Gaussians.](/images/localminimum/figure_1.png)

while the globally optimal arrangement of four Gaussians fits the data perfectly:

![Global minimum for data fitting with 4 Gaussians.](/images/localminimum/figure_2.png)

In the above figure there are three Gaussians with the same mean, that is why there appear to be only one red dot.

Above examples show that local minimum can be much worse than global optimum.

As the number of Gaussians further increases, so does the number of local minima that gradient descent can converge to. However, as you get more Gaussians to fit your data, it becomes harder to fit the data badly.
If you would have some extra Gaussians at your disposal you could "fix" the bad areas of local minimum. For example, adding extra Gaussians to the initialization that lead to bad local minimum yields:

![Example fitting with larger number of Gaussians (7).](/images/localminimum/figure_3.png)

which is in fact very close to the global optimum.

Furthermore, you could think of some simple strategies on how to train a model given unlimited Gaussians such that you are guaranteed that you will not arrive at "bad" model. For example, you can add Gaussian to the place in your data where model and data disagree most; You repeat this procedure until difference between data and model is below some threshold, or when you cross validation error starts to grow. 

An extreme example of such strategies is when every point in your dataset has a separate Gaussian. For every such Gaussian you set its deviation to be small and its height to be equal to the value of the data, and voila! You don't even need training for your model, and you fit your data perfectly. Needles to say though, how bad such model would overfit.

### Definition of learning problem

I assume that some data is given in the form of matrix \\(X \in R^{n \times m}\\) (location of data points) and vector \\(Y \in R^{n} \\) (values to fit at points \\(X\\)). I want to fit a certain neural net \\( f(X, W) \to R^{n} \\) with parameters \\( W \in R^{k} \\) to my data \\(X,Y\\). I can formulate this in vector form as the following optimization problem:

$$ \min\limits\_{W \in R^{k}} || f(X,W) - Y ||\_2^2\label{eq:main}$$

### Shallow networks: in between two extremes

We start with shallow networks, properties of which we will use to show some interesting things for deep networks.

Shallow network consists of a single layer of hidden neurons. Let the output of neuron for some input \\( x \in R^{m} \\) and its parameters \\( w \in R^{m} \\) be denoted as a function \\( g(x,w) \to R \\). I define the output of shallow network for some input \\(x \in R^{m} \\) to be a linear combination of \\(u\\) neuron outputs:

$$ f(x,W) = \sum\_{i \in 1 ... u} g\_i(x,w\_i) s\_i $$

where values of \\( w\_i \\) and \\(s\_i\\) are stored in the vector \\(W\\).

Due to the way the output of the shallow neural network is computed its training is a non-convex optimizaiton problem. Due to this property, training of neural network suffer from local minimum problem.

For convenience, let \\(G \in R^{n \times u} \\) denote separate outputs of neurons for every training point in our dataset.

Imagine that I fix the parameters of every of \\(m\\) neurons of the shallow network. Then the training optimization problem specifies to:

$$
\min\limits\_{s \in R^{u}} || G s - Y ||\_2^2 \label{eq:lin-fxnn}
$$

All of a sudden, learning probelm becomes convex! This means that it can always be solved to global optimality with gradient descent over \\(s \in R^{u}\\). Moreover, its solution is always an upper bound on global optimum of training problem. This means that if we are able to give some guarantes on solution of above problem, they will hold for the non-convex one (initialized at fixed neuron parameters). 

To give you a taste of quality of solutions with fixed neurons, here is example fit of some wiggly function with 10 random neurons:

There are two extreme cases for shallow neural networks with fixed neurons, which define how bad / good such networks can fit the data.

Consider the case when we allow only one neuron. The worst fit depends on the type of neurons used. In general, you can always set vector \\(s\\) to be all zeros, and then the worst objective value of \\(\eqref{eq:lin-fxnn}\\) would be the sum of squared values of data points. Furthermore, if bias can be additionally added to objecitve \\(\eqref{eq:lin-fxnn}\\), the worst objective becomes sum of squared deviations of data values from the mean.

On the other side, consider a case when the number of neurons is equal to the number of data points. Then \\(G\\) becomes a square matrix. Given that determinant of \\(G\\) is non zero (depends on type of neurons used), solution to \\(\eqref{eq:lin-fxnn}\\) can be found by simply solving system of linear equations \\(G s = Y\\). This in turn means that the value of objective for solution \\(s\\) would be zero. As this is an upper bound on non-convex problem \\(\eqref{eq:main}\\), and as its objective always greater equal zero, this implies that \\(s\\) together with fixed neuron parameters is a globally optimal solution to \\(\eqref{eq:main}\\). Again, such neural network would overfit the data beyond anything imaginable (recall the same scenario in the previous section).

For the number of neurons in between 0 and \\(|X|\\) (size of data) the values would be distributed between \\( ||Y||\_2^2 \\) and 0 correspondingly. It is however not clear how "even" these values are distributed with respect to the number of neurons used.

First, I show that adding extra neurons with fixed parameters always improves objective, given that randomly initialized neurons satisfy a certain criterion (which happens with probability almost 1), outlined below.

Theorem 1. Given \\(u\\) neurons, adding one extra neuron to network improves objective \\(\eqref{eq:lin-fxnn}\\) if and only if 

$$ 
g'^T Gs = g'^T Y 
$$

where \\(g'\\) are the output values of added neuron for every training point.

Observe that above equation defines some linear subspace of \\(R^{n}\\). Given some \\(g'\\)  **uniformly sampled** from \\(R^{n}\\) probability of "hitting" such subspace is almost zero. In practice outputs are not uniform, however due to the non-linearity of neurons it is still "hard" to hit linear space.

All of the above reasoning means that extending network by additional neurons with non-linear outputs would almost always yield improvement of objective.

It is however not clear how much of improvement extra neurons are causing. To verify the claims I make I present experimental results for simple artificial problem of fitting the 2d function with different number of neurons. Average values for different number of neurons are shown below. One hundred different random instances of function were considered to make results more robust.

All of the above values are upper bounds on any local minimum of \\( \eqref{eq:main} \\), which bounds how "bad" local minimum could be. This demonstrates that local minimum quality, as measured by objective function, improves with increase of number of neurons. Furthermore is shows that supervised pretraining of neural net can already achieve good results.

### Going deep for error correction

Imagine that you trained your shallow neural net with fixed neuron parameters, but you are not satisfied with objective value you are getting. Instead of adding additional neurons or optimizing over the neuron parameters, you want to add extra layer, in hope that it would "correct the errors" of previous layer.

How do you correct errors? First you make sure that you do not create any extra ones! For neural net this means that extending network by one more layer should be done such that it does not increase the objective value.

Firstly, here I still assume that all of the neuron parameters are fixed. This allows to write extension of a network by one layer very simple by setting \\(X\\) equal to \\(G\\), and setting as \\(G\\) the outputs of neurons of new layer. Also for simplicity I assume that number of neurons is similar on each layer of (now deep) network.

So how do you not mess up with new layer? Let \\(s \in R^{u}\\) denote weights of optimal linear combination for outputs of previous layer now denoted as \\(X\\). By adding neuron with linear activation function \\(x^T s\\) to the new layer we necessary preserve the objective value, as weight for such neuron is necessary selected in the best way due to convexity of \\(\eqref{eq:main}\\).

Above trick guarantees that extra layers necessary do not degrade the value of objective function. This together with result in previous section implies that adding extra layer should only improve the value of objective function.

There is however one caveat here: it might happen that information that is passed through the layers is scrambled to such extend by all of the processing that further error correction yields very small to none improvements of objective function. To avoid this, I connect every additional layer to input data.

Below the experimental results are shown for deep network with variable number of layers. Number of neurons was fixed to be 5 + one extra linear neuron.

Above results show that already with described supervised pretraining for deep network, objective value of arbitrary quality can be achieved, when depth is not fixed. Such objective value is an upper bound for local minimum achieved with gradient descent.

### Conclusion

Indeed, as the size of neural network grows, learning becomes less susceptible to the local minimum problem. This was demonstrated for both shallow and deep neural networks using a supervised pretraining procedure, which allows to obtain any desired objective value, given that number of neurons / layers is not fixed, and which provides upper bound on local minimum objective value. 

Computational power accessible to regular user grows exponentially by Moore's law. This means that larger models can be used for "machine learning in the wild", and together with my findings this means that deep learning will continue being successful in the foreseeable future. 

### Proof of Theorem 1

The objective with one extra neuron is

$$
|| \left(\begin{array}{cc}
  g' & G 
\end{array}
\right) s - Y||\_2^2 = 0
$$

Setting gradient of L2 regression objective yields:

$$
\left(
\begin{array}{cc}
  g' & G 
\end{array}
\right)^T (
\left(
\begin{array}{cc}
  g' & G 
\end{array}
\right)
\left(
\begin{array}{cc}
  s' \\\\
  s 
\end{array}
\right)
-Y) = 0
$$

This alternatively can be written as

$$
\left(
\begin{array}{cc}
  g'^T g' & g'^T G \\\\
  G^T g'  & G^T G
\end{array}
\right) 
\left(
\begin{array}{cc}
  s' \\\\
  s 
\end{array}
\right) = 
\left(
\begin{array}{cc}
  g'^T Y  \\\\
  G^T Y 
\end{array}
\right) 
$$

If neuron being added does not allow to improve objective value, then setting s' to zero and s to weights before the network was extended should satisfy above system of equations due to the convexity of optimization problem. For selected value of s and s' it holds that the only equation that does not necessary holds is

$$
g'^T Gs = g'^T Y 
$$

Thus, if above equation does not hold, then gradient of objective for s' = 0 is non zero, and thus objective can be improved at least a little bit.
