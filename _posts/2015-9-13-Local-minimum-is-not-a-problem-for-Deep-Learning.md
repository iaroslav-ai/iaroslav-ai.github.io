---
layout: post
title: Local minimum&#58; not a problem for Deep Learning
---

A local minimum problem, associated with training of neural networks, is frequently viewed as a their serous drawback. In this post I argue why with proper initialization and popular choices of neuron types this problem does not affect much quality of local minimum.

Recently there appeared many works that show that local minimum becomes less of a problem for deep learning when number of neurons grows. One [very recent work](http://arxiv.org/pdf/1506.07540.pdf) shows that given a certain type of nerual net it can be detected when global optimum is achieved and for sufficient number of neurons it is always possible to achieve global optimum with gradient descent. [Moreover](http://arxiv.org/pdf/1412.0233.pdf), for a network with rectified linear activations, a spin glass physics model was used together with intense experiments to show that as the size of neural network is growing the quality of local minimum for such networks improves.

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

In the above figure there are four Gaussians with the same mean, that is why there appear to be only one red dot.

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

All of a sudden, learning problem becomes convex! This means that it can always be solved to global optimality with gradient descent over \\(s \in R^{u}\\). Moreover, its solution is always an upper bound on global optimum of training problem. This means that if we are able to give some guarantes on solution of above problem, they will hold for the non-convex one (initialized at fixed neuron parameters). 

There are two extreme cases for shallow neural networks with fixed neurons, which define how bad / good such networks can fit the data.

In general, you can always set vector \\(s\\) to be all zeros, and then the worst objective value of \\(\eqref{eq:lin-fxnn}\\) would be the sum of squared values of data points. Furthermore, if bias can be additionally added to objecitve \\(\eqref{eq:lin-fxnn}\\), the worst objective becomes sum of squared deviations of data values from the mean.

On the other side, consider a case when the number of neurons is equal to the number of data points. Then \\(G\\) becomes a square matrix. Given that determinant of \\(G\\) is non zero (depends on type of neurons used), solution to \\(\eqref{eq:lin-fxnn}\\) can be found by simply solving system of linear equations \\(G s = Y\\). This in turn means that the value of objective for solution \\(s\\) would be zero. As this is an upper bound on non-convex problem \\(\eqref{eq:main}\\), and as its objective always greater equal zero, this implies that \\(s\\) together with fixed neuron parameters is a globally optimal solution to \\(\eqref{eq:main}\\). Again, such neural network would overfit the data beyond anything imaginable (recall the same scenario in the previous section).

For the number of neurons in between 0 and \\(|X|\\) (size of data) the values would be distributed between \\( ||Y||\_2^2 \\) and 0 correspondingly. It is however not clear how "even" these values are distributed with respect to the number of neurons used.

First, I show that adding extra neurons with fixed parameters always improves objective, given that randomly initialized neurons satisfy a certain criterion (which happens with probability almost 1), outlined below.

**Theorem 1.** Given \\(u\\) neurons, adding one extra neuron to network improves objective \\(\eqref{eq:lin-fxnn}\\) if and only if it does not hold that

$$ 
g'^T Gs = g'^T Y 
$$

where \\(g'\\) are the output values of added neuron for every training point.

Observe that above equation defines some linear subspace of \\(R^{n}\\). Given some \\(g'\\)  **uniformly sampled** from \\(R^{n}\\) probability of "hitting" such subspace is almost zero. In practice outputs are not uniform, and belong to some non-linear subspace defined by all possible outputs of neuron for training points. One can argue that due to the non-linearity of neurons it would still be "hard" to hit linear space. In order to keep things simple, I verify such claim experimentally with some artificial data. Python code that can be used to reproduce experiments is in [my gihub repository](https://github.com/iaroslav-ai/nn-local-minimum). These experimental results are for simple artificial problem of fitting the 2d function with different number of neurons.

First I try neurons with tanh nonlinearity. As hyperbolic tanhent is non-linear almost everywhere, I expect that due to this property the space where output \\(g \in R^{n}\\) lives is also non-linear almost everywhere, and thus "hitting" its intersection with linear subspace is almost impossible. Indeed, for around 10000 extensions of neural network its objective did not improve only once (and it might have happened due to the numerical errors). 

A different story is for ReLU nonlinearity, which is linear almost everywhere except for 0. For 15000 added neurons 5000 did not improve the objective function. This is due to the sampled weights being too large, which results in many of 5000 neurons either returning 0 or being completely linear for all data points. In order to avoid that, I multiplied sampled weights by 0.01, which reduced the number of non-improving neurons to 1000. It appears that due to ReLU neurons being "more linear", it is easier to "hit" the linear subspace. Nevertheless, for proper sampling of weights such probability is small.

<<<<<<< HEAD
Above results suggest that network can be extended by some random neuron such that it would yield improvement of objective with some large probability.

It is however not clear how much of improvement extra neurons are causing. Average values for different number of neurons are shown below for tanh non-linearity. One hundred different random instances of function were considered to make results more robust.

![Result of extension by random neuron with tanh non-linearity.](/images/localminimum/experiments_1.png)

All of the above values of objective function are upper bounds on any local minimum of \\(
eqref{eq:main}\\), which bounds how "bad" local minimum could be. This confirms our theoretical derivations for shallow networks and demonstrates that local minimum quality, as measured by objective function, rapidly improves with increase of number of neurons. In order to see better how objective behaves for larger amount of neurons, here is the same plot on logarithmic scale:

![Result of extension by random neuron with tanh  non-linearity on logarithmic scale.](/images/localminimum/experiments_2.png)

As expected, when the number of neurons approach size of dataset, objective rapidly approaches zero.

Here are similar results for rectified linear activation, where only neurons that improve objective were used for network extension:

![Result of extension by random neuron with rectified linear non-linearity on logarithmic scale.](/images/localminimum/experiments_3.png)

One can think of different ways on how to improve initialization of network so that it results in better objective values. For example, multiple neuron "candidates" can be sampled and the one is selected which yields the best objective improvement. What I however found more efficient is to make some random changes to the network weights and save the change if it leads to objective improvement. With larger number of changes I get better results, summarized in the following plot for tanh non-linearity:

![Result of extension by random neuron with random permutation of network and with tanh non-linearity on logarithmic scale.](/images/localminimum/experiments_4.png)

Similar results are obtained for rectified linear activation:

![Result of extension by random neuron with random permutation of network and with rectified non-linearity on logarithmic scale.](/images/localminimum/experiments_5.png)

This shows that with more advanced random pretraining procedures it is possible to improve initialization, which provides a better guarantee on local minimum. Such procedure would be used as well in the next section.
=======
All of the above values are upper bounds on any local minimum of \\( \eqref{eq:main} \\), which bounds how "bad" local minimum could be. This demonstrates that local minimum quality, as measured by objective function, improves with increase of number of neurons. Furthermore is shows that supervised pretraining of neural net can already achieve good results.
>>>>>>> origin/master

### Extra layers for error correction

Imagine that you trained your shallow neural net with fixed neuron parameters, but you are not satisfied with objective value you are getting. Instead of adding additional neurons or optimizing over the neuron parameters, you want to add extra layer, in hope that it would "correct the errors" of previous layer.

How do you correct errors? First you make sure that you do not create any extra ones! For neural net this means that extending network by one more layer should be done such that it does not increase the objective value.

Firstly, here I still assume that all of the neuron parameters are fixed. This allows to write extension of a network by one layer very simple by setting \\(X\\) equal to \\(G\\), and setting as \\(G\\) the outputs of neurons of new layer. Also for simplicity I assume that number of neurons is similar on each layer of (now deep) network. 

So how do you not mess up with new layer? Let \\(s \in R^{u}\\) denote weights of optimal linear combination for outputs of previous layer now denoted as \\(X\\). By adding neuron with linear activation function \\(g(x) = x^T s\\) to the new layer we necessary preserve the objective value, as weight for such neuron is necessary selected in the best way due to convexity of \\(\eqref{eq:main}\\).

Above trick guarantees that extra layers necessary do not degrade the value of objective function. This together with result in previous section implies that adding extra layer should only improve the value of objective function.

I start my experiments with only one neuron with rectified linear activation on the layer + one linear neuron. Every time I add one extra neuron, I perform 100 iterations of network permutation to improve objective. Ten random instances were considered. Results look as follows:

![Result of extension by a layer with one rectified linear neuron and one with linear activation.](/images/localminimum/experiments_6.png)

It appears that extending network by extra layers allows to steadily decrease the objective value, even though for this case I cannot state bound on number of neurons when objective would turn to zero. Wider networks should allow to perform more corrections and thus with larger number of neurons in layers objective should improve faster. This is confirmed by following results with 5 ReLU neurons:

![Result of extension by a layer with 5 rectified linear neurons and one with linear activation.](/images/localminimum/experiments_7.png)

Results are similar to ones with one ReLU, except that here 10 times less layers were used.

It appears that already with described supervised pretraining for deep network, objective value of arbitrary quality can be achieved, when depth is not fixed. It is not clear however for which number of layers objective would turn to zero and if there might exist some example of data for which objective would never turn to zero. However, similar to shallow networks, such guarantee can be given to some class of deep topologies.

One of the issues I expected from deep nets is the following: it might happen that information that is passed through the layers is scrambled to such extend by all of the processing that further error correction yields very small to none improvements of objective function. To avoid this, I connect every additional layer to input data, so that it always has access to "unscrambled" information. Here are results with 5 ReLU neurons:

![Result of extension by a layer with 5 rectified linear neurons and one with linear activation, where every layer is connected to input.](/images/localminimum/experiments_8.png)

Indeed, compared to network with no connection of deeper layers to input, network objective improve much faster. 

What is more interesting is that such deep network can be constructed from shallow network as follows: let l be number of layers, and t be number of neurons in the layer. First, pretrain shallow network with \\(l n\\) neurons as in previous section. Then, select first \\(l\\) neurons of shallow net with output weights \\(s\\), and add them as first layer of deep network. Add the next layer; To the deepest layer add neuron with linear activation which corresponds to \\(f(x) = s^T x\\) and next \\(l\\) neurons  of shallow net and set \\(s\\) to their output weights. Add one more layer. To the deepest layer add linear neuron which has weights \\(s\\) for ReLU neurons on previous layers and 1 for the linear activation neuron. Recursively apply the above procedure, until all of neurons of a shallow net are not used.

By construction, above network should have objective value similar to the shallow one. This means that all of the analysis in previous section applies to such network!

It might not be always practical however to always connect layers to input, as input size can be large, which would increase greatly the number of parameters in network and thus make it more prone to overfitting. However it [was shown](http://arxiv.org/abs/1409.4842) to be helpful to connect some deeper layers to earlier layers, which allows to obtain "less scrambled" version of the input, while avoiding explosion of the parameters. Additionally, instead of connecting to the input, deeper layers can connect to the first layer after the input, which can be made to have smaller number of outputs compared to size of input.

### Conclusion

Indeed, as the size of neural network grows, learning becomes less sensitive to the local minimum problem. This was demonstrated for both shallow and deep neural networks using a supervised pretraining procedure, which allows to obtain any desired objective value, given that number of neurons / layers is not fixed, and which provides upper bound on local minimum objective value. 

<<<<<<< HEAD
Computational power accessible to regular user grows exponentially by Moore's law. This means that larger models can be used for "machine learning in the wild", and together with presented results this means that deep learning will continue being successful in the foreseeable future. 
=======
Computational power accessible to regular user grows exponentially by Moore's law. This means that larger models can be used for "machine learning in the wild", and as local minimum becomes less of a problem for larger networks this means that training of deep network should only become easier, which would contribute to futher success of deep learning. 
>>>>>>> origin/master

### Proof of Theorem 1

The objective with one extra neuron is

$$
|| \left(\begin{array}{cc}
  g' & G 
\end{array}
\right) \left(
\begin{array}{cc}
  s' \\\\
  s 
\end{array}
\right) - Y||\_2^2 = 0
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
