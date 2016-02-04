---
layout: post
comments: true
title: Train Neural Networks with Neural Networks
---

I explore some simple approaches for automatic learing of the training algorithms for nerual netwoks (as well as for other predicitve models). I show why this is a hard problem for standard approaches (in particular for deep learning) and what new insights can be learned while solving it. I continue with more dedicated approaches in part 2 of this post.

**The main idea**

... is summarized in the figure below:

![The main idea is to learn the training algorithm.](/images/train-nn-with-nn-part1/Main_Idea.svg)

Main advances in deep learning come from increase of the size of network, [which reduces the local minimum problem](http://arxiv.org/pdf/1412.0233.pdf), and effective adaptation of the network architecture for the problem at hand (e.g. [convolutional NN](http://www.nature.com/nature/journal/v521/n7553/full/nature14539.html) for computer vision applications). This shows that training algorithms which scale well (allow to train complex models) and which can efficiently adapt predictive model to the learning problem (e.g. select the architecture of NN well) are of great practical interest.

In this post I try to automatize search for new learning algorithms, where I cast the problem of coming up with a new training algorithm as a learning problem.

**Formulation as a learning problem**

I learn a training algorithm which takes as input a dataset (set of inputs and desired outputs), and predicts model parameters. To learn such algorithm, I artificially generate a dataset of datasets for training of such algorithm. More specifically, I generate a number of random single layer neural network with fixed number of neurons (ReLU nonlinearity) and for random inputs I sample their outputs; A pair of samples / otputs and network weights is one item in the training dataset. 

It would be best if I would use real data, but it would take way too much time to scrape all possible datasets on the internet. Furthermore, if one assumes that "real" data generating models are subset of uniformly generated ones, then algorithm which works nicely for artificial dataset will also work good for "real" data.

**RNN trained to achieve good generalization**

During training I feed pairs of inputs / outputs one by one to RNN, and after single pass over the dataset I reshape outputs of RNN into neural network weights, where NN has a single hidden layer and fixed number of neurons. Then I compute the loss of such neural network on separate dataset; I minimize mean loss over all predicted networks by backpropagation.

This way I directly train RNN to predict models which will have good generalization. The approach is summarized in the figure below.

![Algorithm trained to predict models with good generalization.](/images/train-nn-with-nn-part1/RNN_Generalization.svg)

**Results with RNN**

Results were generated using [this](https://github.com/iaroslav-ai/train_dnn_with_dnn/) Theano implementation; Surprisingly, I get results which are not much better than random guessing. Lets see why this is happening.

**Why RNN fails**

... is because single layer network representation is not unique. For example, order of neurons can be permuted in the matrix of neuron weights, which will not change network outputs, but will give different network representation, see figure below:

![Non uniqueness of the neural netowrk.](/images/train-nn-with-nn-part1/NN_not_unique.svg)

In above figure every row of X corresponds to a single training point.

Above means that for a input dataset there are many output weights, all of which are "correct". This would simply mean that algorithm would try to predict something close to the mean of outputs, which is likely incorrect:

![Mean of correct outputs is not necessary correct.](/images/train-nn-with-nn-part1/RNN_IncorrectMean.svg)

Thus one would expect that for models with unique representation RNN or other models will perform better.

**Predicting models with SVM**

Now I try a simplified setting where I train SVM to predict linear model (3 weights + bias). I flatten the dataset into feature vector, and assign linear model weights as outputs. The average testing coefficient of determination that I get is 0.8, for \\(2^{14}\\) datasets.

Now this is better! Lets try "uniqifying" the predicted NN and see what results are (still using SVM, same amont of data). The average testing coefficient of determination that I get is 0.5.

This looks better, however still in our simple setting a large amount of datasets and thus complicated SVM are requried; Furthermore, one can observe from the code that size of dataset and SVM need to grow exponentially to improve significantly the coefficient of determination.

This is due to the fact that RNN or SVM models are not well adapted to the training task. For example, for exact L2 regression solutions is:

$$ w = (X^T X)^{-1}(X^T y) $$

Notice that dependency between \\(w\\) and \\(X,y\\) is strongly non - linear, and thus would require large amount of data and support vectors of SVM to represent. 

Furthermore, it is known that training small neural networks to global optimality is [NP hard task](https://people.csail.mit.edu/rivest/pubs/BR93.pdf), which would mean that exponential size training SVM models would be required to train them. However, one can still hope that decent generalization (but not the best one) can be achieved with polynomial sized models.

**Predicting uniquely defined models with deep RNN**

Lets throw the heavy machinery of deep learning on the training task and see what happens!

[in progress ...]

**Conclusion**

Learning training algorithms is hard to attack with standard appraoches, thus different strategy needs to be taken. It is known that gradient descent works fine in practice; Thus starting from vanilla gradient descent, I will train neural networks with reinforcement learning to tune gradient descent parameters and improve its random initialization.