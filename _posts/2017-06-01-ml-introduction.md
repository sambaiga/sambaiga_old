---
title: "Introduction to Machine Learning"
description: The post presents the basic of machine learning with a focus on supervised learning ( linear regression) problem and how to implement it in python.

toc: true
comments: true
layout: post
categories: [Machine learning]
image: images/post/ml_into.jpg
author: Anthony Faustine
---

## Introduction

Machine learning is a set of algorithms that automatically detect patterns in data and use the uncovered pattern to make inferences or predictions. It is a subfield of artificial intelligence that aims to enable computers to learn on their own. Any machine learning algorithms involve the necessary three steps: first, you identify a pattern from data, build (train) model that best explains the pattern and the world (unseen data), and lastly, use the model to predict or make an inference. Model training (building)  can be seen as a learning process where the model is exposed to new, unfamiliar data step by step. 

Machine learning is an exciting and fast-moving field of computer science with many new applications. Applications where machine learning algorithms are regularly deployed includes: 

- Computer vision: Object Classification in Photograph, [image captioning](https://petapixel.com/2016/09/23/googles-image-captioning-ai-can-describe-photos-94-accuracy/).
- Speech recognition, Automatic Machine Translation.
- Detecting anomalies (e.g. Security, credit card fraud)
- Speech recognition.
- Communication systems<sup>[ref](https://www.hhi.fraunhofer.de/en/departments/wn/research-groups/signal-and-information-processing/research-topics/machine-learning-and-data-mining-for-communication-systems.html)<sup>
- Robots learning complex behaviors
- Recommendations services like in Amazo or Netflix where intelligent machine learning algorithms analyze your activity and compare it to the millions of other users to determine what you might like to buy or binge watch next<sup>[ref](https://www.forbes.com/sites/bernardmarr/2016/09/30/what-are-the-top-10-use-cases-for-machine-learning-and-ai/#4f49a7d894c9)</sup>.

Machine learning algorithms that learn to recognize what they see have been the heart of Apple, Google, Amazon, Facebook, Netflix, Microsoft, etc. 

### Why Machine learning

For many problems such as recognizing people and objects and understanding human speech, it's difficult to program the correct behavior by hand. However, with machine learning, these tasks are easier. Other reasons we might want to use machine learning to solve a given problem:

- A system might need to adapt to a changing environment. For instance, spammers are always trying to figure out ways to trick our e-mail spam classifiers, so the classification algorithms will need to adapt continually.
- A learning algorithm might be able to perform better than its human programmers. Learning algorithms have become world champions at a variety of games, from checkers to chess to Go. This would be impossible if the programs were only doing what they were explicitly told to do.
- We may want an algorithm to behave autonomously for privacy or fairness reasons, such as with ranking search results or targeting ads.


### Types of Machine Learning

Machine learning is usually divided into three major types: Supervised Learning, Unsupervised Learning and

**Supervised Learning**: Supervised learning is where you have input variables x and an output variable y, and use an algorithm to learn the mapping function from the input to the output<sup>[ref](http://machinelearningmastery.com/supervised-and-unsupervised-machine-learning-algorithms/)</sup>.  
For instance, if we're trying to train a machine-learning algorithm to distinguish cars and trucks, we would collect car and truck images and label each one as a car or a truck. Supervised learning problems can be further grouped into regression and classification problems.
 - **A regression problem**: is when the output variable is a real value, such as "dollars" or "weight" e.g Linear regression and Random forest.
 - **Classification**: A classification problem is when the output variable is a category, such as "red" or "blue" or "disease" and "no disease", e.g. Support vector machines, random forest, and logistic regression.
 Some famous examples of supervised machine learning algorithms are:


**Unsupervised Learning** : Unsupervised learning is where you only have input data (X) and no corresponding output variables. We just have a bunch of data and want to look for patterns in the data. For instance, maybe we have lots of examples of patients with autism and want to identify different subtypes of the condition. The most important types of unsupervised learning include:

 * **Distribution modeling** where one has an unlabeled dataset (such as a collection of images or sentences), and the goal is to learn a probability distribution that matches the dataset as closely as possible.
 * **Clustering** where the aim is to discover the inherent groupings in the data, such as grouping customers by purchasing behavior.


**Reinforcement Learning**: is [learning best actions based on reward or punishment](https://www.oreilly.com/ideas/reinforcement-learning-explained). It involves learning what actions to take in a given situation, based on *rewards*
and *penalties*. For example, a robot takes a big step forward, then falls. The next time, it takes a smaller stage and is able to hold its balance. The robot tries variations like this many times; eventually, it learns the right size of steps to take and walks steadily. It has succeeded.

There are three basic concepts in reinforcement learning: state, action, and reward. The state describes the current situation. Action is what an agent can do in each state. When a robot takes action in a state, it receives a reward, feedback from the environment. A reward can be positive or negative (penalties). 

## Typical ML task: Linear Regression

In regression, we are interested in predicting a scalar-valued target, such as the price of a stock. By linear, we mean that the target must be predicted as a linear function of the inputs. This is a kind of supervised learning algorithm; recall that, in supervised learning, we have a collection of training examples labeled with the correct outputs. Example applications of linear regression include weather forecasting, house pricing prediction, student performance (GPA) prediction, just to mention a few.

### Linear Regression: Formulating a learning problem
To formulate a learning problem mathematically, we need to define two things: a *model (hypothesis)** and a *loss function*. After defining model and loss function, we solve an optimization problem with the aim to find the model parameters that best fit the data.

**Model (Hypothesis)**: It is the set of allowable hypotheses or functions that compute predictions from the inputs. In the case of linear regression, the model simply consists of linear functions given by:

$$
y = \sum_j w_jx_j + b
$$

where $$w$$ is the weights, and $$b$$ is an intercept term, which we'll call the bias. These two terms are called model parameters denoted as $$\theta$$.

**Loss function**: It defines how well the model fit the data and thus show how far off the prediction $$y$$ is from the target $$t$$ and given as:

$$
\mathcal{L(y,t)} = \frac{1}{2}(y - t)^2
$$

Since the loss function show how far off the prediction is from the target for one data point. We also need to define a cost function. The cost function is simply the loss, averaged over all the training examples.

$$
\begin{aligned} 
J (w_1\ldots w_D,b) & = \frac{1}{N} \sum_{i=1}^N \mathcal{L}(y^{(i)},t^{(i)}) \\
 & = \frac{1}{2N}\sum_{i=1}^N (y^{(i)} - t^{(i)})^2 \\
 &=\frac{1}{2N}\sum_{i=1}^N \left(\sum_j w_jx_j^{(i)} + b -t^{(i)} \right)
\end{aligned}
$$

In vectorized form: 

$$ 
\mathbf{J} =\frac{1}{2N} \lVert\mathbf{y-t}\lVert^2 =\frac{1}{2N}\mathbf{(y - t)^T(y-t)} \quad \text{where}\quad \mathbf{y = w^Tx} 
$$


The python implementation of the cost function (vectorized) is shown below.

```python
def loss(x, w, t):
 N, D = np.shape(x)
 y = np.matmul(x,w.T)
 loss = (y - t)
 return loss
```

```python
def cost(x,w, t):
 '''
 Evaluate the cost function in a vectorized manner for 
 inputs `x` and targets `t`, at weights `w1`, `w2` and `b`.

 N, D = np.shape(x)
 return (loss(x, w,t) **2).sum() / (2.0 * N) 
```

Combine our model and loss function, we get an optimization problem, where we are trying to minimize a cost function concerning the model parameters $$\theta$$ (i.e. the weights and bias).

## Solving the optimization problem
We now want to find the choice of model parameters $$\theta _{w_1\ldots w_D,b}$$ that minimizes $$J (w_1\ldots w_D,b)$$ as given in the cost function above.There are two methods that we can use: direct solution and gradient descent.

### Direct Solution
One way to compute the minimum of a function is to set the partial derivatives to zero. For simplicity, let's assume the model doesn't have a bias term, as shown in the equation below. 

$$
J_\theta =\frac{1}{2N}\sum_{i=1}^N \left(\sum_j w_jx_j^{(i)} -t^{(i)} \right)
$$

In vectorized form

$$
\mathbf{J} =\frac{1}{2N}\lVert \mathbf{y-t}\rVert ^2 \frac{1}{2N}\mathbf{(y - t)^T(y-t)} \quad \text{where}\quad \mathbf{y = wx} 
$$

For matrix differentiation we need the following results:

$$
\begin{aligned}
 \frac{\partial \mathbf{Ax}}{\partial \mathbf{x}} & = \mathbf{A}^T \frac{\partial (\mathbf{x}^T\mathbf{Ax})}{\partial \mathbf{x}}\\ & = 2\mathbf{A}^T\mathbf{x}
\end{aligned}
$$

Setting the partial derivatives of cost function in vectorized form to zero we obtain:

$$
\begin{aligned}\frac{\partial \mathbf{J}}{\partial \mathbf{w}} & =\frac{1}{2N}\frac{\partial \left(\mathbf{w^Tx^Tx w} -2 \mathbf{t^Twx} + \mathbf{t^Tt}\right)}{\partial \mathbf{w}} \\
&=\frac{1}{2N}\left(2\mathbf{x}^T\mathbf{xw} -2\mathbf{x}^T\mathbf{t}\right) \\
\mathbf{w} &= (\mathbf{x^Tx})^{-1}\mathbf{x^Tt}
\end{aligned}
$$

In python, this result can be implemented as follows:

```python
def direct method(x, t):
'''
 Solve linear regression exactly. (fully vectorized)
 
 Given `x` - NxD matrix of inputs
 `t` - target outputs
 Returns the optimal weights as a D-dimensional vector
 '''
 N, D = np.shape(x)
 A = np.matmul(x.T, x)
 c = np.dot(x.T, t)
 return np.matmul(linalg.inv(A), c)
```

## Gradient Descent

The optimization algorithm commonly used to train machine learning is the gradient descent algorithm. It works by taking the derivative of the cost function $$J$$ with respect to the parameters at a specific position on this cost function and updates the parameters in the direction of the negative gradient. The entries of the gradient vector are simply the partial derivatives with respect to each of the variables: 

$$
\frac{\partial \mathbf{J}}{\partial \mathbf{w}} = \begin{pmatrix} \frac{\partial J}{\partial w_1}\\
 \vdots\\ \frac{\partial J}{\partial w_D}
\end{pmatrix}
$$

The parameter $$\mathbf{w}$$ is iteratively updated by taking steps proportional to the negative of the gradient:

$$
\mathbf{w_{t+1}} = \mathbf{ w_t }- \alpha \frac{\partial \mathbf{J}}{\partial \mathbf{w}} = \mathbf{w_t} - \mathbf{\frac{\alpha}{N}x^T(y-t)}
$$


In coordinate systems this is equivalent to:

$$
w_{t+1} = w_t - \alpha \frac{1}{N}\sum_{i=1}^{N} x_t (y^{(i)}-t^{(i)})
$$

The python implementation of gradient descent is shown below:

```python
def getGradient(x, w, t):
 N, D = np.shape(x)
 gradient = (1.0/ float(N)) * np.matmul(np.transpose(x), loss(x,w,t))
 return gradient
```

```python
def gradientDescentMethod(x, t, alpha=0.1, tolerance=1e-2):
 N, D = np.shape(x)
 #w = np.random.randn(D)
 w = np.zeros([D])
 # Perform Gradient Descent
 iterations = 1
 w_cost = [(w, cost(x,w, t))]
 while True:
 dw = getGradient(x, w, t)
 w_k = w - alpha * dw
 w_cost.append((w, cost(x, w, t)))
 # Stopping Condition
 if np.sum(abs(w_k - w)) < tolerance:
 print ("Converged.")
 break
 if iterations % 100 == 0:
 print ("Iteration: %d - cost: %.4f" %(iterations, cost(x, w, t)))
 iterations += 1
 w = w_k
 return w, w_cost
```


## Generalization

The goal of a learning algorithm is not only to make correct predictions on the training examples but also to be generalized to patterns not seen before. The average squared error on new examples is known as the generalization error, and we'd like this to be as small as possible. In practice, we normally tune model parameters by partitioning the dataset into three different subsets:

* The training set is used to train the model.
* The validation set is used to estimate the generalization error of each hyperparameter setting.
* The test set is used at the very end, to estimate the generalization error of the final model, once all hyperparameters have been chosen.