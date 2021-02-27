---
title: "Introduction to Machine Learning - Classification."
description:  The post present introduce machine learning classification problem with focus on logistic and multi-class logistic regression.

toc: false
comments: false
layout: post
categories: [Machine learning]
image: images/post/classification.png
author: Anthony Faustine
---

## Introduction


Previously we learned how to predict continuous-valued quantities as a linear function of input values. This post will describe a classification problem where the goal is to learn a mapping from inputs $x$ to target $t$ such that $t \in \{1\ldots C \}$ with $C$ being the number of classes.If $C = 2$, this is called binary classification (in which case we often assume $y \in \{0, 1\}$; if $C > 2$, this is called multiclass classification.

We will first consider binary classification problem in which the target classes $t$ will be generated from 2 class distributions: blue ($t=1$) and red ($t=0$). Samples from both classes are sampled from their respective distributions. These samples are plotted in the figure below.

Note that $X$ is a $N \times 2$ matrix of individual input samples $\mathbf{x}_i$, and that $\mathbf{t}$ is a corresponding $N \times 1$ vector of target values $t_i$.

## Logistic Regression

With logistic regression the goal is to predict the target class $t$ from the input values $x$. The network is defined as having an input $\mathbf{x} = [x_1, x_2]$ which gets transformed by the weights $\mathbf{w} = [w_1, w_2]$ to generate the probability that sample $\mathbf{x}$ belongs to class $t=1$$ This probability $P(t=1\mid \mathbf{x},\mathbf{w})$ is represented by the output $y$ of the network computed as $y = \sigma(\mathbf{x} * \mathbf{w}^T)$. $\sigma$ is the [logistic function](http://en.wikipedia.org/wiki/Logistic_function) and is defined as:

$$ 
\sigma(z) = \frac{1}{1+e^{-z}} 
$$ 

which squashes the predictions to be between 0 and 1 such that:

$$
\begin{aligned}
P(t=1| \mathbf{x},\mathbf{w}) &= y(\sigma(z))P(t=0\mid \mathbf{x},\mathbf{w})\\
 &= 1 - P(t=1\mid \mathbf{x},\mathbf{w}) = 1 - y(\sigma(z))
\end{aligned}
$$

The loss function for logistic function is called crossentropy and defined as:

$$
\mathcal{L}_{CE}(y,t)=\begin{cases} -\log y \quad \text{if } t = 1\\ -\log (1-y) \quad \text{if } t = 0
\end{cases}
$$

The crossentropy can be written in other form as:

$$
\mathcal{L}_{CE}(y,t)= -t \log y -(1-t)\log(1-y)
$$

When we combine the logistic activation function with cross-entropy loss, we get logistic regression:

$$
\begin{aligned}
z & = \mathbf{w^Tx + b}\\\ y & = \sigma(z)\\\ \mathcal{L}_{CE}(y,t) &= -t \log y -(1-t)\log(1-y)
\end{aligned}
$$

The cost function with respect to the model parameters $$\theta$$ (i.e. the weights and bias) is therefore: 

$$
\begin{aligned}
\varepsilon_{\theta} & = \frac{1}{N}\sum_{i=1}^N \mathcal{L}_{CE}(y,t)\\\ & = \frac{1}{N}\sum_{i=1}^N \left(-t^{(i)} \log y^{(i)} -(1-t^{(i)})\log(1-y^{(i)})\right)
\end{aligned}
$$

which can be implemented in python as follows:

```python
# Define the cost function
def cost(x, w, t):
 N, D = np.shape(x)
 z = z_value(x,w) 
 y = y_value(z)
 result = np.sum(np.multiply(t, np.log(y)) + np.multiply((1-t), np.log(1-y)))/float(N)
 return -result 
```

### Gradient Descent for Logistic Function

To derive the gradient descent updates, we'll need the partial derivatives of the cost function. We'll do this by applying the Chain Rule twice: first to compute 
$$
\frac{\partial \mathcal{L}_{CE}}{\partial z}
$$ 
and then again to compute $\frac{\partial \mathcal{L}_{CE}}{\partial w_j}$ But first, let's find 
$\frac{\partial y}{\partial z}$.

$$
\frac{\partial y}{ \partial z} = \frac{e^{-z}}{(1 + e^{-z})^2}= y(1-y)
$$

Now for the Chain Rule:

$$
\begin{aligned}
\frac{\partial \mathcal{L}_{CE}}{\partial z} & =\frac{\partial \mathcal{L}_{CE}}{\partial y}\frac{\partial y}{ \partial z}\\\ & = \left(\frac{-t}{y} + \frac{1-t}{1-y} \right) y(1-y)\\\ &= y - t
\end{aligned}
$$

Similary:

$$
\begin{aligned}
\frac{\partial \mathcal{L}_{CE}}{\partial w_j} & =\frac{\partial \mathcal{L}_{CE}}{\partial z}\frac{\partial z}{ \partial w_j}\\\ & =\frac{\partial \mathcal{L}_{CE}}{\partial z} x_j\\\ &= (y - t)x_j
\end{aligned}
$$

We can also obtain $$\frac{\partial \mathcal{L}_{CE}}{\partial b}$$ as follows:

$$
\begin{aligned}
\frac{\partial \mathcal{L}_{CE}}{\partial b} &= \frac{\partial \mathcal{L}_{CE}}{\partial z}\frac{\partial z}{\partial b}\\ & = (y-t)
\end{aligned}
$$

The gradient descent algorithm works by taking the derivative of the cost function $$\varepsilon_{\theta}$$ with respect to the parameters, and updates the parameters in the direction of the negative gradient.The parameter $$\mathbf{w}$$ is iteratively updated by taking steps proportional to the negative of the gradient:

$$
\mathbf{w_{k+1}} = \mathbf{ w_k }- \alpha \frac{\partial \mathbf{\varepsilon}}{\partial \mathbf{w}}
$$

where: 

$$
\begin{aligned}
\frac{\partial \mathcal{L}_{CE}}{\partial \varepsilon} &= \frac{\partial \varepsilon }{\partial \mathcal{L}_{CE}}\cdot\frac{\partial \mathcal{L}_{CE}}{\partial \mathbf{w}}\\ &= \frac{1}{N} \mathbf{x^T(y - t)}
\end{aligned}
$$

which can be implemented in python as follows:

```python
#gradient
def gradient(x, w, t):
 z = z_value(x,w) 
 y = y_value(z)
 error = y-t
 dw = x.T.dot(error)
 return dw.T

 def solve_gradient(x, t, alpha=0.1, tolerance=1e-2):
 N, D = np.shape(x)
 w = np.zeros([D])
 iterations = 1
 w_cost = [(w, cost(x,w, t))]
 while True:
 dw = gradient(x, w, t)
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
 return w 
```

Let us apply the above concept in the following example. Consider the case we want to predict whether a student with a specific pass mark can be admitted or not.

```python
# load dataset
admission = pd.read_csv('data/admission.csv', names = ["grade1", "grade2", "remark"])
admission.head()
```

The data-preprosessing is done using the following python code:
```python
features = ['grade1', 'grade2']
target = ['remark']
targetVal = admission[target]
featureVal = admission[features]
y = np.array(targetVal)

# Standardize the features
for i in range(2):
 featureVal.iloc[:,i] = (featureVal.iloc[:,i] / featureVal.iloc[:,i].max())

# Add bias term to feature data
b = np.ones((featureVal.shape[0], 1))
X = np.hstack((b, featureVal))

# randomly separate data into training and test data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=50) 
```

We use the solve_gradient function defined before to find the parameter for logistic regression. 

```python
w_g = solve_gradient(X_train, y_train, alpha=0.05, tolerance = 1e-9)
```
Now that you learned the parameters of the model, you can use the model to predict whether a particular student will be admitted.

Let define the prediction function that only 1 or 0 depending on the predicted class.

```python
def predict(x,w): 
 z = z_value(x,w)
 y = y_value(z)
 return np.around(y)
```

To find the accuracy of the model:

```python
p_test = predict(X_test, w_g)
p_train = predict(X_train, w_g)
print ('Test Accuracy: %f' % ((y_test[np.where(p_test == y_test)].size / float(y_test.size)) * 100.0))
print ('Train Accuracy: %f' % ((y_train[np.where(p_train == y_train)].size / float(y_train.size)) * 100.0))

```
After running the above codes, we found that our model performs a training accuracy of $$91.25$$ and a test accuracy of $$85$$ percents.

## Multiclass classification

So far, we've talked about binary classification, but most classification problems involve more than two categories. Fortunately, this doesn't require any new ideas: everything pretty much works by analogy with the binary
case. The first question is how to represent the targets. We could describe them as integers, but it's convenient to use indicator vectors or a one-of-K encoding.

Since there are $K$ outputs and $$D$$ inputs, the linear function requires $K \times D$ matrix as well as $K$ dimensional bias vector. We use **softmax function** which is the multivariate generalization given as:

$$
y_k = softmax(z_1 \ldots z_k) = \frac{e^{z_k}}{\sum_k e^{z_k}}
$$

 and can be implemented in python as

```python
 def softmax(x,w):
 z = z_value(x,w)
 e_x = np.exp(x - np.max(x))
 y = np.exp(z - max(z)) / np.sum(np.exp(z - max(z)))
 return y.reshape(len(y), 1)
```

Finally, the loss function (cross-entropy) for multiple-output case can be generalized as follows:

$$
\begin{aligned}
\mathcal{L}_{CE}(y,t) &= -\sum_{k=1}^K t_k \log y_k\\ &= -\mathbf{t^T}\log\mathbf{y}
\end{aligned}
$$

Combining these things together, we get multiclass logistic regression:

$$
\begin{aligned} 
\mathbf{z} &= \mathbf{wx + b} \\ \mathbf{y} &= softmax(\mathbf{z})\\ \mathcal{L}_{CE}(y,t) &=-\mathbf{t^T}\log\mathbf{y} \\
\end{aligned}
$$

## Gradient Descent for Multiclass Logistic Regression for Multiclass logistic regression:

Let consider the derivative with respect to the loss:

$$
\begin{aligned}
\frac{\partial {\mathcal L}_\text{CE}}{\partial w_{kj}} &= \frac{\partial }{\partial w_{kj}} \left(-\sum_l t_l \log(y_l)\right) \\ &= -\sum_l \frac{t_l}{y_l} \frac{\partial y_l}{\partial w_{kj}}
\end{aligned}
$$

Normally in calculus we have the rule:

$$
\begin{aligned}
\frac{\partial y_l}{\partial w_{kj}} &= \sum_m \frac{\partial y_l}{\partial z_m} \frac{\partial z_m}{\partial w_{kj}}
\end{aligned}
$$

But $$w_{kj}$$ is independent of $$z_m$$ for $$m \ne k$$, so 

$$
\begin{aligned}
\frac{\partial y_l}{\partial w_{kj}} &= \frac{\partial y_l}{\partial z_k} \frac{\partial z_k}{\partial w_{kj}}
\end{aligned}
$$


AND

$$
\frac{\partial z_k}{\partial w_{kj}} = x_j
$$

Thus

$$
\begin{aligned}
\frac{\partial {\mathcal L}_\text{CE}}{\partial w_{kj}} &= -\sum_l \frac{t_l}{y_l} \frac{\partial y_l}{\partial z_k} \frac{\partial z_k}{\partial w_{kj}} \\
 &= -\sum_l \frac{t_l}{y_l} \frac{\partial y_l}{\partial z_k} x_j \\
 &= x_j (-\sum_l \frac{t_l}{y_l} \frac{\partial y_l}{\partial z_k}) \\
 &= x_j \frac{\partial {\mathcal L}_\text{CE}}{\partial z_k} 
\end{aligned}
$$

Now consider derivative with respect to $z_k$ we can show (onboard) that.

$$
\frac{\partial y_l}{\partial z_k} = y_k (I_{k,l} - y_l)
$$

Where $I_{k,l} = 1$ if $k=l$ and $0$ otherwise.

Therefore

$$
\begin{aligned}
\frac{\partial {\mathcal L}_\text{CE}}{\partial z_k} &= -\sum_l \frac{t_l}{y_l} (y_k (I_{k,l} - y_l)) \\ &=-\frac{t_k}{y_k} y_k(1 - y_k) - \sum_{l \ne k} \frac{t_l}{y_l} (-y_k y_l) \\
 &= - t_k(1 - y_k) + \sum_{l \ne k} t_l y_k \\
 &= -t_k + t_k y_k + \sum_{l \ne k} t_l y_k \\
 &= -t_k + \sum_{l} t_l y_k \\
 &= -t_k + y_k \sum_{l} t_l \\
 &= -t_k + y_k \\
 &= y_k - t_k
\end{aligned}
$$

Putting it all together

$$
\begin{aligned}
\frac{\partial {\mathcal L}_\text{CE}}{\partial w_{kj}} &= x_j (y_k - t_k)
\end{aligned}
$$

In vectorization form it become:


$$
\begin{aligned}
\frac{\partial \mathcal {L}_{CE}}{\partial {\mathbf W}} = (\mathbf{y} - \mathbf{t}) \mathbf{x}^T 
\end{aligned}
$$

### Cross-entropy cost function

The cross entropy cost function for multiclass classification is given with respect to the model parameters $$\theta$$ (i.e. the weights and bias) is therefore: 

$$
\begin{aligned}
\varepsilon_{\theta} & = \frac{1}{N}\sum_{i=1}^N \mathcal{L}_{CE}(y,t)\\
 & = \frac{-1}{N}\sum_{i=1}^N \sum_{k=1}^K t_k \log y_k
\end{aligned}
$$

The gradient descent algorithm will be:
$$
\mathbf{w_{k+1}} = \mathbf{ w_k }- \alpha \frac{\partial \mathbf{\varepsilon}}{\partial \mathbf{w}}
$$

where:

$$
\begin{aligned}
\frac{\partial \mathcal{L}_{CE}}{\partial \varepsilon} &= \frac{\partial \varepsilon }{\partial \mathcal{L}_{CE}}\cdot\frac{\partial \mathcal{L}_{CE}}{\partial \mathbf{w}}\\
 &= \frac{1}{N} \mathbf{x^T(y - t)}
\end{aligned}
$$



## References
* [CSC321 Intro to Neural Networks and Machine Learning](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/)
* [Supervised and Unsupervised Machine Learning Algorithms](http://machinelearningmastery.com/supervised-and-unsupervised-machine-learning-algorithms/)