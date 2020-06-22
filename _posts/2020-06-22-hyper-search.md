---
title: "Super-charge Deep learning hyper-paramater search with Optuna"
description:  Learn how to perform hyper-paramater search using Optuna
toc: false
comments: false
layout: post
show_tags: true
categories: [Machine learning, Deep learning]
image:  images/post/search.jpg
author: Anthony Faustine
---


## Introduction
Training machine learning sometimes involves various hyperparameter settings. Performing a hyperparameter search is an integral element in building machine learning models. It consists of attuning different sets of parameters to find the best settings for best model performance. It should be remarked that deep neural networks can involve many hyperparameter settings. Getting the best set parameters for such a high dimensional space might a challenging task. Opportunely, different strategies and tools can be used to simplify the process. This post will guide you on how to use Optuna for a hyper-parameter search using [PyTorch](https://pytorch.org/) and [PyTorch lightning](https://github.com/PyTorchLightning/pytorch-lightning) framework.
The notebook with all the code for this post can be found on this [colab] link (https://colab.research.google.com/drive/1QVST56bq3zNyIYx9595HVcq5fwFNH44x?usp=sharing).

### Optuna
[Optuna](https://optuna.org/) is an open-source hyperparameter optimization framework. It automates the process of searching for optimal hyperparameter using Python conditionals, loops, and syntax. The optuna library offers efficiently hyper-parameter search in large spaces while pruning unpromising trials for faster results. It is also possible to run a hyperparameter search over multiple processes without modifying code.
For a brief introduction of optuna, you can watch this video
{% include youtube.html content="https://youtu.be/J_aymk4YXhg" %}

The optuna optimization problem consists of three main building blocks; **objective function**, **trial**, and **study**. Let consider a simple optimisation problem: *Suppose a rectangular garden is to be constructed using a rock wall as one side of the garden and wire fencing for the other three sides as shown in figure below (taken from this [link](https://math.libretexts.org/Bookshelves/Calculus/Map%3A_Calculus_-_Early_Transcendentals_(Stewart)/04%3A_Applications_of_Differentiation/4.07%3A_Optimization_Problems)). Given 500m of wire fencing, determine the dimensions that would create a garden of maximum area. What is the maximum area?*
![]({{ site.baseurl }}/images/optuna/optuna_one.png)

Let $x$ denote the side of the garden's side perpendicular to the rock wall, and $y$ indicates the side parallel to the rock wall. Then the area of the garden $A= x \cdot y$. We want to find the maximum possible area subject to the constraint that the total fencing is 500m. The total amount of fencing used will be $2x+y$. Therefore, the constraint equation is 
$$
\begin{aligned}
500 & = 2x +y \\
y & = 500-2x\\
A(x) &= x \cdot (500-2x) = 500x - 2x^2
\end{aligned}
$$

From equation above, $A(x) = 500x - 2x^2$ is an **objective function**, the function to be optimized. To maximize this function, we need to determine optimization constraints. We know that to construct a rectangular garden, we certainly need the lengths of both sides to be positive $y>0$, and $x>0$. Since $500 = 2x +y$ and $y>0$ then $x<250$. Therefore, we will try to determine the maximum value of A(x) for x over the open interval (0,50).

Optuna [**trial**](https://optuna.readthedocs.io/en/stable/reference/trial.html) corresponds to a single execution of the **objective function** and is internally instantiated upon each invocation of the function. 
To obtain the parameters for each trial within a provided *constraints* the [**suggest**] method (https://optuna.readthedocs.io/en/stable/reference/trial.html) is used. 
```python
trial.suggest_uniform('x', 0, 250)
```

We can now code the objective function that be optimized for our problem.
```python
def gardent_area(trial):
 x = trial.suggest_uniform('x', 0, 250)
 return (500*x - 2*x**2 ) 
```

Once the objective function has been defined, the [study object]() is used to start the optimization.  The **study** is an optimization session, a set of trials. Optuna provide different [sampler strategies](https://optuna.readthedocs.io/en/latest/reference/samplers.html) such as Random Sampler and [Tree-structured Parzen Estimator (TPE)](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf) sampler. A sampler has the responsibility to determine the parameter values to be evaluated in a trial. By default, Optuna uses [TPE](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf) sampler, which is a form of Bayesian Optimization. The TPE provides a more efficient search than a random sampler search by choosing points closer to past good results. It possible to add a custom sampler as described in this [link](https://optuna.readthedocs.io/en/latest/tutorial/sampler.html#overview-of-sampler)
We can now create a study and start the optimization process. 
{% gist 4975107146ea19cf0dff730516c1f933 %}

Once the study is completed, you can get the best parameters using ```study.best_params``` and ```study.best_value``` will give you the best value.

## Hyper-param search for deep neural net

Suppose we want to build MLP classifier to recognize handwritten digits using the MNIST dataset. We will first build a pytorch MLP model with the following default parameters
```python
hparams = {"in_size": 28*28, "hidden_size":128, "out_size":10, "layer_size":5, "dropout":0.2}
```
{% gist 1056337e11a4887008a5b95d0f12a026 %}

For the above MLP model, we need to specify the following parameters *hidden size, dropout, and number of linear layers*. The critical question is, how do we pick these parameters. We will use optuna to search for optimal parameters that will give us an excellent performance. First, we will create a PyTorch lightning model that will provide the structure for organizing the fundamentals component of any machine learning project. These elements include the data, architecture or model, optimizer, loss function, training, and evaluation step. Since we fine defined our MLP, we go ahead and create a PyTorch lightning module.
The complete code with all the component mentioned above can be found on [gist link](https://gist.github.com/sambaiga/b835ab905d0b8199a859eae2ff7adfe6)
To learn the parameters of the MLP we will use Stochastic Gradient Descent Optimizer (SGD)  optimizer. The SGD has several other hper-parameters such as learning rate just we can also optimize.
```pyhon
 optimizer = torch.optim.SGD(self.model.parameters(), 
 lr=self.hparams['learning_rate'], 
 momentum=self.hparams['momentum'], 
 nesterov=self.hparams['nesterov'],
 weight_decay=self.hparams['weight_decay']) 
```

Thus the SGD optimizer will add four additional parameters. We can also treat the batch size as hyper-parameter to optimize. We will have the following set of parameters to optimizers.

```python
default_params = {"in_size": 28*28, "hidden_size":128, "out_size":10, 
 "layer_size":5, "dropout":0.2, "batch_size":32,
 'learning_rate':1e-3, 'momentum':0.9, 'nesterov': True,
 'weight_decay':1e-5,
 'epochs':2}
```

### Defining the hyperparameters and objective function to be optimized
Since we know all the parameters that we want to optimize, we will use the optuna **suggest** to define a search space for each hyperparameter that we want to tune. Optuna supports a variety of suggests which can be used to optimize floats, integers, or discrete categorical values. Numerical values such as learning rate can be suggested using a logarithmic scale.
{% gist b0c5dfe55d15cbdf2fc1983132a5ee03 %}

To create an objective function, we use the trainer module within PyTorch lightning with the default TensorBoard logger. The trainer will return the validation score. Optuna will use this score to evaluate the performance of the hyperparameters and decide where to sample in upcoming trials.
In addition to sampling strategies, Optuna provides a mechanism to automatically stops unpromising trials at the early stages of the training. This allows computing time to be used for tests that show more potential. This feature is called [**pruning**](https://optuna.readthedocs.io/en/stable/tutorial/pruning.html), and it is a form of automated early-stopping. The [PyTorchLightingPruningCallBack](https://optuna.readthedocs.io/en/stable/reference/integration.html) provides integration Optuna pruning  function to PyTorch lightning

{% gist 64c756606d04c1ab2f300f6b0ae9236d %}

To start the optimization, we create a study object and pass the objective function to method optimize() as follows.
```python
def run_study(num_trials=2):
 study = optuna.create_study(direction='maximize')
 study.optimize(objective, n_trials=num_trials)
 return study
```

After the study is completed, we can export trials as a pandas data frame. This provides various features to analyze studies. It is also useful to draw a histogram of objective values and to export trials as a CSV file. 
```python
df = study.trials_dataframe()
```

The notebook with all the code for this post can be found on this [colab link](https://colab.research.google.com/drive/1QVST56bq3zNyIYx9595HVcq5fwFNH44x?usp=sharing)
## References

- [Using Optuna to Optimize PyTorch Lightning Hyperparameters](https://medium.com/optuna/using-optuna-to-optimize-pytorch-lightning-hyperparameters-d9e04a481585)
- [Optuna documentation](https://optuna.readthedocs.io/en/stable/index.html)
- [Pytorch-lightning example](https://github.com/optuna/optuna/blob/master/examples/pytorch_lightning_simple.py)

