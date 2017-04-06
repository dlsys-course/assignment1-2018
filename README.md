# Assignment 1: Reverse-mode Automatic Differentiation

In this assignment, we would implement reverse-mode auto-diff.

Our code should be able to construct simple expressions, e.g. y=x1*x2+x1,
and evaluate their outputs as well as their gradients (or adjoints), e.g. y, dy/dx1 and dy/dx2.

There are many ways to implement auto-diff, as explained in the slides for Lecture 4.
For this assignment, we use the approach of a computation graph and an explicit construction of gradient (adjoint) nodes, similar to what MXNet and Tensorflow do.

Key concepts and data structures that we would need to implement are
- Computation graph and Node
- Operator, e.g. Add, MatMul, Placeholder, Oneslike
- Construction of gradient nodes given forward graph
- Executor

## Overview of Module API and Data Structures

Here we use a simple example to illustrate the API and data structures of the autodiff module.

Suppose our expression is y=x1*x2+x1, we first define our variables x1 and x2 symbolically,

```python
import autodiff as ad

x1 = ad.Variable(name = "x1")
x2 = ad.Variable(name = "x2")
```
Then, you can define the symoblic expression for y,

```python
y = x1 * x2 + x1
```
Now, the computation graph looks like this,

![1st graph](https://github.com/dlsys-course/assignment1/blob/master/img/hwk1_graph1.png)

Here, each node is associated with an operator object (we only need a singleton instance for each operator since it is used in an immutable way).
- Node x1 and x2 are associated with Placeholder Op.
- Node (x1*x2) is associated with MulOp, and y with AddOp.

With this computation graph, we can evaluate the value of y given any values of x1 and x2: simply walk the graph in a topological order, and for each node, use its associated operator to compute an output value given input values. The evaluation is done in Executor.run method.

```python
executor = ad.Executor([y])
y_val = executor.run(feed_dict = {x1 : x1_val, x2 : x2_val})
```

If we want to evaluate the gradients of y with respect to x1 and x2, as we would often do for loss function wrt parameters in usual machine learning training steps, we need to construct the gradient nodes, grad_x1 and grad_x2.

```python
grad_x1, grad_x2 = ad.gradients(y, [x1, x2])
```

According to the reverse-mode autodiff algorithm described in the lecture, we create a gradient node for each node in the existing graph and return those that user are interested in evaluating.

We do this in a reverse topological order, e.g., y, (x1+x2), x1, x2, as shown in the figures below

![2nd graph](https://github.com/dlsys-course/assignment1/blob/master/img/hwk1_graph2.png)
![3rd graph](https://github.com/dlsys-course/assignment1/blob/master/img/hwk1_graph3.png)


Once we construct the gradients node, and have references to them, we can evaluate the gradients using Executor as before,
```python
executor = ad.Executor([y, grad_x1, grad_x2])
y_val, grad_x1_val, grad_x2_val = executor.run(feed_dict = {x1 : x1_val, x2 : x2_val})
```
grad_x1_val, grad_x2_val now contain the values of dy/dx1 and dy/dx2.

### Special notes here:
- For simplicity, our implementation expect all variables to have numpy.ndarray data type. See tests feed_dict usage.
- When taking derivative of dy/dx, even though y can be a vector, we are implicitly assuming that we are taking derivative of the reduce_sum(y) wrt x. This is the common case for machine learning applications as loss function is scalar or the reduce sum of vectors. Our code skeletion takes care of this by initializing the dy as ones_like(y) in Executor.gradients method.

## What you need to do?
Understand the code skeleton and tests. Fill in implementation wherever marked """TODO: Your code here""".

## Tests cases
We have 10 tests in autodiff_test.py. We would grade you based on those tests.

Run all tests with
```bash
# sudo pip install nose
nosetests -v autodiff_test.py
```

### Bonus points
Once your code can clear all tests, your autodiff module is almost ready to train a logistic regression model. If you are up for a challenge, try 

- Implement all missing operators necessary for a logistic regression, e.g. log, reduce_sum. 
- Write a simple training loop that updates parameters using gradients computed from autodiff module.

### Grading rubrics
- autodiff_test.test_identity ... 1 pt
- autodiff_test.test_add_by_const ... 1 pt
- autodiff_test.test_mul_by_const ... 1 pt
- autodiff_test.test_add_two_vars ... 1 pt
- autodiff_test.test_mul_two_vars ... 1 pt
- autodiff_test.test_add_mul_mix_1 ... 2 pt
- autodiff_test.test_add_mul_mix_2 ... 2 pt
- autodiff_test.test_add_mul_mix_3 ... 2 pt
- autodiff_test.test_grad_of_grad ... 2 pt
- autodiff_test.test_matmul_two_vars ... 2 pt

- bonus (training logistic regession) ... 5 pt

## Submitting your work

Please submit your autodiff.tar.gz to Catalyst dropbox under [Assignment 1](https://catalyst.uw.edu/collectit/dropbox/arvindk/40126).
```bash
# compress
tar czvf autodiff.tar.gz autodiff/
```
