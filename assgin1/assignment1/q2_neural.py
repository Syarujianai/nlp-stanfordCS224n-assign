#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(X, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    the backward propagation for the gradients for all parameters.

    Notice the gradients computed here are different from the gradients in
    the assignment sheet: they are w.r.t. weights, not inputs.

    Arguments:
    X -- M x Dx matrix, where each row is a training example x.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Note: compute cost based on `sum` not `mean`.
    ### YOUR CODE HERE: forward propagation
    _z1 = np.dot(X, W1) + b1    # _z1: (N, Dx) * (Dx, H) + (1, H) = (N, H)
    _h = sigmoid(_z1)   # _h: (N, H)
    _z2 = np.dot(_h, W2) + b2   # _z2: (N, H) * (H, Dy) + (1, Dy) = (N, Dy)
    _logits = softmax(_z2)   # _logit: (N, Dy)
    
    # raise NotImplementedError
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    # Cross entropy loss with softmax.
    # Note: softmax layer's input isn't sigmoided.
    cost = np.sum(-labels * np.log(_logits))   # (N, Dy) * (N, Dy) = (N, Dy)
    
    # Return: gradient value.
    # Softmax layer gradient with cross entropy loss formula:
    # d(cost)/d(_z2) = _logits - labels.
    # Note: gradW2.shape = W2.shape, so that we can update it.
    
    N = len(X)   # batch size
    
    # Softamax layer:
    # _logits = softmax(_z2)
    grad_z2 = _logits - labels   # gradz2: (N, Dy)
    
    # Ouput layer:
    # _z2 = _h * W2 + b2
    gradW2 = np.dot(_h.T, grad_z2)   # gradW2: (H, Dy) = (H, N) * (N, Dy)
    gradb2 = np.sum(grad_z2, axis=0)    # gradb2: (1,Dy) = (1, N) * (N, Dy), i.e., np.dot(np.ones(N).reshape(1,-1), (_logit - labels))
    grad_h = np.dot(grad_z2, W2.T)   # grad_h: (N, H) = (N, Dy) * (Dy, H)
    
    # Hidden layer:
    # _z1 = sigmoid(X * W1 + b1)
    grad_z1 = grad_h * sigmoid_grad(_h)   # grad_z1: (N, H) = (N, H)(N, H), elementwise product.
    gradW1 = np.dot(X.T, grad_z1)   # gradW1: (Dx, H) = (Dx, N) * (N, H)
    gradb1 = np.sum(grad_z1, axis=0)   # gradb1: (1, H) = (1, N) * (N, H), i.e., np.dot(np.ones(N).reshape(1,-1), grad_z1)
    
    # raise NotImplementedError
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    # N: batch size.
    N = 20
    dimensions = [10, 5, 10]
    # Each row will be a datum.
    data = np.random.randn(N, dimensions[0])
    labels = np.zeros((N, dimensions[2]))
    # xrange(): range() in python 3.
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    # A size-N training batch share the same DC value b1, b2.
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )
    
    # f(params): return cost, grad
    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
