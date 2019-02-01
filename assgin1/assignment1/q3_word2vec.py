#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """

    ### YOUR CODE HERE
    
    x = np.array([x_row / np.sqrt(np.sum(x_row**2)) for x_row in x])
    
    # raise NotImplementedError
    ### END YOUR CODE

    return x


def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    print x
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print ""


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    ### YOUR CODE HERE
    
    # Note:
    # For single input predicted word vector vc:
    # 1. predicted word vector: vc;
    # 2. other word vectors: U.
    
    # Softmax prediction
    logits = softmax(np.dot(predicted, outputVectors.T))   # logits: (1, 5) = (1, 3) * (3, 5)
    
    # Cross entropy loss with softmax.
    cost = - np.squeeze(np.log(logits[:, target]))
    # TESTING
    # print 'single c', cost
    
    # Softmax gradient.
    gradSoftmax = logits.copy()
    gradSoftmax[:, target] = logits[:, target] - 1
    
    # vc's gradient.
    gradPred = np.squeeze(np.dot(logits, outputVectors) - outputVectors[target])   # gradPred: (1, 3) = (1, 5) * (5, 3)
    
    # U's gradient.
    # Note: U = outputVectors.T
    grad = np.dot(gradSoftmax.T, predicted)   # grad: (5, 1) * (1, 3) = (5, 3), outer product.
    
    # raise NotImplementedError
    ### END YOUR CODE

    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models, 

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """
    
    # Note: call negSamplingCostAndGradient() for per word pair.
    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))
    
    ### YOUR CODE HERE
       
    # Difference1: Positive target context word and sample K negative non-target context.
    # unused:
    # 1. list(set(indices)).
    # 2. sigmoid_grad.
    # refer:
    # 1. https://blog.csdn.net/friyal/article/details/84875266
    # 2. http://demo.clab.cs.cmu.edu/cdyer/nce_notes.pdf
    sigmoidPOS = sigmoid(np.dot(predicted, outputVectors[target].T))
    sigmoidNEG = sigmoid(- np.dot(predicted, outputVectors[indices].T))   # sigmoidNEG: (1, K+1) = (1, 3) * (3, K+1)
    
    # Difference2: NEG loss.
    cost = np.squeeze(- np.log(sigmoidPOS) - np.sum(np.log(sigmoidNEG)))
    
    # Gradient:       
    # 1. vc's gradient.
    # 2. U's gradient.
    gradPred = np.squeeze((sigmoidPOS - 1) * outputVectors[target] - \
            np.dot(sigmoidNEG - 1, outputVectors[indices]))
            
    grad = np.zeros(outputVectors.shape)
    grad[target] += np.squeeze((sigmoidPOS - 1) * predicted)
    # version1:
    # grad[indices] = np.dot((1 - sigmoidNEG).T, predicted)   # grad[indices]: (K+1, 3) = (K+1, 1) * (1, 3)
    # version2:
    # cumulate the repeate negative sample's gradient.
    for i in range(K+1):
        grad[indices[i]] -= np.squeeze((sigmoidNEG[:, i] - 1) * predicted)
    
    # raise NotImplementedError
    ### END YOUR CODE

    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no 
    than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    
    # For current center word, sum up all context word pairs' loss and gradient.
    current = tokens[currentWord]
    currentVector = inputVectors[current].reshape(1, -1)
    pairsize = 2 * C
    
    # TESTING
    # print "size: ", len(contextWords)
    for c in contextWords:    
        if word2vecCostAndGradient == negSamplingCostAndGradient:
            cost_pair, gin_pair, gout_pair = negSamplingCostAndGradient(
                                            predicted=currentVector,  target=tokens[c], 
                                            outputVectors=outputVectors, dataset=dataset, K=10)
        else:
            cost_pair, gin_pair, gout_pair = softmaxCostAndGradient(
                                            predicted=currentVector, target=tokens[c], 
                                            outputVectors=outputVectors, dataset=dataset)
                                            
        cost += cost_pair / pairsize
        # print('note: ', cost, c)
        # Cumulate vc's gradient.
        gradIn[current] += gin_pair / pairsize
        # Cumulate U's gradient.
        gradOut += gout_pair / pairsize
        
    # raise NotImplementedError
    ### END YOUR CODE

    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    
    # For current center word, distribute all context word pairs' gradient.
    current = tokens[currentWord]    
    predictedVector = np.sum([inputVectors[tokens[c]] for c in contextWords], axis=0).reshape(1, -1)
    pairsize = 2 * C
    
    if word2vecCostAndGradient == negSamplingCostAndGradient:
        cost_all, gin_all, gout_all = negSamplingCostAndGradient(
                                        predicted=predictedVector, target=current, 
                                        outputVectors=outputVectors, dataset=dataset, K=10)
    else:
        cost_all, gin_all, gout_all = softmaxCostAndGradient(
                                        predicted=predictedVector, target=current, 
                                        outputVectors=outputVectors, dataset=dataset)
    
    for c in contextWords:        
        # Distribute vc's gradient, and it needn't to be normalized.
        # Note: 2*C context words may be repeate.
        gradIn[tokens[c]] += gin_all / pairsize
        
    # Normalize U's gradient and loss.
    # TODO: if len(contextWords) != 2*C ?
    gradOut += gout_all / pairsize
    cost = cost_all / pairsize
       
    # raise NotImplementedError
    ### END YOUR CODE

    return cost, gradIn, gradOut


#############################################
# My Testing code.   #
#############################################

def skipgram_check(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    cost, gradIn, gradOut = skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient)
    # TESTING
    # print "\ncost", cost
    
    return cost, np.concatenate((gradIn, gradOut), axis=0)


def cbow_check(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    cost, gradIn, gradOut = cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient)
    # TESTING
    # print "\ncost", cost
    
    return cost, np.concatenate((gradIn, gradOut), axis=0)


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    # Word vectors for all 5 tokens:
    # (5, 3) for input vectors(center word in skip-gram or CBOW), 
    # (5, 3) for output(context word).
    # wordVectors.shape: (10, 3) for 5 test tokens.
    wordVectors = np.array(wordVectors)
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    # Firstly, random training sample of single batch: 1 center word + 2 * C1 context word.
    # Secondly, C1: random size 1 ~ C.
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        # Sum up all averaged sample center word's loss and grad components.
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)
    
    # Random training sample of single batch:
    # which include center word and its context word,
    # sampling randomly from all tokens. 
    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    print "\n==== Gradient check for CBOW_NEG      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)
    
    print "\n=== naive check ==="
    print "==== Gradient check for skip-gram with one sample ===="
    gradcheck_naive(lambda vec: skipgram_check("c", 3, ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, vec[:5,:], vec[5:,:], dataset), dummy_vectors)
    gradcheck_naive(lambda vec: skipgram_check("c", 1, ["a", "b"],
        dummy_tokens, vec[:5,:], vec[5:,:], dataset,
        negSamplingCostAndGradient), dummy_vectors)
    print "==== Gradient check for cbow with one sample ===="
    gradcheck_naive(lambda vec: cbow_check("a", 2, ["a", "b", "c", "a"],
        dummy_tokens, vec[:5,:], vec[5:,:], dataset), dummy_vectors)
    gradcheck_naive(lambda vec: cbow_check("a", 2, ["a", "b", "a", "c"],
        dummy_tokens, vec[:5,:], vec[5:,:], dataset,
        negSamplingCostAndGradient), dummy_vectors)


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
