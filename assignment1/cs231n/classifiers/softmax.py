import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #pass
  num_training = X.shape[0]
  num_classes = W.shape[1]

  for ii in xrange(num_training):
      scores = X[ii].dot(W)
      max_score = np.max(scores)
      scores -= max_score
      sum_exp = np.sum(np.exp(scores))
      exp_scores = lambda k: np.exp(scores[k]) / sum_exp
      #exp_scores /= sum_exp
      loss += -np.log(exp_scores(y[ii]))

      for kk in range(num_classes):
          dW[:, kk] += (exp_scores(kk) - (kk == y[ii])) * X[ii]

  loss /= num_training
  loss += 0.5 * reg * np.sum(W * W)

  dW /= num_training
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_training = X.shape[0]
  num_classes = W.shape[1]


  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  scores -= np.max(scores, axis=1, keepdims=True)
  scores = np.exp(scores)
  scores /= np.sum(scores, axis=1, keepdims=True)

  loss = np.sum(-np.log(scores[np.arange(num_training), y]))
  loss /= num_training
  loss += 0.5 * reg * np.sum(W * W)

  ind = np.zeros_like(scores)
  ind[np.arange(num_training), y] = 1

  dW = X.T.dot(scores - ind)
  dW /= num_training
  dW += reg * W



  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

