import numpy as np
from random import shuffle

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
  num_train = X.shape[0]
  num_classes = W.shape[1]
  
  for i in range(num_train):
    scores = np.dot(X[i], W)
    loss +=  -np.log(softmax(scores)[y[i]])
    dscores = softmax(scores)
    dscores[y[i]] -= 1

    dW += np.dot(X[i].reshape((X[i].shape[0],1)), dscores.reshape((dscores.shape[0],1)).T)

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  loss /= num_train
  dW /= num_train
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

def softmax(vec):
  vec -= np.max(vec)
  return np.exp(vec) / np.sum(np.exp(vec))

def softmax_mat(mat):
  N = mat.shape[0]
  mat -= np.max(mat, axis=1).reshape(N,1)
  return np.exp(mat) / np.sum(np.exp(mat), axis=1).reshape(N,1)

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]

  scores = np.dot(X, W)

  loss = -np.sum(np.log(softmax_mat(scores)[range(num_train), y]))
  
  dscores = softmax_mat(scores)
  dscores[range(num_train),y] -= 1
  dW += np.dot(X.T, dscores)
  
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  loss /= num_train
  dW /= num_train

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

