import numpy as np
from random import shuffle

def softmax(f_scores):
    f_scores = np.matrix(f_scores)
    f_prob = np.exp(f_scores)/np.vstack(np.sum(np.exp(f_scores), axis=1))
    return f_prob

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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in range(num_train):
    f_scores = X[i].dot(W)
    f_scores -= np.max(f_scores) # for numerical stability
    
    # Softmax probabilities 
    p = np.exp(f_scores)/np.sum(np.exp(f_scores))
    
    # Cross entropy loss
    loss += -np.log(p[y[i]])
    
    for j in range(num_classes):
        # Gradient 
        # Ref: http://nbviewer.jupyter.org/github/yrevar/machine_learning_blog/blob/draft/softmax_gradient_derivation/softmax_gradient_derivation.ipynb
        dW[:,j] += (p[j] - (j==y[i]))*X[i,:]

  # Compute average
  loss /= num_train
  dW /= num_train

  # Regularization
  loss += 0.5*reg*np.sum(W*W)
  dW += reg*W
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
  num_classes = W.shape[1]
  num_train = X.shape[0]
    
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  f_scores = X.dot(W)
  f_scores -= np.max(f_scores, axis=1, keepdims=True) # for numerical stability
    
  # Softmax probabilities
  p = np.exp(f_scores)/np.sum(np.exp(f_scores), axis=1, keepdims=True) 
  
  # Cross entropy loss
  loss = -np.mean(np.log(p[range(num_train),y]))

  # Add regularization to the loss
  loss += 0.5*reg*np.sum(W*W)

  # we want p[j] - 1 for when j == y[i], otherwise p[j] - 0
  # Prepare a zero matrix of shape p and assign 1 to elements where p[j] == y[i] 
  p[range(num_train), y] -= 1

  # Gradient of the loss
  dW = np.dot(X.T, p) 
  dW /= num_train # gradient of the data loss
  dW += reg*W # gradient of the regularization loss
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW

