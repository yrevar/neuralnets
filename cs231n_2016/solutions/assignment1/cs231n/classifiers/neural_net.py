import numpy as np
import matplotlib.pyplot as plt
import math

def relu(x):
#     x[x<=0]=0
    return np.clip(x,0,float('Inf'))

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0, do_dropout=False, dropout_percent=0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    
    # Layer 1 
    layer1 = X.dot(W1) + b1
    layer1_relu = relu(layer1)
    if y is not None and dropout_percent:
        #layer1_relu *= np.random.binomial([np.ones_like(layer1_relu)],1-dropout_percent)[0] * (1.0/(1-dropout_percent))
        dropout_mask = (np.random.rand(*layer1_relu.shape) < (1-dropout_percent))/(1-dropout_percent)
        layer1_relu = dropout_mask*layer1_relu
        
    # Layer 2
    layer2 = layer1_relu.dot(W2) + b2
    scores = layer2
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Class probabilities
    f_scores = scores
    f_scores -= np.max(f_scores, axis=1, keepdims=True) # for numerical stability
    f_prob = np.exp(f_scores)/np.sum(np.exp(f_scores), axis=1, keepdims=True)
    
    # Cross entropy loss 
    correct_log_probs = np.log(f_prob[range(N),y])
    data_loss = -np.sum(correct_log_probs)/N
    
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################
    # Regularization loss
    reg_loss = 0.5*reg*(np.sum(W1*W1) + np.sum(W2*W2))
    
    # Total loss
    loss = data_loss + reg_loss
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    ###########################################################################
    # Softmax gradient 
    # Ref: http://nbviewer.jupyter.org/github/yrevar/machine_learning_blog/blob/draft/softmax_gradient_derivation/softmax_gradient_derivation.ipynb
    dscores = f_prob
    dscores[range(N),y] -= 1
    dscores /= N
    
    # Backprop into last layer
    # W2 * layer1_relu + b2 = scores (<-dscores). So, dW2 = layer1_relu * dscores, db2 = dscores
    grads['W2'] = np.dot(layer1_relu.T, dscores)
    grads['b2'] = np.sum(dscores, axis=0)
    
    # Next, Backprop into hidden layer
    # dlayer1_relu = W2 * dscores
    dlayer1_relu = np.dot(dscores, W2.T)
    
    # Backprop through dropout 
    if dropout_percent > 0:
        dlayer1_relu = dropout_mask*dlayer1_relu
    
    # Backprop throguh ReLU non-linearity
    dlayer1 = dlayer1_relu
    dlayer1[layer1_relu <= 0] = 0
    
    # Finally, backprop into first layer W1,b1
    # W1 * X + b1 = layer1. So, dW1 = dlayer1 * X, db1 = dlayer1
    grads['W1'] = np.dot(X.T, dlayer1)
    grads['b1'] = np.sum(dlayer1, axis=0)
    
    # Gradient of the regularization loss
    grads['W2'] += reg*W2
    grads['W1'] += reg*W1
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, dropout_percent=0, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      batch_indices = np.random.choice(num_train, batch_size, replace=True)
      X_batch = X[batch_indices]
      y_batch = y[batch_indices]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      if dropout_percent:
        loss, grads = self.loss(X_batch, y=y_batch, reg=reg, dropout_percent=dropout_percent)
      else:
        loss, grads = self.loss(X_batch, y=y_batch, reg=reg, dropout_percent=0)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      self.params['W1'] += -1*learning_rate*grads['W1']
      self.params['b1'] += -1*learning_rate*grads['b1']
      self.params['W2'] += -1*learning_rate*grads['W2']
      self.params['b2'] += -1*learning_rate*grads['b2']
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 300 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    y_pred = np.argmax(self.loss(X), axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


