import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *
from time import time


class ExpConvNet(object):
    """
      A 11-layer convolutional network with the following architecture:
  
      (conv - leaky_relu - conv - leaky_relu - 3x3s2 pooling)x1 - batchnorm - affine - relu - affine - softmax

      The network operates on minibatches of data that have shape (N, C, H, W)
      consisting of N images, each with height H and width W and with C input
      channels.
    """
    def __init__(self, input_dim=(3, 32, 32), num_filters=[16]*2, filter_size=[3]*2,
                 fc_neurons=[100, 10], weight_scale=1e-3, reg=0.0, relu_leak_coeff = 1/3.0,
                 dtype=np.float32, use_batchnorm=False, dropout=0, seed=None, time_it=False):
        
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.time_it = time_it 
        self.filter_size = filter_size
        self.num_filters = len(num_filters)
        self.relu_leak_coeff = relu_leak_coeff
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        
        self.dropout_param = {}
        if self.use_dropout:
          self.dropout_param = {'mode': 'train', 'p': dropout}
          if seed is not None:
            self.dropout_param['seed'] = seed
        
        C, H, W = input_dim

        for i in range(self.num_filters):
            
            HH = WW = filter_size[i]
            
            print "Conv layer %d size %dx%dx%dx%d"%(i+1, num_filters[i], HH, WW, C)
            self.params['W'+str(i+1)] = weight_scale*np.random.randn(num_filters[i], C, HH, WW)
            self.params['b'+str(i+1)] = np.zeros((num_filters[i]))
            
            stride = 1
            padding = (filter_size[i] - 1) / 2
            assert padding == int(padding)
            
            H_out = (H - HH + 2*padding)/stride + 1
            W_out = (W - WW + 2*padding)/stride + 1
            assert H_out is int(H_out)
            assert W_out is int(W_out)
            
            C, H, W = num_filters[i], H_out, W_out
            
        # 2x2 max pooling
        H_out /= 2
        W_out /= 2
        
        num_neurons_prev = num_filters[i]*H_out*W_out
        
        for j in range(len(fc_neurons)):
            
            layer_no = self.num_filters + j + 1
            print "FC layer %d size %dx%d"%(layer_no, num_neurons_prev, fc_neurons[j])
            
            # He et. al 2015 initialization 
            self.params['W'+str(layer_no)] \
                    = np.random.randn(num_neurons_prev, fc_neurons[j])/np.sqrt(num_neurons_prev/2)
            self.params['b'+str(layer_no)] \
                    = np.zeros((fc_neurons[j]))
            if self.use_batchnorm and j != len(fc_neurons)-1:
                self.params["gamma"+str(layer_no)] = np.random.randn(fc_neurons[j])
                self.params["beta"+str(layer_no)] = np.random.randn(fc_neurons[j])
                
            num_neurons_prev = fc_neurons[j]

        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'}]
    
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)
    
    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        """
        scores = None
        
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        
        if self.use_batchnorm:
            gamma3, beta3 = self.params['gamma3'], self.params['beta3']
            
        W4, b4 = self.params['W4'], self.params['b4']

        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
        layer1_conv, layer1_conv_cache = conv_forward_fast(X, W1, b1, conv_param)
        layer1_relu, layer1_relu_cache = leaky_relu_forward(layer1_conv, self.relu_leak_coeff)
        
        
        filter_size = W2.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
        layer2_conv, layer2_conv_cache = conv_forward_fast(layer1_relu, W2, b2, conv_param)
        layer2_relu, layer2_relu_cache = leaky_relu_forward(layer2_conv, self.relu_leak_coeff)
        

        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        layer2_pool, layer2_pool_cache = max_pool_forward_fast(layer2_relu, pool_param)
        
        if self.use_batchnorm:
            layer3_out, layer3_affine_relu_cache = \
                            affine_batchnorm_relu_forward(layer2_pool, W3, b3, gamma3, beta3, self.bn_params[0])
        else:
            layer3_out, layer3_affine_relu_cache = \
                            affine_relu_forward(layer2_pool, W3, b3)
        if self.use_dropout:
            layer3_out, layer3_dropout_cache = dropout_forward(layer3_out, self.dropout_param)
            
        scores, layer4_affine_cache = affine_forward(layer3_out, W4, b4)
        
        if y is None:
          return scores
    
        loss, grads = 0.0, {}
        
        loss, dx = softmax_loss(scores, y)
        
        dx, grads["W4"], grads["b4"] = affine_backward(dx, layer4_affine_cache)
        
        if self.use_dropout:
            dx = dropout_backward(dx, layer3_dropout_cache)
            
        if self.use_batchnorm:
            dx, grads["W3"], grads["b3"], \
                grads['gamma3'], grads['beta3'] = affine_batchnorm_relu_backward(dx, layer3_affine_relu_cache)
        else:
            dx, grads["W3"], grads["b3"] = affine_relu_backward(dx, layer3_affine_relu_cache)
        
        dx = max_pool_backward_fast(dx, layer2_pool_cache)
        dx = leaky_relu_backward(dx, layer2_relu_cache)
        dx, grads["W2"], grads["b2"] = conv_backward_fast(dx, layer2_conv_cache)        
        dx = leaky_relu_backward(dx, layer1_relu_cache)
        dx, grads["W1"], grads["b1"] = conv_backward_fast(dx, layer1_conv_cache)
        
        loss += 0.5 * self.reg * ( \
                              np.sum(self.params["W1"]*self.params["W1"]) \
                              + np.sum(self.params["W2"]*self.params["W2"]) \
                              + np.sum(self.params["W3"]*self.params["W3"]) \
                              + np.sum(self.params["W4"]*self.params["W4"]) )
    
        grads["W1"] += self.reg * self.params["W1"]
        grads["W2"] += self.reg * self.params["W2"]
        grads["W3"] += self.reg * self.params["W3"]
        grads["W4"] += self.reg * self.params["W4"]

        return loss, grads


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, time_it=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.time_it = time_it
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    padding = (filter_size - 1) / 2
    stride = 1
    HH = WW = filter_size
    H_out = (H - HH + 2*padding)/stride + 1
    W_out = (W - WW + 2*padding)/stride + 1
    
    assert H_out is int(H_out)
    assert W_out is int(W_out)
    
    print "Conv layer %d size %dx%dx%dx%d"%(1, num_filters, HH, WW, C)
    
    print "Conv layer %d size %dx%d"%(2, num_filters*H_out*W_out/4, hidden_dim)
    
    print "Conv layer %d size %dx%d"%(3, hidden_dim, num_classes)
    
    self.params['W1'] = weight_scale*np.random.randn(num_filters, C, HH, WW)
    self.params['b1'] = np.zeros((num_filters))
    self.params['W2'] = weight_scale*np.random.randn(num_filters*H_out*W_out/4, hidden_dim)
    self.params['b2'] = np.zeros((hidden_dim))
    self.params['W3'] = weight_scale*np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros((num_classes))
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    #print W1.shape
    #print conv_param
    if self.time_it: t = {}
    if self.time_it: t['0'] = time()    
    layer1_conv_relu_pool, layer1_cache_conv_relu_pool = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    if self.time_it: t['1'] = time()
    layer2_affine_relu, layer2_cache_affine_relu = affine_relu_forward(layer1_conv_relu_pool, W2, b2)
    if self.time_it: t['2'] = time()
    scores, layer3_cache_affine = affine_forward(layer2_affine_relu, W3, b3)
    if self.time_it: t['3'] = time()
        
    if self.time_it:
        for i in range(1,4):
            print "layer %d time %fs"%(i, t[str(i)]-t[str(i-1)])    
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dx = softmax_loss(scores, y)
    dx, grads["W3"], grads["b3"] = affine_backward(dx, layer3_cache_affine)
    dx, grads["W2"], grads["b2"] = affine_relu_backward(dx, layer2_cache_affine_relu)
    dx, grads["W1"], grads["b1"] = conv_relu_pool_backward(dx, layer1_cache_conv_relu_pool)
    
    loss += 0.5 * self.reg * ( \
                              np.sum(self.params["W1"]*self.params["W1"]) \
                              + np.sum(self.params["W2"]*self.params["W2"]) \
                              + np.sum(self.params["W3"]*self.params["W3"]) )
    
    grads["W1"] += self.reg * self.params["W1"]
    grads["W2"] += self.reg * self.params["W2"]
    grads["W3"] += self.reg * self.params["W3"]
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
