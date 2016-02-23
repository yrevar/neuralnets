    
class ExpConvNet(object):
    """
      A 11-layer convolutional network with the following architecture:
  
      (conv - relu)x4 - affine - relu - affine - softmax

      The network operates on minibatches of data that have shape (N, C, H, W)
      consisting of N images, each with height H and width W and with C input
      channels.
    """
    def __init__(self, input_dim=(3, 32, 32), num_filters=[32]*4, filter_size=[3]*4,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, time_it=False):
        
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.time_it = time_it 
        
        C, H, W = input_dim

        for i in range(4):
            
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
            
        i += 1
        print "FC layer %d size %dx%d"%(i+1, num_filters[i-1]*H_out*W_out, hidden_dim)
        self.params['W'+str(i+1)] = np.random.randn(num_filters[i-1]*H_out*W_out, hidden_dim)/np.sqrt(num_filters[i-1]*H_out*W_out/2)
        self.params['b'+str(i+1)] = np.zeros((hidden_dim))
        
        i += 1
        print "FC layer %d size %dx%d"%(i+1, hidden_dim, num_classes)
        self.params['W'+str(i+1)] = np.random.randn(hidden_dim, num_classes)/np.sqrt(hidden_dim/2)
        self.params['b'+str(i+1)] = np.zeros((num_classes))

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
        W4, b4 = self.params['W4'], self.params['b4']

        W5, b5 = self.params['W5'], self.params['b5']
        W6, b6 = self.params['W6'], self.params['b6']
        

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        scores = None
        
        if self.time_it: t = {}
        if self.time_it: t['0'] = time()
        layer1_conv_relu, layer1_cache = conv_relu_forward(X, W1, b1, conv_param)
        if self.time_it: t['1'] = time()
        layer2_conv_relu, layer2_cache = conv_relu_forward(layer1_conv_relu, W2, b2, conv_param)
        if self.time_it: t['2'] = time()
        layer3_conv_relu, layer3_cache = conv_relu_forward(layer2_conv_relu, W3, b3, conv_param)
        if self.time_it: t['3'] = time()
        layer4_conv_relu, layer4_cache = conv_relu_forward(layer3_conv_relu, W4, b4, conv_param)
        if self.time_it: t['4'] = time()

        layer5_affine_relu, layer5_cache_affine_relu = affine_relu_forward(layer4_conv_relu, W5, b5)
        if self.time_it: t['5'] = time()
        scores, layer6_cache_affine = affine_forward(layer5_affine_relu, W6, b6)
        if self.time_it: t['6'] = time()
        if self.time_it:
            for i in range(1,7):
                print "layer %d time %fs"%(i, t[str(i)]-t[str(i-1)])

        if y is None:
          return scores

        loss, grads = 0.0, {}

        loss, dx = softmax_loss(scores, y)
        dx, grads["W6"], grads["b6"] = affine_backward(dx, layer6_cache_affine)
        dx, grads["W5"], grads["b5"] = affine_relu_backward(dx, layer5_cache_affine_relu)
        dx, grads["W4"], grads["b4"] = conv_relu_backward(dx, layer4_cache)
        dx, grads["W3"], grads["b3"] = conv_relu_backward(dx, layer3_cache)
        dx, grads["W2"], grads["b2"] = conv_relu_backward(dx, layer2_cache)
        dx, grads["W1"], grads["b1"] = conv_relu_backward(dx, layer1_cache)

        loss += 0.5 * self.reg * ( \
                                  np.sum(self.params["W1"]*self.params["W1"]) \
                                  + np.sum(self.params["W2"]*self.params["W2"]) \
                                  + np.sum(self.params["W3"]*self.params["W3"]) \
                                  + np.sum(self.params["W4"]*self.params["W4"]) \
                                  + np.sum(self.params["W5"]*self.params["W5"]) \
                                  + np.sum(self.params["W6"]*self.params["W6"]) )

        grads["W1"] += self.reg * self.params["W1"]
        grads["W2"] += self.reg * self.params["W2"]
        grads["W3"] += self.reg * self.params["W3"]
        grads["W4"] += self.reg * self.params["W4"]
        grads["W5"] += self.reg * self.params["W5"]
        grads["W6"] += self.reg * self.params["W6"]

        return loss, grads

    
class ElevenLayerConvNet(object):
    """
      A 11-layer convolutional network with the following architecture:
  
      (conv - relu - conv - relu - conv - relu - 2x2 max pool)x2 - affine - relu - affine - softmax

      The network operates on minibatches of data that have shape (N, C, H, W)
      consisting of N images, each with height H and width W and with C input
      channels.
    """
    def __init__(self, input_dim=(3, 32, 32), num_filters=[64]*6, filter_size=[3]*6,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
        
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        
        data_in = input_dim

        for i in range(3):
            
            C, H, W = data_in
            
            padding = (filter_size[i] - 1) / 2
            assert padding == int(padding)
            stride = 1
            HH = WW = filter_size[i]
            H_out = (H - HH + 2*padding)/stride + 1
            W_out = (W - WW + 2*padding)/stride + 1
            assert H_out is int(H_out)
            assert W_out is int(W_out)
            
            self.params['W'+str(i+1)] = weight_scale*np.random.randn(num_filters[i], C, HH, WW)
            self.params['b'+str(i+1)] = np.zeros((num_filters[i]))
            data_in = num_filters[i], H_out, W_out
            
        # 2x2 max pooling
        H_out /= 2
        W_out /= 2
        
        data_in = num_filters[i], H_out, W_out
        
        for i in range(3,6):
            
            C, H, W = data_in
            
            padding = (filter_size[i] - 1) / 2
            assert padding == int(padding)
            stride = 1
            HH = WW = filter_size[i]
            H_out = (H - HH + 2*padding)/stride + 1
            W_out = (W - WW + 2*padding)/stride + 1
            assert H_out is int(H_out)
            assert W_out is int(W_out)
            
            self.params['W'+str(i+1)] = weight_scale*np.random.randn(num_filters[i], C, HH, WW)
            self.params['b'+str(i+1)] = np.zeros((num_filters[i]))
            data_in = num_filters[i], H_out, W_out
            
        # 2x2 max pooling
        H_out /= 2
        W_out /= 2
        i += 1
        self.params['W'+str(i+1)] = weight_scale*np.random.randn(num_filters[i-1]*H_out*W_out, hidden_dim)
        self.params['b'+str(i+1)] = np.zeros((hidden_dim))
        
        i += 1
        self.params['W'+str(i+1)] = weight_scale*np.random.randn(hidden_dim, num_classes)
        self.params['b'+str(i+1)] = np.zeros((num_classes))

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
        W4, b4 = self.params['W4'], self.params['b4']
        W5, b5 = self.params['W5'], self.params['b5']
        W6, b6 = self.params['W6'], self.params['b6']
        
        W7, b7 = self.params['W7'], self.params['b7']
        W8, b8 = self.params['W8'], self.params['b8']
        

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        
        layer1_conv_relu, layer1_cache = conv_relu_forward(X, W1, b1, conv_param)
        layer2_conv_relu, layer2_cache = conv_relu_forward(layer1_conv_relu, W2, b2, conv_param)
        layer3_conv_relu, layer3_cache = conv_relu_pool_forward(layer2_conv_relu, W3, b3, conv_param, pool_param)
        
        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W4.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        
        layer4_conv_relu, layer4_cache = conv_relu_forward(layer3_conv_relu, W4, b4, conv_param)
        layer5_conv_relu, layer5_cache = conv_relu_forward(layer4_conv_relu, W5, b5, conv_param)
        layer6_conv_relu, layer6_cache = conv_relu_pool_forward(layer5_conv_relu, W6, b6, conv_param, pool_param)

        layer7_affine_relu, layer7_cache_affine_relu = affine_relu_forward(layer6_conv_relu, W7, b7)
        scores, layer8_cache_affine = affine_forward(layer7_affine_relu, W8, b8)

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
        dx, grads["W8"], grads["b8"] = affine_backward(dx, layer8_cache_affine)
        dx, grads["W7"], grads["b7"] = affine_relu_backward(dx, layer7_cache_affine_relu)
        dx, grads["W6"], grads["b6"] = conv_relu_pool_backward(dx, layer6_cache)
        dx, grads["W5"], grads["b5"] = conv_relu_backward(dx, layer5_cache)
        dx, grads["W4"], grads["b4"] = conv_relu_backward(dx, layer4_cache)
        dx, grads["W3"], grads["b3"] = conv_relu_pool_backward(dx, layer3_cache)
        dx, grads["W2"], grads["b2"] = conv_relu_backward(dx, layer2_cache)
        dx, grads["W1"], grads["b1"] = conv_relu_backward(dx, layer1_cache)

        loss += 0.5 * self.reg * ( \
                                  np.sum(self.params["W1"]*self.params["W1"]) \
                                  + np.sum(self.params["W2"]*self.params["W2"]) \
                                  + np.sum(self.params["W3"]*self.params["W3"]) \
                                  + np.sum(self.params["W4"]*self.params["W4"]) \
                                  + np.sum(self.params["W5"]*self.params["W5"]) \
                                  + np.sum(self.params["W6"]*self.params["W6"]) \
                                  + np.sum(self.params["W7"]*self.params["W7"]) \
                                  + np.sum(self.params["W8"]*self.params["W8"]) )

        grads["W1"] += self.reg * self.params["W1"]
        grads["W2"] += self.reg * self.params["W2"]
        grads["W3"] += self.reg * self.params["W3"]
        grads["W4"] += self.reg * self.params["W4"]
        grads["W5"] += self.reg * self.params["W5"]
        grads["W6"] += self.reg * self.params["W6"]
        
        grads["W7"] += self.reg * self.params["W7"]
        grads["W8"] += self.reg * self.params["W8"]

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads