import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        # TODO: Implement forward pass
        input_shape = A.shape
        flattened_shape = (np.prod(input_shape[:-1]), input_shape[-1])
        A_r = A.reshape(flattened_shape)
        Z = A_r @ self.W.T + self.b.T
        Z = Z.reshape((*input_shape[:-1], Z.shape[-1]))
        # Store input for backward pass
        self.A = A
        
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # TODO: Implement backward pass
        z_shape = dLdZ.shape
        z_flat_shape = (np.prod(z_shape[:-1]), z_shape[-1])
        dLdZ = dLdZ.reshape(z_flat_shape)
        A = self.A.reshape(np.prod(z_flat_shape[:-1]), self.A.shape[-1])
        # Compute gradients
        self.dLdA = dLdZ @ self.W
        self.dLdW = dLdZ.T @ A
        self.dLdb = np.sum(dLdZ, axis=0)
        
        # Return gradient of loss wrt input
        self.dLdA = self.dLdA.reshape(self.A.shape)
        return self.dLdA
