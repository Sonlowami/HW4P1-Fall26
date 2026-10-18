import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    
    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        Z_hat = np.moveaxis(Z, self.dim, -1)
        Z_hat_shape = Z_hat.shape
        flat_shape = (np.prod(Z_hat_shape[:-1]), Z_hat_shape[-1])
        Z_hat = Z_hat.reshape(flat_shape)
        exponent = np.exp(Z_hat - np.max(Z_hat, axis=1, keepdims=True))
        self.A = exponent / np.sum(exponent, axis=1, keepdims=True)
        self.A = self.A.reshape(Z_hat_shape)
        self.A = np.moveaxis(self.A, -1, self.dim)
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        dLdA_hat = np.moveaxis(dLdA, self.dim, -1)
        dLdA_hat_shape = dLdA_hat.shape
        N = np.prod(dLdA_hat_shape[:-1])
        C = dLdA_hat_shape[-1]
        dLdA_hat = dLdA_hat.reshape([N, C])
        A = np.moveaxis(self.A, self.dim, -1)
        A = A.reshape([np.prod(A.shape[:-1]), A.shape[-1]])

        dLdZ = np.zeros([N, C])

        for i in range(N):
            J = np.zeros([C, C])

            for m in range(C):
                for n in range(C):
                    J[m, n] = np.where(m == n, A[i, m]*(1 - A[i, m]), -A[i, m]*A[i, n])
            dLdZ[i, :] = np.matmul(dLdA_hat[i], J)

        dLdZ = dLdZ.reshape(dLdA_hat_shape)
        dLdZ = np.moveaxis(dLdZ, -1, self.dim)
        return dLdZ