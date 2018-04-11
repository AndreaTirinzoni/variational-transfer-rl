class Operator:

    def bellman_residual(self, Q, samples, weights=None):
        """General function for computing Bellman residuals"""
        pass

    def gradient_be(self, Q, samples, weights):
        """General function for gradients of the Bellman error"""
        pass

    def bellman_error(self, Q, samples, weights=None):
        """General function for computing the Bellman error"""
        pass

    def expected_bellman_error(self, Q, samples, weights):
        """Approximates the expected Bellman error with a finite sample of weights"""
        pass
