import numpy as np
import scipy.sparse as sp
import math

class SVM:
    def __init__(self, X, y, reg):
        """ Initialize the SVM attributes and initialize the weights vector to the zero vector. 
            Attributes: 
                X (array_like) : training data intputs
                y (vector) : 1D numpy array of training data outputs
                reg (float) : regularizer parameter
                theta : 1D numpy array of weights
        """
        self.X = X
        self.y = sp.diags(y)
        self.reg = reg
        self.theta = np.zeros(X.shape[1])
    
    def objective(self, X, y):
        """ Calculate the objective value of the SVM. When given the training data (self.X, self.y), this is the 
            actual objective being optimized. 
            Args:
                X (array_like) : array of examples, where each row is an example
                y (array_like) : array of outputs for the training examples
            Output:
                (float) : objective value of the SVM when calculated on X,y
        """
        return np.maximum(1 - y*(X.dot(self.theta)), 0).sum() + self.reg/2*(self.theta.dot(self.theta))

    def gradient(self):
        """ Calculate the gradient of the objective value on the training examples. 
            Output:
                (vector) : 1D numpy array containing the gradient
        """ 
        indicator = (self.y*(self.X.dot(self.theta)) <= 1)
        return -np.array((sp.diags(self.y*indicator).dot(self.X)).sum(axis=0)).squeeze() + self.reg*self.theta

    def train(self, niters=100, learning_rate=1, verbose=False):
        """ Train the support vector machine with the given parameters. 
            Args: 
                niters (int) : the number of iterations of gradient descent to run
                learning_rate (float) : the learning rate (or step size) to use when training
                verbose (bool) : an optional parameter that you can use to print useful information (like objective value)
        """
        for i in xrange(niters):
            self.theta -= learning_rate*self.gradient()
            if verbose and i % 10 == 0:
                print self.objective(self.X, self.y)

    def predict(self, X):
        """ Predict the class of each label in X. 
            Args: 
                X (array_like) : array of examples, where each row is an example
            Output:
                (vector) : 1D numpy array containing predicted labels
        """
        return (X.dot(self.theta) >= 0).astype(int)*2 - 1

class ModelSelector:
    """ A class that performs model selection. 
        Attributes:
            blocks (list) : list of lists of indices of each block used for k-fold cross validation, e.g. blocks[i] 
            gives the indices of the examples in the ith block 
            test_block (list) : list of indices of the test block that used only for reporting results
            
    """
    def __init__(self, X, y, P, k, niters):
        """ Initialize the model selection with data and split into train/valid/test sets. Split the permutation into blocks 
            and save the block indices as an attribute to the model. 
            Args:
                X (array_like) : array of features for the datapoints
                y (vector) : 1D numpy array containing the output labels for the datapoints
                P (vector) : 1D numpy array containing a random permutation of the datapoints
                k (int) : number of folds
                niters (int) : number of iterations to train for
        """
        blocks = np.array_split(P, k + 1)
        self.test_block = blocks[-1]
        self.blocks = np.delete(blocks, -1, axis=0)
        self.X = X
        self.y = y
        self.k = k
        self.niters = niters

    def cross_validation(self, lr, reg):
        """ Given the permutation P in the class, evaluate the SVM using k-fold cross validation for the given parameters 
            over the permutation
            Args: 
                lr (float) : learning rate
                reg (float) : regularizer parameter
            Output: 
                (float) : the cross validated error rate
        """
        errors = []
        for i in xrange(self.k):
            holdout_indices = self.blocks[i]
            training_indices = np.concatenate(np.delete(self.blocks, i, axis=0))
            holdout_X, holdout_y = self.X[holdout_indices], self.y[holdout_indices]
            training_X, training_y = self.X[training_indices], self.y[training_indices]
            svm = SVM(training_X, training_y, reg)
            svm.train(learning_rate=lr, niters=self.niters)
            predictions = svm.predict(holdout_X)
            num_errors = np.count_nonzero(predictions - holdout_y)
            error = 1.0 * num_errors / len(holdout_y)
            errors.append(error)
        return np.average(errors)
            
    def grid_search(self, lrs, regs):
        """ Given two lists of parameters for learning rate and regularization parameter, perform a grid search using
            k-wise cross validation to select the best parameters. 
            Args:  
                lrs (list) : list of potential learning rates
                regs (list) : list of potential regularizers
            Output: 
                (lr, reg) : 2-tuple of the best found parameters
        """
        best_lr, best_reg, best_error = None, None, None
        for lr in lrs:
            for reg in regs:
                error = self.cross_validation(lr, reg)
                if best_error is None or error < best_error:
                    best_lr, best_reg, best_error = lr, reg, error
                    
        return best_lr, best_reg
    
    def test(self, lr, reg):
        """ Given parameters, calculate the error rate of the test data given the rest of the data. 
            Args: 
                lr (float) : learning rate
                reg (float) : regularizer parameter
            Output: 
                (err, svm) : tuple of the error rate of the SVM on the test data and the learned model
        """
        training_indices = np.concatenate(self.blocks)
        training_X, training_y = self.X[training_indices], self.y[training_indices]
        test_indices = self.test_block
        test_X, test_y = self.X[test_indices], self.y[test_indices]
        svm = SVM(training_X, training_y, reg)
        svm.train(learning_rate=lr, niters=self.niters)
        predictions = svm.predict(test_X)
        num_errors = np.count_nonzero(predictions - test_y)
        return (1.0 * num_errors / len(test_y), svm)
