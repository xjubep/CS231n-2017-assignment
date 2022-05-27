import numpy as np
import math

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      for j in range(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################

        # L2 distance with numpy
        # dists[i,j] = np.sqrt(np.sum((X[i,:]-self.X_train[j,:])**2))
        
        # 문제 1: 위 구문(line: 78)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
        dist = math.sqrt(sum((X[i, :] - self.X_train[j, :]) ** 2))
        dists[i][j] = dist

        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################

      # L2 distance with numpy
      # dists[i,:] = np.sqrt(np.sum((X[i,:] - self.X_train)**2, axis = 1))

      # 문제 2: 위 구문(line: 105)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
      # (list to np array, tuple to np array 변환 함수(np.array())는 사용 가능)
      dists_li = [0 for j in range(num_train)]
      for j in range(num_train):
        dists_li[j] = math.sqrt(sum((X[i, :] - self.X_train[j, :]) ** 2))
      dists[i, :] = np.array(dists_li)
    
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################

    # L2 distance vectorized with numpy
    # X_squared = np.sum(X**2,axis=1)
    # Y_squared = np.sum(self.X_train**2,axis=1)
    # XY = np.dot(X, self.X_train.T)
    # dists = np.sqrt(X_squared[:,np.newaxis] + Y_squared -2*XY)
    
    # 문제 3: 위 구문(line: 139~142)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
    # (list to np array, tuple to np array 변환 함수(np.array())는 사용 가능)
    X_li = [0 for i in range(num_test)]
    Y_li = [0 for i in range(num_train)]

    for i, x in enumerate(X):
      X_li[i] = sum(x ** 2)

    for i, y in enumerate(self.X_train):
      Y_li[i] = sum(y ** 2)

    XY_li = [[0 for j in range(num_train)] for i in range(num_test)]
    for i in range(num_test):
      for j in range(num_train):
        for k in range(X.shape[1]):
          XY_li[i][j] += X[i][k] * self.X_train[j][k]

    dists_li = [[0 for j in range(num_train)] for i in range(num_test)]
    for i in range(num_test):
      for j in range(num_train):
        dists_li[i][j] = math.sqrt(X_li[i] + Y_li[j] - 2*XY_li[i][j])

    dists = np.array(dists_li)

    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################

      test_row = dists[i,:]
      # sorted_row = np.argsort(test_row)
      # 문제 4-1: 위 구문(line: 181)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
      test_row_li = list(test_row)
      sorted_row = sorted(range(len(test_row_li)), key=test_row_li.__getitem__)

      closest_y = self.y_train[sorted_row[0:k]]

      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################

      # y_pred[i] = np.argmax(np.bincount(closest_y))
        
      # 문제 4-2: 위 구문(line:193)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
      # (list to np array, tuple to np array 변환 함수(np.array())는 사용 가능)
      argmax = 0
      max_cnt = 0
      closest_y_li = list(closest_y)
      for idx in range(len(sorted_row)):
        bin_cnt = closest_y_li.count(idx)
        if bin_cnt > max_cnt:
          max_cnt = bin_cnt
          argmax = idx
      y_pred[i] = argmax

      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred

