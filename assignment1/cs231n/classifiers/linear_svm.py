import numpy as np
from random import shuffle


def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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

  # Initialize loss and the gradient of W to zero.
  dW = np.zeros(W.shape)
  loss = 0.0
  num_classes = W.shape[1]
  num_train = X.shape[0]

  # Compute the data loss and the gradient.
  for i in range(num_train):  # For each image in training.
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    num_classes_greater_margin = 0

    for j in range(num_classes):  # For each calculated class score for this image.

      # Skip if images target class, no loss computed for that case.
      if j == y[i]:
        continue

      # Calculate our margin, delta = 1
      margin = scores[j] - correct_class_score + 1

      # Only calculate loss and gradient if margin condition is violated.
      if margin > 0:
        num_classes_greater_margin += 1
        # Gradient for non correct class weight.
        dW[:, j] = dW[:, j] + X[i, :]
        loss += margin

    # Gradient for correct class weight.
    dW[:, y[i]] = dW[:, y[i]] - X[i, :]*num_classes_greater_margin

  # Average our data loss across the batch.
  loss /= num_train

  # Add regularization loss to the data loss.
  loss += reg * np.sum(W * W)

  # Average our gradient across the batch and add gradient of regularization term.
  dW = dW /num_train + 2*reg *W
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  num_train = X.shape[0]

  scores = np.dot(X, W)

  # correct_class_scores = np.choose(y, scores.T)  # np.choose uses y to select elements from scores.T
  # 문제 5-1: 위 구문(line: 85)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
  correct_class_scores_li = [0 for i in range(y.shape[0])]
  for i, j in enumerate(y):
    correct_class_scores_li[i] = scores[i][j]
  correct_class_scores = np.array(correct_class_scores_li)

  mask = np.ones(scores.shape, dtype=bool)
  mask[range(scores.shape[0]), y] = False
  scores_ = scores[mask].reshape(scores.shape[0], scores.shape[1]-1)

  margin = scores_ - correct_class_scores[..., np.newaxis] + 1

  margin[margin < 0] = 0

  # loss = np.sum(margin) / num_train
  # 문제 5-2: 위 구문(line: 96)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
  loss = sum(sum(margin)) / num_train
  loss += reg * np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  original_margin = scores - correct_class_scores[...,np.newaxis] + 1
  pos_margin_mask = (original_margin > 0).astype(float)
  sum_margin = pos_margin_mask.sum(1) - 1
  pos_margin_mask[range(pos_margin_mask.shape[0]), y] = -sum_margin

  # dW = np.dot(X.T, pos_margin_mask)
  # 문제 5-3: 위 구문(line: 120)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
  dW_li = [[0 for j in range(pos_margin_mask.shape[1])] for i in range(X.shape[1])]

  for i in range(X.shape[1]):
    for j in range(pos_margin_mask.shape[1]):
      for k in range(X.shape[0]):
        dW_li[i][j] += X[k][i] * pos_margin_mask[k][j]
  dW = np.array(dW_li)

  dW = dW / num_train + 2 * reg * W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
