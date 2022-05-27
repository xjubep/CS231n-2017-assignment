import numpy as np
from random import shuffle
import math

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

  # scores = np.dot(X, W)
  # 문제 6-1: 위 구문(line:36)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
  scores_li = [[0 for j in range(num_classes)] for i in range(num_train)]  
  for i in range(num_train):
    for j in range(num_classes):
      for k in range(X.shape[1]):
        scores_li[i][j] += X[i][k] * W[k][j]  
  scores = np.array(scores_li)

  for ii in range(num_train):
    current_scores = scores[ii, :]

    shift_scores = current_scores - np.max(current_scores)

    # loss_ii = -shift_scores[y[ii]] + np.log(np.sum(np.exp(shift_scores)))
    # 문제 6-2: 위 구문(line:43)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
    sum_score = 0
    for shift_score in shift_scores:
      sum_score += math.exp(shift_score)
    loss_ii = -shift_scores[y[ii]] + math.log(sum_score)

    loss += loss_ii

    for jj in range(num_classes):
      # softmax_score = np.exp(shift_scores[jj]) / np.sum(np.exp(shift_scores))
      # 문제 6-3: 위 구문(line:43)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
      softmax_score = math.exp(shift_scores[jj]) / sum_score

      if jj == y[ii]:
        dW[:, jj] += (-1 + softmax_score) * X[ii]
      else:
        dW[:, jj] += softmax_score * X[ii]

  loss /= num_train
  # loss += reg * np.sum(W*W)
  # 문제 6-4: 위 구문(line:43)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
  loss += reg * sum(sum(W*W))

  dW /= num_train
  dW += 2*reg*W

  # (list to np array, tuple to np array 변환 함수(np.array())는 사용 가능)

    
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_train = X.shape[0]
  num_classes = W.shape[1]

  scores = np.dot(X, W)
  # shift_scores = scores - np.max(scores, axis=1)[...,np.newaxis]
  # 문제 7-1: 위 구문(line: 96)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
  max_scores_li = [0 for i in range(num_train)]
  for i, score in enumerate(scores):
    max_scores_li[i] = max(score) # 500x1
  for i in range(num_classes):
    shift_scores = scores - max_scores_li[i]

  # softmax_scores = np.exp(shift_scores)/ np.sum(np.exp(shift_scores), axis=1)[..., np.newaxis]
  # 문제 7-2: 위 구문(line: 99)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
  exp_shift_scores_li = [[0 for j in range(num_classes)] for i in range(num_train)]
  softmax_scores_li = [[0 for j in range(num_classes)] for i in range(num_train)]

  for i in range(num_train):
    for j in range(num_classes):
      exp_shift_scores_li[i][j] = math.exp(shift_scores[i][j])

  for i in range(num_train):
    for j in range(num_classes):
      softmax_scores_li[i][j] = exp_shift_scores_li[i][j] / sum(exp_shift_scores_li[i])

  softmax_scores = np.array(softmax_scores_li)  

  dScore = softmax_scores
  dScore[range(num_train),y] = dScore[range(num_train),y] - 1

  dW = np.dot(X.T, dScore)
  dW /= num_train
  dW += 2*reg*W

  correct_class_scores = np.choose(y, shift_scores.T)  # Size N vector
  # loss = -correct_class_scores + np.log(np.sum(np.exp(shift_scores), axis=1))
  # loss = np.sum(loss)
  # 문제 7-3: 위 구문(line: 110, 111)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
  loss_li = [0 for i in range(num_train)]
  for i in range(num_train):
    loss_li[i] = -correct_class_scores[i] + math.log(sum(exp_shift_scores_li[i]))
  loss = sum(loss_li)

  loss /= num_train
  # loss += reg * np.sum(W*W)
  # 문제 7-4: 위 구문(line: 110, 111)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
  loss += reg * sum(sum(W*W))

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

