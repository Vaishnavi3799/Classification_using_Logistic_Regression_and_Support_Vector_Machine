import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn import svm
import matplotlib.pyplot as plt


def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    X = np.concatenate((np.ones((train_data.shape[0],1)), train_data), axis=1)
    y = labeli
    w = initialWeights.reshape((n_features + 1),1)  
    
    theta = sigmoid(np.dot(X,w))

    term1 = np.multiply(y, np.log(theta))
    term21 = (1.0 - y)
    term22 = np.log(1.0 - theta)
    term2 = np.multiply(term21, term22)
    sum_term = np.add(term1, term2)
    summation_term = np.sum(sum_term)
    error = (- 1.0) * (summation_term  / n_data)

    grad_term11 = np.subtract(theta, y)
    grad_term1 = np.multiply(grad_term11, X)
    error_grad = np.sum(grad_term1, axis=0) / n_data


    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    X = np.concatenate((np.ones((data.shape[0],1)), data), axis=1)
    
    p = sigmoid(np.dot(X,W))
    
    pred = np.argmax(p,axis=1)
    
    label = np.reshape(pred, ((data.shape[0]),1))


    return label

def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights_b: the weight vector of size (D + 1) x 10
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    n_class = 10
    train_data, labeli = args
    n_data    = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    X = np.concatenate((np.ones((n_data,1)), train_data), axis=1)
    
    W = np.reshape(params,(n_feature + 1, n_class))
    
    numer = np.exp(np.dot(X,W))
    denom = np.sum(numer,axis=1)
    denom = np.reshape(denom,(denom.shape[0],1))
    theta = numer/denom

    term11 = np.multiply(Y,np.log(theta))
    sum_term = np.sum(term11)
    error = (-1) * (np.sum(sum_term))

    grad_term11 = np.transpose(X)
    grad_term12 = (theta - labeli)
    error_grad = np.dot(grad_term11, grad_term12)
    error_grad = error_grad.ravel()


    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))
    row   = data.shape[0]

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    X = np.concatenate((np.ones((row,1)), data), axis=1)

    term11 = np.dot(X,W)
    t = np.sum(np.exp(term11),axis=1)    
    t = np.reshape(t,(t.shape[0],1))
    
    theta = np.exp(term11)/t
    
    label = np.argmax(theta,axis=1)
    label = label.reshape(row,1)

    return label


def class_error_clac(predicted_label, data_label):

  n_class = 10
  i = 0
  while i < 10:
    class_predicted_label = predicted_label[np.where(data_label==np.array([i]))]
    class_data_label = data_label[np.where(data_label==np.array([i]))]
    float_comparison = (class_predicted_label == class_data_label).astype(float)
    print('Training set Accuracy for class:' + str(i) + '  ' + str(100 * np.mean(float_comparison)) + '%')
    i += 1


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')
class_error_clac(predicted_label, train_label)

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')
class_error_clac(predicted_label, validation_label)

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
class_error_clac(predicted_label, test_label)

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################

# Random Selection of Samples
rand_index = np.random.randint(50000, size = 10000)
rand_sample_data = train_data[rand_index,:]
rand_sample_label = train_label[rand_index,:]

linear_ker = svm.SVC(kernel='linear')
linear_ker.fit(rand_sample_data, rand_sample_label)

print('---------Linear Kernel---------')
print('Training Accuracy: ' + str(100 * linear_ker.score(train_data, train_label)))
print('Validation Accuracy: ' + str(100 * linear_ker.score(validation_data, validation_label)))
print('Testing Accuracy: ' + str(100 * linear_ker.score(test_data, test_label)))

rbf_ker = svm.SVC(kernel='rbf', gamma = 1.0)
rbf_ker.fit(rand_sample_data, rand_sample_label)

print('---------RBF Kernel with Gamma = 1---------')
print('Training Accuracy: ' + str(100 * rbf_ker.score(rand_sample_data, rand_sample_label)))
print('Validation Accuracy: ' + str(100 * rbf_ker.score(validation_data, validation_label)))
print('Testing Accuracy: ' + str(100 * rbf_ker.score(test_data, test_label)))

rbf_ker_def = svm.SVC(kernel='rbf', gamma = 'auto')
rbf_ker_def.fit(rand_sample_data, rand_sample_label)

print('---------RBF Kernel with Gamma = default---------')
print('Training Accuracy: ' + str(100 * rbf_ker_def.score(train_data, train_label)))
print('Validation Accuracy: ' + str(100 * rbf_ker_def.score(validation_data, validation_label)))
print('Testing Accuracy: ' + str(100 * rbf_ker_def.score(test_data, test_label)))


accuracy = np.zeros((11,3), float)
C_val = np.array([1])
C_val = np.append(C_val,np.arange(10, 101, 10))
i = 0

# iterating C values
for c in C_val:
    print("C Value: \n", c)
    rbf_mod2 = svm.SVC(kernel = 'rbf', C = c)
    rbf_mod2.fit(rand_sample_data, rand_sample_label.ravel())
    if i <= 10:

        accuracy[i][0] = 100 * rbf_mod2.score(train_data, train_label)
        accuracy[i][1] = 100 * rbf_mod2.score(validation_data, validation_label)
        accuracy[i][2] = 100 * rbf_mod2.score(test_data, test_label)
        
        print('---------RBF Kernel with Gamma = default and C = '+ str(c) +'---------')
        print('Training Accuracy: ' + str(accuracy[i][0]))
        print('Validation Accuracy: ' + str(accuracy[i][1]))
        print('Testing Accuracy: ' + str(accuracy[i][2]))

    i = i + 1


rbf_full = svm.SVC(kernel = 'rbf', gamma = 'auto', C = 10)
rbf_full.fit(train_data, train_label.ravel())

print('----------RBF with all training data and optimal C------------')
print('Training Accuracy: ' + str(100 * rbf_full.score(train_data, train_label)))
print('Validation Accuracy: ' + str(100 * rbf_full.score(validation_data, validation_label)))
print('Testing Accuracy: ' + str(100 * rbf_full.score(test_data, test_label)))


plt.figure(figsize=(16,12))
plt.title('Accuracy vs C',pad=10,fontsize=20,fontweight = 'bold')

plt.xlabel('Value of C', labelpad=20, weight='bold', size=15)
plt.ylabel('Accuracy',   labelpad=20, weight='bold', size=15)

plt.xticks( np.array([1,  10,  20,  30,  40,  50,  60,  70,  80,  90, 100]), fontsize=15)
plt.yticks( np.arange(85,100, step=0.5),  fontsize=15)


plt.plot(C_val, accuracy[:,0], color='g')
plt.plot(C_val, accuracy[:,1], color='b')
plt.plot(C_val, accuracy[:,2], color='r')

plt.legend(['Training_Data','Validation_Data','Test_Data'])


"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')
class_error_clac(predicted_label_b, train_label)

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')
class_error_clac(predicted_label_b, validation_label)


# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
class_error_clac(predicted_label_b, test_label)
