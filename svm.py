#################################
# Your name: Haim Petcherski
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

"""
Q4.1 skeleton.

Please use the provided functions signature for the SVM implementation.
Feel free to add functions and other code, and submit this file with the name svm.py
"""

# generate points in 2D
# return training_data, training_labels, validation_data, validation_labels
def get_points():
    X, y = make_blobs(n_samples=120, centers=2, random_state=0, cluster_std=0.88)
    return X[:80], y[:80], X[80:], y[80:]


def create_plot(X, y, clf):
    plt.clf()

    # plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.PiYG)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0] - 2, xlim[1] + 2, 30)
    yy = np.linspace(ylim[0] - 2, ylim[1] + 2, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])


def train_three_kernels(X_train, y_train, X_val, y_val):
    """
    Returns: np.ndarray of shape (3,2) :
                A two dimensional array of size 3 that contains the number of support vectors for each class(2) in the three kernels.
    """
    # TODO: add your code here
    vectors_support_list = np.zeros((3,2))
    clf = svm.SVC(C=1000,kernel='linear', degree=1)
    clf.fit(X_train, y_train)
    create_plot(X_train, y_train, clf)
    vectors_support_list[0] = clf.n_support_
    clf = svm.SVC(C=1000,kernel='poly', degree=2)
    clf.fit(X_train, y_train)
    create_plot(X_train, y_train, clf)
    vectors_support_list[1] = clf.n_support_
    clf = svm.SVC(C=1000, kernel='rbf')
    clf.fit(X_train, y_train)
    create_plot(X_train, y_train, clf)
    vectors_support_list[2] = clf.n_support_

    return vectors_support_list




def linear_accuracy_per_C(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    # TODO: add your code here
    c_array = [10**j for j in range(-5, 6)]

    accuracy_per_C_training = np.zeros(11)
    accuracy_per_C_validation= np.zeros(11)
    index_in_accuracy_per_C=-1
    for c in c_array:
        index_in_accuracy_per_C+=1
        clf = svm.SVC(C=c, kernel='linear', degree=1)
        clf.fit(X_train, y_train)
        accuracy_per_C_validation[index_in_accuracy_per_C] = np.sum(np.equal(clf.predict(X_val), y_val))/len(y_val)
        accuracy_per_C_training[index_in_accuracy_per_C] = np.sum(np.equal(clf.predict(X_train), y_train))/len(y_train)
    plt.xlabel('constant penalty C')
    plt.ylabel('prediction accuracy')
    plt.xscale('log')
    plt.grid(True)

    plt.plot(c_array, accuracy_per_C_validation, 'g', label="accuracy for validation set")
    plt.plot(c_array, accuracy_per_C_training, 'r', label="accuracy for training set")
    plt.legend()
    #plt.show()
    clf = svm.SVC(C=10**-5, kernel='linear', degree=1)
    clf.fit(X_train, y_train)
    create_plot(X_train, y_train, clf)
    #plt.show()
    clf = svm.SVC(C=1, kernel='linear', degree=1)
    clf.fit(X_train, y_train)
    create_plot(X_train, y_train, clf)
    #plt.show()
    clf = svm.SVC(C=10**5, kernel='linear', degree=1)
    clf.fit(X_train, y_train)
    create_plot(X_train, y_train, clf)
    #plt.show()
    return accuracy_per_C_validation





def rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    # TODO: add your code here
    gamma_array = [10 ** j for j in range(-5, 6)]
    accuracy_per_C_training = np.zeros(11)
    accuracy_per_C_validation= np.zeros(11)
    index_in_accuracy_per_C=-1
    for gamma in gamma_array:
        index_in_accuracy_per_C+=1
        clf = svm.SVC(C=10, kernel='rbf', gamma=gamma)
        clf.fit(X_train, y_train)
        accuracy_per_C_validation[index_in_accuracy_per_C] = np.sum(np.equal(clf.predict(X_val), y_val))/len(y_val)
        accuracy_per_C_training[index_in_accuracy_per_C] = np.sum(np.equal(clf.predict(X_train), y_train))/len(y_train)

    plt.xlabel('Gamma')
    plt.ylabel('prediction accuracy')
    plt.xscale('log')
    plt.grid(True)

    plt.plot(gamma_array, accuracy_per_C_validation, 'g', label="accuracy for validation set")
    plt.plot(gamma_array, accuracy_per_C_training, 'r', label="accuracy for training set")
    plt.legend()
    #plt.show()

    clf = svm.SVC(C=10, kernel='rbf', gamma=10**-5)
    clf.fit(X_train, y_train)
    create_plot(X_train, y_train, clf)
    #plt.show()
    clf = svm.SVC(C=10, kernel='rbf', gamma=1)
    clf.fit(X_train, y_train)
    create_plot(X_train, y_train, clf)
    #plt.show()
    clf = svm.SVC(C=10, kernel='rbf', gamma=100)
    clf.fit(X_train, y_train)
    create_plot(X_train, y_train, clf)
    #plt.show()
    return accuracy_per_C_validation

if __name__ == '__main__':
    training_data, training_labels, validation_data, validation_labels = get_points()
    train_three_kernels(training_data, training_labels, validation_data, validation_labels)
    linear_accuracy_per_C(training_data, training_labels, validation_data, validation_labels)
    rbf_accuracy_per_gamma(training_data, training_labels, validation_data, validation_labels)

