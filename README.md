"# SVM" 

In this project I used an existing implementation
of SVM: the SVC class from sklearn.svm. This class solves the soft-margin SVM problem.
By using the training data from SVC class the project
trains and comapres between 3 kernel SVM models - linear, quadratic and RBF.
This project implements two main functions:
• get points - returns training and validation sets of points in 2D, and their labels.
• create plot - receives a set of points, their labels and a trained SVM model, and
creates a plot of the points and the separating line of the model. The plot is created in
the background using matplotlib.
