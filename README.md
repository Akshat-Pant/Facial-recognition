# Face-recognition
Facial recognition implemented by dimensionality reduction (Principal Component and Linear Discriminant Analysis). 
Bayes Classifier and Nearest Neighbour algorithms implemented for recognition.

## PCA
PCA stands for Principal Component Analysis. It is a technique used in Machine Learning to attain dimensionality reduction.
It seeks to best represent the data. The cost function in PCA seeks to minimize the mean of squared distance between the projected vectors and the vectors in the original dimensions. 

## LDA
LDA stands for Linear Discriminant Analysis. It is a technique used in machine learning to attain dimensionality reduction.
Unlike PCA, it seeks to best discriminate the data. The cost function here seeks to minimize the within class scatter and maximize the between class scatter. 

## Bayes Classifier
Bayes classifier aims to classify samples according to the maximum posterior probablities of the classes.<br/>
According to Bayes rule, posterior= likelihood x prior / evidence  <br/>For this project, I have assumed the underlying distribution of the data to be Gaussian. Calculate the mean and covariance by Maximum Likelihood Estimation.

## KNN
KNN stands for K Nearest Neighbour. This algorithm assigns the new sample(test sample) to the majority vote of classes of the nearest K neighbours. So, if K= 3, the test sample is classified as the majority of classes of its 3 nearest neighbours.
