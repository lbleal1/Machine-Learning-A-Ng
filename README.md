# Machine Learning by Andrew Ng 
![Scholarship Email](/assets/ml-cert.png)

This repository contains my answers, in code form and with explanation in pdf form, for the programming assignments in this course. The programming language used is MATLAB.

## Exercises 
The structures below show a bird's-eye view of how I made each pdf report.

### Exercise 1: Linear Regression
The exercise covered and implemented Linear Regression with one variable to predict profits for a food truck. 
The data contain profits and populations from the cities.

1 Defining the problem and dataset <br>
2 Exploring the data <br>
3 Gradient Descent <br>
&nbsp;&nbsp; 3.1 Update Equations <br>
&nbsp;&nbsp; 3.2 Implementation <br>
&nbsp;&nbsp; 3.3 Computing the Cost <br>
&nbsp;&nbsp; 3.4 Gradient Descent <br>
4 Visualizations <br>

### Exercise 2: Logistic Regression
The exercise covered logistic regression applied to two different datasets. The first dataset was used to create a model that will help predict whether a student gets admitted into a university. The second dataset was used to explore the concept of regularization and predict whether microchips from a fabrication plant pass the quality assurance (QA).

1 Logistic Regression <br>
&nbsp;&nbsp; 1.1 Challenge <br>
&nbsp;&nbsp; 1.2 Visualizing the data <br>
&nbsp;&nbsp; 1.3 Implementation <br>
&nbsp;&nbsp;&nbsp;&nbsp; 1.3.1 Hypothesis and Sigmoid Function <br>
&nbsp;&nbsp;&nbsp;&nbsp; 1.3.2 Cost Function and Gradient of the Cost <br>
&nbsp;&nbsp;&nbsp;&nbsp; 1.3.3 Learning parameters using fminunc <br>
&nbsp;&nbsp;&nbsp;&nbsp; 1.3.4 Evaluating logistic regression <br>
2 Regularized Logistic Regression <br>
&nbsp;&nbsp; 2.1 Challenge <br>
&nbsp;&nbsp; 2.2 Visualizing the data <br>
&nbsp;&nbsp; 2.3 Feature Mapping <br>
&nbsp;&nbsp; 2.4 Cost Function and Gradient <br>
&nbsp;&nbsp; 2.5 Plotting the decision boundary <br>

### Exercise 3: Multi-class Classification and Neural Networks
The exercise covered the problem of multi-class classification in recognizing hand-written digits and the implementation of the solution using one-vs-all logistic regression and neural networks(feedforward only). 

1 Multi-class Classification <br>
&nbsp;&nbsp; 1.1 Challenge <br>
&nbsp;&nbsp; 1.2 Dataset <br>
&nbsp;&nbsp; 1.3 Visualizing the Data <br>
&nbsp;&nbsp; 1.4 Vectorizing Logistic Regression <br>
&nbsp;&nbsp;&nbsp;&nbsp; 1.4.1 Vectorizing the cost function <br>
&nbsp;&nbsp;&nbsp;&nbsp; 1.4.2 Vectorizing the gradient <br>
&nbsp;&nbsp;&nbsp;&nbsp; 1.4.3 Vectorizing regularized logistic regression <br>
&nbsp;&nbsp; 1.5 One-vs-all Classification <br>
&nbsp;&nbsp;&nbsp;&nbsp; 1.5.1 One-vs-all Training <br>
2 One-vs-all VS Neural Nets <br> 
3 Neural Networks <br>
&nbsp;&nbsp; 3.1 Model Representation <br>
&nbsp;&nbsp; 3.2 Feedforward Propagation and Prediction <br>

### Exercise 4: Neural Networks Learning
The exercise covered the implementation of the backpropagation algorithm for neural networks applied to the task of handwritten digit recognition. 

1 Neural Networks <br>
&nbsp;&nbsp; 1.1 Visualizing the data <br>
&nbsp;&nbsp; 1.2 Model Presentation <br>
&nbsp;&nbsp; 1.3 Feedforward and cost function <br>
&nbsp;&nbsp; 1.4 Regularized cost function <br>
2 Backpropagation <br>
&nbsp;&nbsp; 2.1 Sigmoid gradient <br>
&nbsp;&nbsp; 2.2 Random Initialization <br>
&nbsp;&nbsp; 2.3 Backpropagation <br>
&nbsp;&nbsp; 2.4 Regularized Neural Networks <br>

### Exercise 5: Regularized Linear Regression and Bias VS Variance
The exercise covered the implementation of regularized linear regression and the study of different bias-variance properties.

1 Regularized Linear Regression <br>
&nbsp;&nbsp; 1.1 Visualizing the dataset <br>
&nbsp;&nbsp; 1.2 Regularized linear regression's cost function <br>
&nbsp;&nbsp; 1.3 Regularized linear regression's gradient <br>
&nbsp;&nbsp; 1.4 Fitting linear regression <br>
2 Bias-variance <br>
&nbsp;&nbsp; 2.1 Learning curves <br>
3 Polynomial Regression <br>
&nbsp;&nbsp; 3.1 Learning Polynomial Regression <br>
&nbsp;&nbsp; 3.2 Selecting lambda using a cross validation set <br>

### Exercise 6: Support Vector Machines
The exercise covered the implementation of Support Vector Machines (SVM) to build a spam classifier.

1 Support Vector Machines <br>
&nbsp;&nbsp; 1.1 Example Dataset 1 <br>
&nbsp;&nbsp; 1.2 SVM with Gaussian Kernels <br>
&nbsp;&nbsp;&nbsp;&nbsp; 1.2.1 Gaussian Kernel <br>
&nbsp;&nbsp;&nbsp;&nbsp; 1.2.2 Example Dataset 2 <br>
&nbsp;&nbsp;&nbsp;&nbsp; 1.2.3 Example Dataset 3 <br>
2 Spam Classification <br>
&nbsp;&nbsp; 2.1 Preprocessing Emails <br>
&nbsp;&nbsp; 2.2 Extracting Features from Emails <br>
&nbsp;&nbsp; 2.3 Training SVM for Spam Classification <br>
&nbsp;&nbsp; 2.4 Top predictors for Spam <br>

### Exercise 7: K-Means Clustering and Principal Component Analysis
The exercise covered the implementation of K-Means Algorithm and its application to image compression and Principal Component Analysis (PCA) to find low-dimensional representation of face images.

1 K-Means Clustering <br>
&nbsp;&nbsp; 1.1 Implementing K-Means <br>
&nbsp;&nbsp;&nbsp;&nbsp; 1.1.1 Finding closest centroids (Cluster Assignment Step) <br>
&nbsp;&nbsp;&nbsp;&nbsp; 1.1.2 Computing centroid means <br>
&nbsp;&nbsp; 1.2 K-means on example dataset <br>
&nbsp;&nbsp; 1.3 Random Initialization <br>
&nbsp;&nbsp; 1.4 Image Compression with K-means <br>
&nbsp;&nbsp;&nbsp;&nbsp; 1.4.1 K-means on pixels <br>
2 Principal Component Analysis <br>
&nbsp;&nbsp; 2.1 Example Dataset <br>
&nbsp;&nbsp; 2.2 Implementing PCA <br>
&nbsp;&nbsp; 2.3 Dimensionality Reduction with PCA <br>
&nbsp;&nbsp;&nbsp;&nbsp; 2.3.1 Projecting the data onto the principal components <br>
&nbsp;&nbsp;&nbsp;&nbsp; 2.3.2 Reconstructing an approximation of the data <br>
&nbsp;&nbsp;&nbsp;&nbsp; 2.3.3 Visualizing the projections <br>
&nbsp;&nbsp; 2.4 Face Image Dataset <br>
&nbsp;&nbsp;&nbsp;&nbsp; 2.4.1 PCA on Faces <br>
&nbsp;&nbsp;&nbsp;&nbsp; 2.4.2 Dimensionality Reduction <br>

### Exercise 8: Anomaly Detection and Recommender Systems
The exercise covered an Anomaly Detection algorithm and its application in detecting failing service on a network, and Collaborative Filtering in building a recommender system for movies.

1 Anomaly Detection <br>
&nbsp;&nbsp; 1.1 Gaussian distribution <br>
&nbsp;&nbsp; 1.2 Estimating parameters for a Gaussian <br>
&nbsp;&nbsp; 1.3 Selecting the threshold <br>
&nbsp;&nbsp; 1.4 High dimensional dataset <br>
2 Recommender Systems <br>
&nbsp;&nbsp; 2.1 Movie ratings dataset <br>
&nbsp;&nbsp; 2.2 Collaborative filtering learning algorithm <br>
&nbsp;&nbsp;&nbsp;&nbsp; 2.2.1 Collaborative filtering cost function <br>
&nbsp;&nbsp;&nbsp;&nbsp; 2.2.2 Collaborative filtering gradient <br>
&nbsp;&nbsp;&nbsp;&nbsp; 2.2.3 Regularized cost function <br>
&nbsp;&nbsp;&nbsp;&nbsp; 2.2.4 Regularized gradient <br>
