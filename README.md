# Image Classification - Machine Learning Algorithms from Scratch 

<p align = "justify">The code file <a href="https://github.com/lalithavadlamani/ML-Image-Classification/blob/main/Image%20Classification.ipynb"> Image classification.ipynb</a> has machine learning models built from scratch, i.e. without using pre-made models or advanced libraries, to classify images.
</p>

## Introduction 
<p align = "justify">
The aim of the study is to build a classifier to classify 28x28 grayscale images into their respective categories. The images belong to one of the ten categories: T-shirt/Top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag and Ankle boot.</p>
<p align = "justify">Classification is a Supervised Machine learning and data mining technique used to predict group membership for data instances. It predicts categorical labels and is the task of generalizing known structure to apply to new data. (Tran, 2020) </p>
<p align = "justify">Importance of the Classification study is it allows to identify, group and properly name objects or instances since it is the process of organizing data by the categories. The study is based on the features that the objects are made of and the similarities and differences between them. Furthermore, data classification enables more efficient use of the data. </p> 

## Pre-Processing 
<p align = "justify"> The input data is first standardized using (𝑥−𝜇(𝑥))/𝜎(𝑥) . It ensures uniformity of the features and makes sure all features contribute equally to the analysis and prevents biased results that might occur due to the varying differences between the ranges of the variables. It centres the data with the corresponding mean and scales the values to between 0 and 1 retaining the importance. Since the number of features is large the training time will be high so feature reduction technique PCA has been used to reduce the number of features. Principal Component analysis (PCA) is a dimensionality reduction method that reduces the variables or features of the data by transforming large set of variables into smaller components while preserving as much information as possible. Smaller data makes the analysis process easier and faster. For the study PCA has been carried out using the eigen value decomposition. (Jaadi, 2019) </p>

## Classifier Methods experimented 
<p align = "justify"> 1. <strong>K-Nearest Neighbours</strong> – KNN is a non-parametric supervised learning model which classifies the data points based on the points that are most similar to it. For a point to be classified it calculates the distance between the point and to each other point in the data set present. These distances are to be sorted and it takes the ‘k’ number of distances and the corresponding points come as k nearest neighbours to the data point. The final class of the data point is the mode of the classes of the nearest neighbours (i.e. the class most common among its k nearest neighbours). KNN is a lazy learning algo as it doesn’t build models explicitly. There is little training involved and classifying unknown records is relatively expensive. (Schott, 2019) (Chatterjee, 2020) </p>

<p align = "justify"> 2. <strong>Naive Bayes</strong> – Naïve Bayes is a probabilistic framework built on the principle of Bayes Theorem 𝑃(𝐴|𝐵)= 𝑃(𝐵|𝐴)𝑃(𝐴)/𝑃(𝐵). It’s called Naïve as it assumes independence among the attributes/features when a class is given.
For the study the predictors are continuous instead of discrete hence Gaussian distribution is used to calculate the probability function.
  
![](https://github.com/lalithavadlamani/ML-Image-Classification/blob/main/formula.PNG) </p>
<p align = "justify"> Data is first segmented according to each class and the mean and variance are calculated of the data points in each class. The probability of data given a class is then calculated using the above formula. This becomes the likelihood probability. Calculate the likelihood probabilities for all the data points and classes. Probability of class in the dataset given becomes the prior probability. Calculate the prior probability for all the classes. Multiply the prior probability and likelihood probability to get the probabilities of a data point belonging to each class. The output class is the respective class of the highest probability taken from the probabilities. (Tran, 2020) </p>

<p align = "justify"> 3. <strong>Logistic Regression</strong> – Logistic regression is a binary classification model which uses sigmoid function to predict output. The probabilities calculated using sigmoid function are threshold at 0.5. Linear decision boundary is induced where if the probability is greater than 0.5 it belongs to that class else not. For estimating the parameters (weights) of a logistic regression model, we use negative log likelihood and optimization function or gradient descent. The loss is to be minimized and weights are to be optimized for that. For the study since it’s a multiclass problem we use the one vs rest scheme i.e. for each class we do a binary logistic classifier and take the highest possibility of a class for a data point. (Kumar, 2018) (Pant, 2019) </p>

## Experiments 

<div align="justify">
  
1. <strong>KNN</strong> -  For KNN the hyperparameter ‘k’ (the number of neighbours) is experimented. 10-fold Cross validation has been used to obtain the efficient training accuracy without any bias. The highest training accuracy has been observed at k=6. Hence for predictions with the testing data k-value has been chosen to be 6.
2. <strong>Naive Bayes</strong> - For Naïve Bayes there is no hyperparameter tuning involved since it’s calculating probabilities according to the data and calculating the final class based on the probability using Bayes theorem. 10-fold Cross validation has been used to get the efficient training accuracy without any bias involved.
3. <strong>Logistic Regression</strong> - For Logistic regression hyperparameter is the weights. Optimal weights are to be found by minimizing the loss for the model. This can be done using gradient descent or optimization function. In this study optimization functions from scipy optimize have been used. Observations were as below:
   * minimize: In scipy.optimize.minimize various methods like ‘Nelder Mead’,’Powell’,’CG’,BFGS’ have been tried but the optimized value was not coming within the range of the number of function evaluations and there was precision error along with the warning ‘Maximum number of function evaluations has been exceeded’. Might be due to the reason that it does local optimization.
   * fmin_tnc: The weights were not being optimizing due to the precision and the parameters xtol and ftol were difficult to estimate by trial and error for this study.
   * fmin: Weight optimization was not being achieved and it was being terminated with the error message: Maximum number of function evaluations has been exceeded. There is an option to set the parameter maxfun but it was demanding to find out the value for this study using trial and error.
   * fmin_cobyla: Weight optimization has been successful by using fmin_cobyla from scipy.optimize. It has done constrained optimization on the gradient and returned the optimal weights by minimizing the loss. Hence this has been selected and used for the optimization and fitting the model.
</div>

### Training Results
<p align = "justify"> 
  
![](https://github.com/lalithavadlamani/ML-Image-Classification/blob/main/results.PNG)
</p>

## Analysis and Observations 
<div align="justify">
  
* After pre-processing the data which includes standardizing and PCA to reduce the dimensions by retaining most information and important features the reduced number of features come to be 187. Data has reduced from (30000,784) to (30000,187).
* KNN is a lazy learning algorithm. The accuracy of the data depends on the quality of the data, hence though the training accuracy is around 85%, testing accuracy for the 2000 samples from the testing data is around 73%.Predictions fairly depends on the features of the data and the scale of the data used. Also, it can be computationally expensive since it has to store all the training data and use it for classification. Since the dimension of the data is quite large it took time for predictions as it had to calculate the distances between each data row with all the other rows present. 
* Naïve Bayes took less time since it requires small amount of training data to classify the test data as it is just calculating respective probabilities and using them to estimate. But the accuracy observed was less than that of KNN in the study, though theoretically Naïve Bayes is expected to predict better than KNN. Reason for this might be the high dimension of the data and the assumption that the features are independent. And Naïve Bayes is a high bias/low variance classifier. There might be some features which are mutually dependent on each other in the data used for the study. It has also been observed that Naïve Bayes predicts better if there are more features involved i.e. if there is no pre-processing involved.
* Logistic Regression has provided great training accuracy and is efficient compared to the other two algorithms in the study. Training time is also quite efficient because of the probabilistic interpretation. The optimised weights returned by the optimization function has the inference about how much each feature is important. The optimal weights have positive and negative values implying it gives the direction along with the magnitude. Interpretation of weights is easy. Hence the relationship is clear, and it helps in accurately predicting. Since it’s giving out efficient results, we can infer that the features may be separated linearly. Since the data has more dimensions the model might overfit and hence the testing accuracy is not as good as training accuracy. Furthermore, since PCA has been used as a pre-processing step implies that the important features have been used to build the model and hence the predictions are nearly accurate.
</div>

## Conclusion 

<p align = "justify"> 
Based on the Experiments done, Results and analysis above the 3 algorithms have been evaluated and compared based on 3 factors: Predictive power, Calculation time and ease to interpret output. The order of preference with respect to this study is summarized in the table below (1-high to 3-low):
  
![](https://github.com/lalithavadlamani/ML-Image-Classification/blob/main/evaluation.PNG) </p>
<p align = "justify"> Since Logistic regression has high predictive power, easy to interpret output and the time is also considerably less, we can conclude that Logistic Regression is a good performance classifier for this study and is quite efficient and accurate.
</p>

### Future work and Improvements 
<div align="justify">
  
* Naïve Bayes can be worked on by assigning weights to features and then calculating probabilities and see if the performance is improving.
* Ensemble learning can be tried using Bagging and Boosting for improving the performance further according to the variance and bias.
* Random Forest can be used for the classification and the results can be analysed
</div>
