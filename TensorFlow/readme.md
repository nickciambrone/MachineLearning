In this Jupyter Notebook, I conduct research on a real dataset from UCI machine learning repository. 
This dataset contains information about bank notes, i.e. entropy, image skew, etc. and their target class is a 0 for fraudulent of 1 for valid. 
Using the standardscalar module from sci-kit learn I scaled the data so that each predictor variable have hold equal authority in determining the target variable. 
Then, using sklearn I divided the data into a test and train set.
Then I used the functions built off the tensorflow library to create and fit a model to the data. I looped through all of the predicitons and created an array with just the prediction value (since tensorflow by default returns an object)
My model was able to achieve a 99% accuracy rate.
Modules employed: Tensorflow, SKLearn, StandardScalar, Pandas, Numpy
