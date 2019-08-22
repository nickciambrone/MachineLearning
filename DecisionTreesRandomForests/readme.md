In this jupyter notebook, I used real data from LendingClub.com to create a model that predicts wether a person will pay their lender back. 

The dataset had many columns that defined details of the loan as well as the borrower, i.e. installment, rate, credit score. The last column stated whether or not the borrower paid back in full as a binary value. 

Using the Sci-Kit Learn RandomForestClassifier module, I was able to seperate the data, and fit a model to the training data. I then tested the model on the test data and it was able to achieve 81% accuracy when predicting whether or not the borrower paid back in full. 

