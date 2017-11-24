# Machine-Learning
Implement the backpropagation algorithm for Neural Networks and test it on various real world datasets

1. Pre-processing:
Pre-processing involves checking the dataset for null or missing values, cleansing the dataset of any wrong values, standardizing the features and converting any nominal (or categorical) variables to numerical form. This step is essential before running neural net algorithm, as they can only accept numeric data and work best with scaled data.

The arguments to this part will be:
  - complete input path of the raw dataset
  - complete output path of the pre-processed dataset

The pre-processing code will read in a dataset specified using the first command line argument and first check for any null or missing values. It will remove any data points (i.e. rows) that have missing or incomplete features.

Also the following are performed,
  - If the value is numeric, it needs to be standardized, which means subtracting the mean from each of the values and dividing by the standard deviation.
  - If the value is categorical or nominal, it needs to be converted to numerical values.

2. Training a Neural Net:
The processed dataset will be used to build a neural net. The input parameters to the neural net
are as follows:
  - input dataset – complete path of the post-processed input dataset
  - training percent – percentage of the dataset to be used for training
  - maximum_iterations – Maximum number of iterations that the algorithm will run. This parameter is used so that the program terminates in a reasonable time.
  - number of hidden layers
  - number of neurons in each hidden layer
For example, input parameters could be:
ds1 80 200 2 4 2
The above would imply that the dataset is ds1, the percent of the dataset to be used for training is 80%, the maximum number of iterations is 200, and there are 2 hidden layers with (4, 2) neurons. The program would have to initialize the weights randomly. Remember to take care of the bias term (w0) also

NOTE: The Report ReportFinal.pdf contains the Analysis of the Model built for the various Datsets.
