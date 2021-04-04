# [Module 13_Challenge_Submission](https://github.com/sfkonrad/M13_Challenge_Submission/blob/main/M13_Challenge_Submision/M13_Challenge_KonradK_venture_funding_with_deep_learning.ipynb)

##### Konrad Kozicki
### UCB-VIRT-FIN-PT-12-2020-U-B-TTH
---

# Venture Funding with Deep Learning


---
---






For this assignment, we've been hired as a risk management associates at Alphabet Soup, a venture capital firm. Alphabet Soup’s business team receives many funding applications from startups every day. This team has asked us to help them create a model that predicts whether applicants will be successful if funded by Alphabet Soup.

The business team has provided us with a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. With our knowledge of machine learning and neural networks, we nede to decide which features to use in the provided dataset to create a binary classifier model that will predict whether an applicant will become a successful business. The CSV file contains a variety of information about these businesses, including whether or not they ultimately became successful.


The steps for this challenge are broken out into the following sections:

* Prepare the data for use on a neural network model.

* Compile and evaluate a binary classification model using a neural network.

* Optimize the neural network model.

---


### Prepare the Data for Use on a Neural Network Model 

Using our knowledge of Pandas and scikit-learn’s `StandardScaler()`, we preprocessed the dataset in order for us to use it to compile and evaluate the neural network model.

Using the corresponding starter code file, we completed the following data preparation steps:

1. Read in the `applicants_data.csv` file into a Pandas DataFrame. Reviewed the DataFrame, looking for categorical variables that will need to be encoded, as well as columns that could eventually define our features and target variables.   

2. Dropped the “EIN” (Employer Identification Number) and “NAME” columns from the DataFrame, as they are not relevant to the binary classification model.
 
3. Encoded the dataset’s categorical variables using `OneHotEncoder`, and then placed the encoded variables into a new DataFrame.

4. Add the original DataFrame’s numerical variables to the DataFrame containing the encoded variables.

> **📝 Note** 
> 
> To complete this step, we employed the Pandas `concat()` function. 

5. Using the preprocessed data, we created the features (`X`) and target (`y`) datasets. The target dataset was defined by the preprocessed DataFrame column “IS_SUCCESSFUL”. The remaining columns are defined by the features dataset. 

6. Split the features and target sets into training and testing datasets.

7. Employed scikit-learn's `StandardScaler` to scale the features data.


---

### Compile and Evaluate a Binary Classification Model Using a Neural Network

We proceeded to use our knowledge of TensorFlow to design a binary classification deep neural network model. This model encodes the dataset’s features to predict whether an Alphabet Soup funded startup will be successful based on the features in the dataset. We considered the number of inputs before determining the number of layers that our model will contain or the number of neurons on each layer. Next, we compiled and fit our model. Finally, we evaluated our binary classification model by calculated the models' loss and accuracy. 
 
To do so, we completed the following steps:

1. Created a deep neural network by assigning the number of input features, the number of layers, and the number of neurons on each layer using Tensorflow’s Keras.

2. Compiled and fit the Original Model using the `binary_crossentropy` loss function, the `adam` optimizer, and the `accuracy` evaluation metric.

3. Evaluated the model using the test data to determine the model’s loss and accuracy.

4. Saved and export our model to an HDF5 file, and name the file `AlphabetSoup.h5`. 



---

### Optimize the Neural Network Model

Using our knowledge of TensorFlow and Keras, we proceeded to optimize our model to improve the model's accuracy. Even if we do not successfully achieve a better accuracy, we were required to demonstrate at least two attempts to optimize the model. We included these attempts in our existing notebook as [Alternative Models](). We also made copies of the starter notebook in the same folder, renamed them, and coded models for optimization in supplemental notebooks. 

    > **Note** we did not have points deducted if our models don't achieve an improved accuracy, as long as we made at least two [attempts to optimize]() the model.

To do so, we completed the following steps:

1. Defined at least three new deep neural network models (the original plus 2 optimization attempts). With each, attempting to improve on our first model’s predictive accuracy.

    > **📝 Note** 
    >  
    > Recall that perfect accuracy has a value of 1, therefore accuracy improves as its value moves closer to 1. To optimize our model for a predictive accuracy as close to 1 as possible, we used all of the following techniques:
>
> * Adjust the input data by dropping different features columns to ensure that no variables or outliers confuse the model.
>
> * Add more neurons (nodes) to a hidden layer.
>
> * Add more hidden layers.
>
> * Use different activation functions for the hidden layers.
>
> * Add to or reduce the number of epochs in the training regimen.

2. After completing our models, we display the accuracy scores achieved by each model, and compare the results.

3. Saved each of [our models as an HDF5]() file.



