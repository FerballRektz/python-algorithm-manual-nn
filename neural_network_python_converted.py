# %% [markdown]
# # I. Import libraries 

# %%
import pandas as pd
import numpy as np
import collections.abc  as carr

# %% [markdown]
# # II. EDA

# %%
initial_df :carr.MutableSequence[float] = pd.read_csv("updated_pollution_dataset.csv")
initial_df.loc[:,"Temperature"]

m,n= initial_df.shape

# %%
print("m:",m,"n:",n)

# %%
initial_df.iloc[:3,:]

# %%
initial_df.iloc[3:11,:]

# %%
# EDA
num_initial_col_df = len(initial_df.columns)
describe_initial_df = initial_df.describe()
num_na_initial_df = initial_df.isna().sum()
combined_data = [["# of variables","data stats","# of Na"],[num_initial_col_df,describe_initial_df,num_na_initial_df]]
for desc, data in zip(combined_data[0],combined_data[1]):
    print(desc + ":\n", data)

# %% [markdown]
# ## notes from data 
# - based from the data given above, the values has very large in which necitates normalization
# - the target variable is air quality as it is the one that is optimal to integer type categorization

# %% [markdown]
# # III. Data Transformation

# %%
# create copy
categorized_df = initial_df.copy(deep= True)



# %%
# data normalization and removing the target column (variable)
df_normalized = categorized_df.iloc[:,:9].transform(lambda x: (x - x.min())/ (x.max() - x.min()))
df_normalized

# %%
# data mean normalization
df_mean_norm = categorized_df.iloc[:,:9].transform(lambda x: (x - x.mean())/ (x.std()))
df_mean_norm

# %%
# categorizing the target variable and concat in two types of normalization
Air_quality_cat = initial_df.loc[:,"Air Quality"]
categorized_df_y = pd.Categorical(Air_quality_cat)
df_normalized.loc[:,"Air Quality"] = categorized_df_y.codes
df_mean_norm.loc[:,"Air Quality"] = categorized_df_y.codes

# %%
categorized_df_y

# %%
df_normalized

# %%
# test-train split function algorithm
def data_splitting_python(data :carr.MutableSequence[float], target_var :str, test_perc :float, train_perc :float, 
                        print_stats :bool = False ) -> carr.Iterable[carr.MutableSequence[float],4]:
    """
    Function that creates a split between train and test data 

    :param data: dataframe
    :type data: carr.MutableSequence[float] 
    :param target_var: y-variable (target variable) 
    :type target_var: string
    :param test_perc: test percentage (in decimal representation)
    :type test_perc: float
    :param train_perc: train percentage (in decimal representation)
    :type train_perc: float
    :param print_stats: show the data length and data of train and test data
    :type print_stats: bool 
    :return: tuple of four consisting of train and test data 
    :rtype: tuple 
    :raises ValueError: N/A

    :example:
    >>> X_train, Y_train, x_test, y_test = data_splitting_python(dataframe,"target_variable",test_perc,train_perc)
    """
    
    
    length = len(data)
    test_len = int(length * test_perc)
    train_len = int(length * train_perc)
    random_df = data.sample(n = length).reset_index(drop = True)
    
    
    train_data = random_df.iloc[:train_len,:]
    test_data = random_df.iloc[train_len:length,:]

    
    y_train = test_data.drop(columns= target_var)
    y_test =  test_data.loc[:,target_var]

    x_train = train_data.drop(columns= target_var)
    x_test =  train_data.loc[:,target_var]

    
    
    if print_stats:
        print("test length: ", test_len)
        print("test data:\n",test_data)
        print("train length: ", train_len)
        print("train data:\n",train_data)

    
        
    return x_train, x_test, y_train, y_test



X_train, Y_train, x_test, y_test = data_splitting_python(df_mean_norm,"Air Quality",.30,.70)

# in this neural netwoek, we will use the mean normalized values for x train values
X_train


# %%
# transforming from dataframe into array type 
input_matrix = np.array(X_train)
input_T = np.array(input_matrix).T

input_T

# %%
test_matrix = np.array(Y_train, dtype= int)
pred_T = test_matrix.T


pred_T

# %% [markdown]
# # IV. Creating functions for neural network algorithm

# %%
# create activation functions

def softplus(x :carr.MutableSequence[float]) -> carr.MutableSequence[float]:
    """
    Function that serves as a softmax activation function

    :param x: dataframe
    :type data: carr.MutableSequence[float]  
    :return: array of transformed data by the softplus algorithm
    :rtype: np.array 
    :raises ValueError: N/A

    :example:
    >>> softplus([0.1,0.3,0.3,0.5])
    array([0.74439666, 0.85435524, 0.85435524, 0.97407698])
    """
    return np.log(1 + np.exp(x)) # log(1+e^(x))

def ReLU(x :carr.Iterable[float]) -> carr.Iterable[float]:
    """
    Function that serves as the ReLU activation algorithm 

    :param x: dataframe
    :type data: carr.MutableSequence[float]  
    :return: array of transformed data by the ReLU algorithm
    :rtype: np.array 
    :raises ValueError: N/A

    :example:
    >>> ReLU([0.1,-0.3,0.3,-0.5,-0.5,-0.6])
    array([0.1, 0. , 0.3, 0. , 0. , 0. ])
    """
    return np.maximum(0,x) 

def deriv_softplus(x :carr.Iterable[float]) -> carr.Iterable[float]:
    """
    Function that serves for the backpropogation, the derivative of softplus algorithm

    :param x: dataframe
    :type data: carr.MutableSequence[float]  
    :return: array of transformed data by the derivative of softplus, True means 1 and False represents 0
    :rtype: np.array 
    :raises ValueError: N/A

    :example:
    >>> a= softplus([0.1,0.3,0.3,0.5])
    >>> deriv_softplus(a)
    array([0.67795654, 0.70147996, 0.70147996, 0.72593138])
    """
    return  1 / (1 + np.exp(-x)) # 1 / (1 + e^(-x))

def deriv_ReLU(x :carr.Iterable[float]) -> carr.Iterable[float]:
    """
    Function that serves for the backpropogation, the derivative of ReLU algorithm

    :param x: dataframe
    :type data: carr.MutableSequence[float]  
    :return: array of transformed data by the derivative of ReLU, True means 1 and False represents 0
    :rtype: np.array 
    :raises ValueError: N/A

    :example:
    >>> a= ReLU([0.1,-0.3,0.3,-0.5,-0.5,-0.6])
    >>> deriv_ReLU(a)
    array([ True, False,  True, False, False, False])
    """
    return x > 0

# create classification activation function
def softmax(x : carr.MutableSequence[float]) -> carr.MutableSequence[float]:
    """
    Function that serves for classification which is the softmax function

    :param x: dataframe
    :type data: carr.MutableSequence[float]  
    :return: array of transformed data that shows a sequence of floats representing percentages of possible outputs
    :rtype: np.array 
    :raises ValueError: N/A

    : NOTE:
    on the summation of exponents, before adding all the exponents, 
    the array will be first set a boundary of -1 and 1 with values in between having no changes, 
    values that excede each boundary will be converted to its respective nearest boundary
    

    :example:
    >>> softmax([0.1,0.4,0.5])
    array([0.36839031, 0.4972749 , 0.54957376])
    """
    return np.exp(x) / np.sum(np.clip(np.exp(x),a_min=-1,a_max=1))

# %%
softmax([0.1,0.4,0.5])

# %%
# make bias and weight functions
def weight_bias_creator(num_input :int , num_activation :int, 
                        num_hidden_layers :int, num_classifers :int, minmax: carr.Iterable[int,int]  = [-.5,.5]) -> carr.Iterable[float]:

        # create neural map
        neural_map :carr.Iterable[float] = list()


        # initial hidden layer weight
        weight_matrix :carr.Iterable[float] = list()
        initial_hidden =  (minmax[1] - minmax[0]) * np.random.random_sample(size= (num_activation, num_input)) +  minmax[0]
        weight_matrix.append(initial_hidden)
        # middle hidden layers if hidden layers are more than 1 (num_hidden_layers > 1)
        if num_hidden_layers > 1:
                for i in range(num_hidden_layers - 1):
                        weight_matrix.append( (minmax[1] - minmax[0]) * np.random.random_sample(size= (num_activation, num_activation)) +  minmax[0])
        # classification weights
        classification_layer = (minmax[1] - minmax[0]) * np.random.random_sample(size= (num_classifers, num_activation)) +  minmax[0]
        weight_matrix.append(classification_layer)
        neural_map.append(weight_matrix)

        
        #biases from initial until hidden layer before the classificarion
        bias_matrix :carr.Iterable[float] = list()
        for i in range(num_hidden_layers):
                bias_matrix.append( (minmax[1] - minmax[0]) * np.random.random_sample(size= (num_activation,1)) +  minmax[0])
        #classification layer bias
        bias_matrix.append((minmax[1] - minmax[0]) * np.random.random_sample(size= (num_classifers, 1)) +  minmax[0])
        neural_map.append(bias_matrix)

        return neural_map


array_map = weight_bias_creator(len(X_train.columns),8,1,4) 


# %%
array_map

# %%
# feed forward

def feed_forward (array_map: carr.MutableSequence[float], input_matrix: carr.MutableSequence[float]) -> carr.MutableSequence[float]:
    # initialize arrays
    initial_data = input_matrix
    array_edge_result :carr.MutableSequence[float] = list()
    array_activation_result :carr.MutableSequence[float] = list()
    len_layers = len(array_map[0]) # checking for last layer
    
    
    # algorithm for 
    for i,j in enumerate(array_map[0]):

        # last layer checking to put softmax rather than ReLU
        if i == len_layers-1:
            weight_temp = array_map[0][i].dot(initial_data) + array_map[1][i]
            array_edge_result.append(weight_temp)
            activation = softmax(weight_temp)
            array_activation_result.append(activation)
            continue

        # creating edge values and activation values and append
        weight_temp = array_map[0][i].dot(initial_data) + array_map[1][i]
        array_edge_result.append(weight_temp)
        activation = ReLU(weight_temp)
        array_activation_result.append(activation)
        initial_data = activation

    # combining the arrays of edges and activation functions to crete a edge-activation map
    array_map_combined = [array_edge_result,array_activation_result]


    return array_map_combined
        
edge_activation_map = feed_forward(array_map,input_T)



# %%
edge_activation_map

# %%
# onehot encoding
def one_hot(Y : carr.MutableSequence[float]) -> carr.MutableSequence[float]:
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y



# %%
one_hot(pred_T)

# %%
# backprop

def back_propogation(weight_map :carr.MutableSequence[float], activation_map :carr.MutableSequence[float],
                    X :carr.MutableSequence[float],Y : carr.MutableSequence[float]) -> carr.MutableSequence[float]:
    # encoding y value training 
    one_hot_y = one_hot(Y)

    # create necessary variables 
    len_hidden :int = len(activation_map[1]) - 1 # get final index
    weight_adj_data : carr.MutableSequence[float] = list()
    bias_adj_data : carr.MutableSequence[float] = list()
    
    # last layer
    dE = activation_map[1][len_hidden] - one_hot_y
    dW = (1/m) * (dE.dot(activation_map[1][len_hidden-1].T))
    weight_adj_data.append(dW)
    dB = (1/m) * (np.sum(dE))
    bias_adj_data.append(dB)
    
    # middle layers
    dR = weight_map[0][len_hidden].T.dot(dE) * deriv_ReLU(activation_map[0][len_hidden-1])
    index = len_hidden - 1
    while index >= 1:
        dW = (1/m) * (dR.dot(activation_map[1][index-1].T))
        weight_adj_data.append(dW)
        dB = (1/m) * (np.sum(dR))
        bias_adj_data.append(dB)
        dR_new = weight_map[0][index].T.dot(dR) * deriv_ReLU(activation_map[0][index-1])
        dR = dR_new
        index -=1
    
    #first layer 
    dW = (1/m) * (dR.dot(X.T))
    weight_adj_data.append(dW)
    dB = (1/m) * (np.sum(dR))
    bias_adj_data.append(dB)
    
    # reverse the order of both weight and bias for easier updating of values
    array_map_combined = [weight_adj_data[::-1],bias_adj_data[::-1]]


    return array_map_combined


adj_matrix = back_propogation(array_map,edge_activation_map,input_T,pred_T)

# %%
adj_matrix

# %%
# update parameters

def update_params(array_map :carr.MutableSequence[float], adj_matrix :carr.MutableSequence[float], alpha :float) -> carr.MutableSequence[float]:
    
    # initialize main and sub arrays
    len_arr = len(array_map[0]) # know the length of neural network
    new_array_map :carr.MutableSequence[float] = list()
    new_bias :carr.MutableSequence[float] = list()
    new_weight :carr.MutableSequence[float] = list()


    # updating values w/r to the learning rate 
    for index in range(len_arr):
        new_weight.append(array_map[0][index] - alpha * adj_matrix[0][index])
        new_bias.append(array_map[1][index] - alpha * adj_matrix[1][index])
    

    # combine sub arrays into the main array 
    new_array_map = [new_weight,new_bias]

    return new_array_map







new_array_map = update_params(array_map,adj_matrix,0.01)

# %%
new_array_map

# %% [markdown]
# # V.  Create neural network and accuracy models 

# %%
def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X :carr.MutableSequence[float], Y :carr.MutableSequence[float], 
                     alpha :float, iterations :int) -> carr.MutableSequence[float]:
    # initialize the creation of weights and biases w/r to the number of inputs, layers, and outputs
    temp_array = weight_bias_creator(len(X_train.columns),10,3,4) 
    len_feed :int = len(temp_array[0])

    # gradient descent algorithm 
    for i in range(iterations):
        edge_activation_map = feed_forward(temp_array,X)
        adj_matrix = back_propogation(temp_array,edge_activation_map,X,Y)
        new_array_map = update_params(temp_array,adj_matrix,alpha)
        temp_array = new_array_map
        if i % 10 == 0: # per 10 iterations, show accuracy
            print("Iteration: ", i)
            predictions = get_predictions(edge_activation_map[1][len_feed-1])
            print(get_accuracy(predictions, Y))
    return temp_array

# %%
weight_bias_model_matrix : carr.MutableSequence[float] = gradient_descent(input_T, pred_T, 0.1, 2000)

# %%
# shows the optimal values for weigths and biases 
weight_bias_model_matrix

# %% [markdown]
# # VI. Checking for accuracy of the model

# %%
# create prediction alagorithm to find accuracy of the model
def make_predictions(X :carr.MutableSequence[float], 
                     weight_bias_map :carr.MutableSequence[float]) -> carr.MutableSequence[float]:
    prediction_map = feed_forward(weight_bias_model_matrix,X)
    len_pred = len(prediction_map[0])
    predictions = get_predictions(prediction_map[1][len_pred-1])
    return predictions

# %%
dev_predictions = make_predictions(np.array(x_test).T,weight_bias_model_matrix)
get_accuracy(dev_predictions, np.array(y_test).T)

# %%
# creeating table to see the right and wrong values more clearly
x_test["y_obs"] = y_test
x_test["y_pred"] = dev_predictions.T

# %%
# a function comparing test y value to y value generated by neural network
def correct(x : carr.MutableSequence[float])-> carr.MutableSequence[float]:
    if x["y_obs"] == x["y_pred"]:
        return 1
    return 0

# %%
x_test["correct"] = x_test.apply(correct,axis=1)

# %%
x_test

# %%
# show number of correct and wrong
x_test["correct"].value_counts()

# %%
# manual computation of accuracy for double checking
x_test["correct"].value_counts()[1] / (x_test["correct"].value_counts()[0] + x_test["correct"].value_counts()[1] )


