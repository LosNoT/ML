import pandas as pd
from sklearn import datasets
import tensorflow as tf
import itertools

print ("Starting ML los no toribios inc.")

#base of the model
COLUMNS = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "Y"] # training data with x's and y values
FEATURES = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9"] #x's variables
LABEL = "Y"	# y variable
#basically: y = B + x1*a1 + x2*a2 + ... + xn*an 

def read_training_data(path_data, skipInitialspace, skiprows, columns_model):
	"""
	Read all the data collected, the data come in a csv file
	"""
	training_set = pd.read_csv(path_data, skipinitialspace = skipInitialspace, skiprows = skiprows, names =columns_model)
	return training_set

def read_test_data(path_data, skipInitialspace, skiprows, columns_model):
	"""
	Dataset to evaluate the model, the data come in a csv file
	"""
	test_set = pd.read_csv(path_data, skipinitialspace = skipInitialspace, skiprows = skiprows, names =columns_model)
	return test_set

def read_prediction_data(path_data, skipInitialspace, skiprows, columns_model):
	"""
	Test cases, the data come in a csv file
	"""
	prediction_set = pd.read_csv(path_data, skipinitialspace = skipInitialspace, skiprows = skiprows, names =columns_model)
	return prediction_set

def format_features(features_cols_input):
	feature_cols = [tf.feature_column.numeric_column(k) for k in features_cols_input]
	return feature_cols

def create_linear_estimator(feature_cols, model_directory):
	print("using linear estimator")
	estimator = tf.estimator.LinearRegressor(feature_columns=feature_cols, model_dir=model_directory)
	return estimator

def feeder_model(data_set, features_list, target_label, num_epochs=None, n_batch = 128, shuffle=True):    
	return tf.estimator.inputs.pandas_input_fn(       
		x = pd.DataFrame({k: data_set[k].values for k in features_list}),       
		y = pd.Series(data_set[target_label].values),       
		batch_size = n_batch,          
		num_epochs = num_epochs,       
		shuffle = shuffle
	)	

def train_model(training_data, features_list, target_label):
	estimator.train(
		input_fn = feeder_model(
			training_data,
			features_list,
			target_label,
			num_epochs = None,                                      
			n_batch = 128,                                      
			shuffle = True), 
		steps = 10000
	)	

def model_evaluator(evaluator_test_set, features_list, target_label):
	ev = estimator.evaluate(    
        input_fn = feeder_model(
			evaluator_test_set,
			features_list,
			target_label,
			num_epochs = 1,                          
			n_batch = 128,                          
			shuffle = True
		)
   	)
	
	loss_score = ev["loss"]
	print("Loss: {0:f}".format(loss_score))

def model_predictor(prediction_set, features_list, target_label):
	y = estimator.predict(    
         input_fn = feeder_model(
			prediction_set,
			features_list,
			target_label,
			num_epochs = 1,                          
			n_batch = 128,                          
			shuffle = True
		)
	)
	print("values predicted:", y)
	predictions = list(p["predictions"] for p in itertools.islice(y, 6))
	print("Predictions: {}".format(str(predictions)))


#define datasets
training_set = read_training_data("./data/train.csv", True, 1, COLUMNS)
test_set = read_test_data("./data/test.csv", True, 1, COLUMNS)
prediction_set = read_prediction_data("./data/predict.csv", True, 1, COLUMNS)

#format features
feature_cols = format_features(FEATURES)

#define estimator, linear in this case, if there is succeed, it will change to network flow :)
estimator = create_linear_estimator(feature_cols, "./resultsModel")

#print(training_set, test_set, prediction_set)
#print(training_set.shape, test_set.shape, prediction_set.shape)
#print(feature_cols)
print(estimator)

train_model(training_set, FEATURES, LABEL)
model_evaluator(test_set, FEATURES, LABEL)
model_predictor(prediction_set, FEATURES, LABEL)

#print(training_set['medv'].describe())
