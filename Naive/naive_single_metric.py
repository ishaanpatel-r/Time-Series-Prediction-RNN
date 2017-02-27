


# # #########################
# #							#
#   Imports & Declarations  #
# #							#
# # #########################

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import csv
import math
import numpy
import pandas
import itertools

# fix random seed for reproducibility
numpy.random.seed(7)





# # #########################
# #							#
#		 LSTM Neural		#
# #							#
# # #########################


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=5):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

# def to train multiple columns
def model_train_for_column(norm_set):

	# load the dataset
	dataframe = pandas.DataFrame(norm_set)
	dataset = dataframe.values
	dataset = dataset.astype('float32')

	# normalize the dataset
	scaler = MinMaxScaler(feature_range=(0, 1))
	dataset = scaler.fit_transform(dataset)

	# split into train and test sets
	train_size = int(len(dataset) * 0.5)
	test_size = len(dataset) - train_size
	train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

	# reshape into X=t and Y=t+1
	look_back = 3
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)

	# reshape input to be [samples, time steps, features]
	trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
	testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

	# create and fit the LSTM network
	batch_size = 1
	model = Sequential()
	model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
	model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(500): #500
		model.fit(trainX, trainY, nb_epoch=1, batch_size=batch_size, verbose=2, shuffle=False)
		model.reset_states()

	# make predictions
	trainPredict = model.predict(trainX, batch_size=batch_size)
	model.reset_states()
	testPredict = model.predict(testX, batch_size=batch_size)

	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([trainY])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([testY])

	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))

	# return scaler.inverse_transform(dataset), trainPredict, testPredict

	# shift train predictions for plotting
	trainPredictPlot = numpy.empty_like(dataset)
	trainPredictPlot[:, :] = numpy.nan
	trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

	# shift test predictions for plotting
	testPredictPlot = numpy.empty_like(dataset)
	testPredictPlot[:, :] = numpy.nan
	testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

	return scaler.inverse_transform(dataset), trainPredictPlot, testPredictPlot




# # #########################
# #							#
#		For all Files		#
# #							#
# # #########################

def use_files_to_generate(csv_file_name):

	# requisite lists
	trend_data_temp = []
	trend_data = []

	# read all data from csv and split to individual lists accordingly
	with open(csv_file_name, 'r') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

		for lineNum, line in enumerate(spamreader):
			column_counts = len(line)

	with open(csv_file_name, 'r') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		for i in range(1, column_counts):
			for row in spamreader:
				trend_data_temp += [row[1]]

			# remove headers
			skill_set = trend_data_temp.pop(0)
			trend_data_temp.pop(0)

			# convert each list into a float-list
			tt_2 = [float(i) for i in trend_data_temp]

			# get max values from each list
			tt2_max = max(tt_2)

			# normalize every list
			trend_data =  [x / tt2_max for x in tt_2]


			# # #############################
			# #								#
			#		Training & Plotting  	#
			# #								#
			# # #############################

			# train all models here
			data_final = model_train_for_column(trend_data)


			# plot concated results
			plt.plot(data_final[0])
			plt.plot(data_final[1])
			plt.plot(data_final[2])
			plt.savefig(skill_set + '.png', bbox_inches='tight')
			plt.show()

use_files_to_generate('../all_skill_data.csv')
