import os
import numpy as np
from tensorflow import keras
from pyrsgis import raster
from pyrsgis.convert import changeDimension
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# Change the directory
os.chdir("E:\\BuiltUpPrediction")

# Assign file names
raw2011 = 'l5_2011_raw.tif'
builtup2011 = 'l5_2011_builtup.tif'
raw2005 = 'l5_2005_raw.tif'

# Read the rasters as array
ds1, feature2011 = raster.read(raw2011, bands='all')
ds2, label2011 = raster.read(builtup2011, bands=1)
ds3, feature2005 = raster.read(raw2005, bands='all')

# Print the size of the arrays
print("2011 Multispectral image shape: ", feature2011.shape)
print("2011 Binary built-up image shape: ", label2011.shape)
print("2005 Multispectral image shape: ", feature2005.shape)

# Clean the labelled data to replace NoData values by zero
label2011 = np.clip(label2011, 0, 1)

# Reshape the array to single dimensional array
feature2011 = changeDimension(feature2011)
label2011 = changeDimension (label2011)
feature2005 = changeDimension(feature2005)
nBands = feature2011.shape[1]

print("2011 Multispectral image shape: ", feature2011.shape)
print("2011 Binary built-up image shape: ", label2011.shape)
print("2005 Multispectral image shape: ", feature2005.shape)

# Split testing and training datasets
xTrain, xTest, yTrain, yTest = train_test_split(feature2011, label2011, test_size=0.4, random_state=42)

print(xTrain.shape)
print(yTrain.shape)

print(xTest.shape)
print(yTest.shape)

# Normalise the data
xTrain = xTrain / 255.0
xTest = xTest / 255.0
feature2005 = feature2005 / 255.0

# Reshape the data
xTrain = xTrain.reshape((xTrain.shape[0], 1, xTrain.shape[1]))
xTest = xTest.reshape((xTest.shape[0], 1, xTest.shape[1]))
feature2005 = feature2005.reshape((feature2005.shape[0], 1, feature2005.shape[1]))

# Print the shape of reshaped data
print(xTrain.shape, xTest.shape, feature2005.shape)

# Define the parameters of the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(1, nBands)),
    keras.layers.Dense(14, activation='relu'),
    keras.layers.Dense(2, activation='softmax')])

# Define the accuracy metrics and parameters
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Run the model
model.fit(xTrain, yTrain, epochs=2)

# Predict for test data 
yTestPredicted = model.predict(xTest)
yTestPredicted = yTestPredicted[:,1]

# Calculate and display the error metrics
yTestPredicted = (yTestPredicted>0.5).astype(int)
cMatrix = confusion_matrix(yTest, yTestPredicted)
pScore = precision_score(yTest, yTestPredicted)
rScore = recall_score(yTest, yTestPredicted)

print("Confusion matrix: for 14 nodes\n", cMatrix)
print("\nP-Score: %.3f, R-Score: %.3f" % (pScore, rScore))

predicted = model.predict(feature2005)
predicted = predicted[:,1]

# Predict new data and export the probability raster
prediction = np.reshape(predicted, (ds3.RasterYSize, ds3.RasterXSize))
outFile = 'check_2005_BuiltupNN_predicted.tif'
raster.export(prediction, ds3, filename=outFile, dtype='float')
