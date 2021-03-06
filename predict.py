from keras.applications.vgg16 import VGG16, preprocess_input
# from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.xception import Xception, preprocess_input
# from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras import applications
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
import logging
# other imports
from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob
import h5py
import os
import json
import pickle as cPickle
import datetime
import time

from sklearn.metrics import accuracy_score
# organize imports
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

logging.basicConfig(level=logging.DEBUG)

logger.debug("Load Config file")

with open('conf/config.json') as f:
	config=json.load(f)


logger.debug("Load Model parameters")
img_width, img_height = config["size"],config["size"]
train_data_dir = config["test"]
validation_data_dir = config["val"]
batch_size = config["batch_size"]
epochs = config["epochs"]
vgg_output=config["vgg"]
inception_output=config["inception"]
xception_output=config["xception"]

# Load moedels
logger.debug("Load VGG Model")
model_vgg = applications.VGG16(weights='imagenet',include_top=False, input_shape = (img_width, img_height, 3))
logger.debug("Load Inception Model")
model_inception = applications.InceptionV3(weights='imagenet',include_top=False, input_shape = (img_width, img_height, 3))
logger.debug("Load Xception Model")
model_xception =applications.Xception(weights='imagenet',include_top=False, input_shape = (img_width, img_height, 3))

# Define Feature extractor
logger.debug("Define Feature extractors")
feat_ext_vgg=Model(input=model_vgg.input, output=model_vgg.get_layer(vgg_output).output)
feat_ext_inception=Model(input=model_inception.input, output=model_inception.get_layer(inception_output).output)
feat_ext_xception=Model(input=model_xception.input, output=model_xception.get_layer(xception_output).output)



if os.path.isdir(train_data_dir) :
	# path to training dataset
	logger.debug("Load Training Dataset")
	train_labels = os.listdir()
else:
	logger.debug("Training Directory not found")
	raise FileNotFoundError('Training Directory not found')


# variables to hold inception features and labels
inception_features = []
inception_test_features = []

# variables to hold vgg features and labels
vgg_features = []
vgg_test_features = []

# variables to hold vgg features and labels
sq_features = []
sq_test_features = []


labels   = []
test_labels =[]

image_size=(img_width, img_height)
# loop over all the labels in the folder
logger.debug("Loop over all the labels in the train folder")
for label in train_labels:
	cur_path = train_data_dir + "/" + label
	print(cur_path)
	for image_path in glob.glob(cur_path + "/*.jpg"):
		img = image.load_img(image_path, target_size=image_size)
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		feature = feat_ext_inception.predict(x)
		flat = feature.flatten()
		inception_features.append(flat)
		feature = feat_ext_vgg.predict(x)
		flat = feature.flatten()
		vgg_features.append(flat)
		feature = feat_ext_xception.predict(x)
		flat = feature.flatten()
		x_features.append(flat)
		labels.append(label)

# loop over all the labels in the folder
logger.debug("Loop over all the labels in the validation folder")
for label in train_labels:
	cur_path = validation_data_dir + "/" + label
	for image_path in glob.glob(cur_path + "/*.jpg"):
		img = image.load_img(image_path, target_size=image_size)
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		feature = feat_ext_inception.predict(x)
		flat = feature.flatten()
		inception_test_features.append(flat)
		feature = feat_ext_vgg.predict(x)
		flat = feature.flatten()
		vgg_test_features.append(flat)
		feature = feat_ext_xception.predict(x)
		flat = feature.flatten()
		x_test_features.append(flat)
		test_labels.append(label)


# encode the labels using LabelEncoder
logger.debug("Encode the labels using LabelEncoder")
targetNames = np.unique(labels)
le = LabelEncoder()
le_labels = le.fit_transform(labels)
test_labels=le.transform(test_labels)

logger.debug("Concatenate Features ")
features = np.concatenate( [np.asarray(vgg_features), np.asarray(inception_features), np.asarray(x_features)], axis=1 )
test_features = np.concatenate( [np.asarray(vgg_test_features), np.asarray(inception_test_features),np.asarray(x_test_features)], axis=1)
seed=config["seed"]

# use logistic regression as the model
print("[INFO] creating Logistic Regression...")
logger.debug("Creating Logistic Regression ")
clf = LogisticRegression(random_state=seed)
clf.fit(features, le_labels)
# evaluate the model of test data
logger.debug("Evaluate the model of test data")
preds = clf.predict(test_features)
print(classification_report(le.inverse_transform(preds),le.inverse_transform(test_labels)))
print("Accuracy :",accuracy_score(le.inverse_transform(preds),le.inverse_transform(test_labels)))
logger.debug(classification_report(le.inverse_transform(preds),le.inverse_transform(test_labels)))
logger.debug("Accuracy :",accuracy_score(le.inverse_transform(preds),le.inverse_transform(test_labels)))

# Xgboost
import xgboost as xgb

# Prepare the inputs for the model
train_X = np.asmatrix(features)
test_X = np.asmatrix(test_features)
#train_y = train_df['Survived']

# You can experiment with many other options here, using the same .fit() and .predict()
# methods; see http://scikit-learn.org
# This example uses the current build of XGBoost, from https://github.com/dmlc/xgboost
print("[INFO] creating XGBoost...")
logger.debug("Creating XGBoost")
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, le_labels)
preds = gbm.predict(test_X)

print(classification_report(le.inverse_transform(preds),le.inverse_transform(test_labels)))
print("Accuracy :",accuracy_score(le.inverse_transform(preds),le.inverse_transform(test_labels)))
logger.debug(classification_report(le.inverse_transform(preds),le.inverse_transform(test_labels)))
logger.debug("Accuracy :",accuracy_score(le.inverse_transform(preds),le.inverse_transform(test_labels)))

if config["pca"]:
	# Apply PCA
	logger.debug("Apply PCA")
	from sklearn.decomposition import PCA
	logger.debug("Concatenate Test and Train feature set")
	X=np.concatenate( [features,test_features], axis=0 )
	logger.debug("Create PCA with {} of principal components".format(config["n_components"]))
	pca = PCA(n_components=config["n_components"])
	pca.fit(X)

	train_x = pca.transform(features)
	test_x = pca.transform(test_features)

	# Apply Logistic Regression with PCA
	print("[INFO] creating Logistic Regression with PCA...")
	logger.debug("Creating Logistic Regression with PCA")
	clf = LogisticRegression(random_state=seed)
	clf.fit(train_x, le_labels)

	# evaluate the model of test data
	preds = clf.predict(test_x)
	print(classification_report(le.inverse_transform(preds),le.inverse_transform(test_labels)))
	print("Accuracy :",accuracy_score(le.inverse_transform(preds),le.inverse_transform(test_labels)))
	logger.debug(classification_report(le.inverse_transform(preds),le.inverse_transform(test_labels)))
	logger.debug("Accuracy :",accuracy_score(le.inverse_transform(preds),le.inverse_transform(test_labels)))
	# Apply Xgboost with PCA
	# Prepare the inputs for the model
	train_X = np.asmatrix(train_x)
	test_X = np.asmatrix(test_x)

	# You can experiment with many other options here, using the same .fit() and .predict()
	# methods; see http://scikit-learn.org
	# This example uses the current build of XGBoost, from https://github.com/dmlc/xgboost
	print("[INFO] creating XGBoost with PCA...")
	logger.debug("Creating XGBoost with PCA")
	gbm = xgb.XGBClassifier(max_depth=config["xgb_max_dept"], n_estimators=config["xgb_n_estimator"], \
					learning_rate=config["xgb_learning_rate"]).fit(train_X, le_labels)
	preds = gbm.predict(test_X)

	print(classification_report(le.inverse_transform(preds),le.inverse_transform(test_labels)))
	print("Accuracy :",accuracy_score(le.inverse_transform(preds),le.inverse_transform(test_labels)))
	logger.debug(classification_report(le.inverse_transform(preds),le.inverse_transform(test_labels)))
	logger.debug("Accuracy :",accuracy_score(le.inverse_transform(preds),le.inverse_transform(test_labels)))
