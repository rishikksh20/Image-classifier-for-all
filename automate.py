from keras import applications
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json

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
import h5py
from sklearn.metrics import accuracy_score
# organize imports
import logging
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import os
import glob
import pathlib
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
models=[]
model_names=[]
json_dir_name = config["model_dir"]
weight_name=""
json_pattern = os.path.join(json_dir_name,'*.json')
file_list = glob.glob(json_pattern)

logger.debug("Load JSON Models")
for file in file_list:
	logger.debug("Json file path :",file)
    # print("Json file path :",file)
    # load json and create model
    json_file = open(file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    weight_name= file.split(".")
    logger.debug("Weight path : {}.h5".format(weight_name[0]))
    #load weights into new model
    loaded_model.load_weights("{}.h5".format(weight_name[0]))
    models.append(loaded_model)
	# For windows use '\\' for mac and linux use '/'
    model_names.append(weight_name[0].split("\\")[-1])

# path to training dataset
train_labels = os.listdir(train_data_dir)
# Load feature directory
feature_dir=config["feature_dir"]

logger.debug("Check weather Feature Directory exists")

if os.path.exists(feature_dir):
	path = pathlib.Path(feature_dir)
	logger.debug("Feature Directory exists, delete it and create new one")
	path.rmdir()
    os.mkdir(feature_dir)
else:
	logger.debug("Feature Directory doesn't exists, creating new one")
    os.mkdir(feature_dir)

# Load classifier directory
clf_dir=config["classifier_dir"]
logger.debug("Check weather classifier Directory exists")
if not os.path.exists(clf_dir):
	logger.debug("Feature classifier doesn't exists, creating new one")
    os.mkdir(clf_dir)


image_size=(img_width, img_height)
# loop over all the labels in the folder
logger.debug("loop over all the labels in the folder")
with open("{}/label.pkl".format(feature_dir), "wb") as label_file:
    for model,name in zip(models,model_names):
        features = []
        labels = []
        # h5f = h5py.File('features/{}.h5'.format(name), 'w')
        with h5py.File('{}/{}.hdf5'.format(feature_dir,name),'w') as f:
            # loop over all the labels in the folder
            for label in train_labels:
                cur_path = train_data_dir + "/" + label
                logger.debug(cur_path)
                for image_path in glob.glob(cur_path + "/*.jpg"):
                    img = image.load_img(image_path, target_size=image_size)
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)
                    feature = model.predict(x)
                    flat = feature.flatten()
                    features.append(flat)
                    labels.append(label)
            f.create_dataset('train', data=features)
    cPickle.dump(labels, label_file)

with open("{}/test_label.pkl".format(feature_dir), "wb") as label_file:
    for model,name in zip(models,model_names):
        # h5f = h5py.File('features/{}.h5'.format(name), 'w')
        with h5py.File('{}/{}.h5'.format(feature_dir,name),'w') as f:
            test_features = []
            test_labels = []
            # loop over all the labels in the folder
            for label in train_labels:
                cur_path = validation_data_dir + "/" + label
                for image_path in glob.glob(cur_path + "/*.jpg"):
                    img = image.load_img(image_path, target_size=image_size)
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)
                    feature = model.predict(x)
                    flat = feature.flatten()
                    test_features.append(flat)
                    test_labels.append(label)
            f.create_dataset('test', data=test_features)

    cPickle.dump(test_labels, label_file)
#h5f.close()

logger.debug("Load Label files")
with open('{}/test_label.pkl'.format(feature_dir), 'rb') as f:
    test_labels = cPickle.load(f)

with open('{}/label.pkl'.format(feature_dir), 'rb') as f:
    labels = cPickle.load(f)

# encode the labels using LabelEncoder
logger.debug("Encode the labels using LabelEncoder")
targetNames = np.unique(labels)
le = LabelEncoder()
le_labels = le.fit_transform(labels)
test_labels=le.transform(test_labels)

with open("{}/labelEncoder.pkl".format(clf_dir), "wb") as f:
	cPickle.dump(le, f)

logger.debug("Load saved features maps")
feat_pattern = os.path.join(feature_dir,'*.hdf5')
file_list = glob.glob(feat_pattern)
feat=[]
for file in file_list:
    print("Json file path :",file)
    # load json and create model
    with  h5py.File(file, 'r') as f:
        feat.append(f['train'][:])
i=0
for feature in feat:
    if i==0:
        features=np.asarray(feature)
    else:
        features = np.concatenate( [features,np.asarray(feature)], axis=1 )
    logger.debug("Feat: ",np.asarray(feature).shape)
    logger.debug("Features: ",features.shape)
    i=i+1

test_feat_pattern = os.path.join(feature_dir,'*.h5')
file_list = glob.glob(test_feat_pattern)
test_feat=[]
for file in file_list:
    print("Json file path :",file)
    # load json and create model
    with  h5py.File(file, 'r') as f:
        test_feat.append(f['test'][:])

i=0
for feature in test_feat:
    if i==0:
        test_features=np.asarray(feature)
    else:
        test_features = np.concatenate( [test_features,np.asarray(feature)], axis=1 )
    logger.debug("Feat: ",np.asarray(feature).shape)
    logger.debug("Features: ",test_features.shape)
    i=i+1

seed=config["seed"]

# use logistic regression as the model
logger.debug("[INFO] creating Logistic Regression...")
clf = LogisticRegression(random_state=seed)
clf.fit(features, le_labels)
# evaluate the model of test data
preds = clf.predict(test_features)
with open("{}/logistic.pkl".format(clf_dir), "wb") as f:
	cPickle.dump(clf, f)
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
logger.debug("[INFO] creating XGBoost...")
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, le_labels)
preds = gbm.predict(test_X)
with open("{}/xgb.pkl".format(clf_dir), "wb") as f:
	cPickle.dump(gbm, f)

logger.debug(classification_report(le.inverse_transform(preds),le.inverse_transform(test_labels)))
logger.debug("Accuracy :",accuracy_score(le.inverse_transform(preds),le.inverse_transform(test_labels)))


if config["pca"]:
	# Apply PCA

	from sklearn.decomposition import PCA

	X=np.concatenate( [features,test_features], axis=0 )
	pca = PCA(n_components=config["n_components"])
	pca.fit(X)
	with open("{}/pca.pkl".format(clf_dir), "wb") as f:
		cPickle.dump(pca, f)
	train_x = pca.transform(features)
	test_x = pca.transform(test_features)

	# Apply Logistic Regression with PCA
	logger.debug("[INFO] creating Logistic Regression with PCA...")
	clf = LogisticRegression(random_state=seed)
	clf.fit(train_x, le_labels)

	# evaluate the model of test data
	preds = clf.predict(test_x)
	with open("{}/logistic.pkl".format(clf_dir), "wb") as f:
		cPickle.dump(clf, f)
	logger.debug(classification_report(le.inverse_transform(preds),le.inverse_transform(test_labels)))
	logger.debug("Accuracy :",accuracy_score(le.inverse_transform(preds),le.inverse_transform(test_labels)))

	# Apply Xgboost with PCA
	# Prepare the inputs for the model
	train_X = np.asmatrix(train_x)
	test_X = np.asmatrix(test_x)

	# You can experiment with many other options here, using the same .fit() and .predict()
	# methods; see http://scikit-learn.org
	# This example uses the current build of XGBoost, from https://github.com/dmlc/xgboost
	logger.debug("[INFO] creating XGBoost with PCA...")
	gbm = xgb.XGBClassifier(max_depth=config["xgb_max_dept"], n_estimators=config["xgb_n_estimator"], \
					learning_rate=config["xgb_learning_rate"]).fit(train_X, le_labels)
	preds = gbm.predict(test_X)
	with open("{}/xgb.pkl".format(clf_dir), "wb") as f:
		cPickle.dump(gbm, f)
	logger.debug(classification_report(le.inverse_transform(preds),le.inverse_transform(test_labels)))
	logger.debug("Accuracy :",accuracy_score(le.inverse_transform(preds),le.inverse_transform(test_labels)))
