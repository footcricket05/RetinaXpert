#!/usr/bin/env python
# coding: utf-8

# # ResNet50 Model for EDP 
# 
# ### Authors - Charvi Jain and Shaurya Singh Srinet

# ### Importing required libraries

# In[1]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import os, gdown, cv2, shutil, math
from glob import glob
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
from keras.layers import *
from keras.models import Model, load_model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow_addons.metrics import F1Score
from sklearn.metrics import confusion_matrix


# ### Loading the dataset

# In[24]:


input_directory = r"C:\Users\Charvi Jain\Downloads\dataset"

for dirname, _, filenames in os.walk(input_directory):
        print("Dataset Accessed from Local Machine")


# ### Data Distribution and visualization

# In[3]:


import matplotlib.pyplot as plt
class_names = ["glaucoma", "cataract", "normal", "diabetic_retinopathy"]
# Count for each class
count_for_class_0 = 1007
count_for_class_1 = 1038
count_for_class_2 = 1074
count_for_class_3 = 1098

# Creating a list with the counts
class_counts = [count_for_class_0, count_for_class_1, count_for_class_2, count_for_class_3]

# Defining colors for the bar graph
colors = ['lightcoral', 'lightgrey', 'plum', 'lightsteelblue']

# Creating a bar graph
plt.figure(figsize=(8, 6))
plt.bar(class_names, class_counts, color=colors)

# Adding labels and a title
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Data Distribution")
plt.show()


# ### Making Dataframes

# In[4]:


sdir = r"C:\Users\Charvi Jain\Downloads\dataset"


def make_dataframes(sdir):
    bad_images = []
    good_ext = ['jpg', 'jpeg', 'png', 'tiff']
    filepaths = []
    labels = []
    classes = sorted(os.listdir(sdir))
    for klass in classes:
        classpath = os.path.join(sdir, klass)
        flist = sorted(os.listdir(classpath))
        desc = f'{klass:23s}'
        for f in tqdm(flist, ncols=110, desc=desc, unit='file', colour='blue'):
            fpath = os.path.join(classpath, f)
            fl = f.lower()
            index = fl.rfind('.')
            ext = fl[index + 1:]
            if ext in good_ext:
                try:
                    img = cv2.imread(fpath)
                    shape = img.shape
                    filepaths.append(fpath)
                    labels.append(klass)
                except:
                    bad_images.append(fpath)
                    print('defective image file: ', fpath)
            else:
                bad_images.append(fpath)
    Fseries = pd.Series(filepaths, name='filepaths')
    Lseries = pd.Series(labels, name='labels')
    df = pd.concat([Fseries, Lseries], axis=1)

    train_df, dummy_df = train_test_split(df, train_size=.8, shuffle=True, random_state=123, stratify=df['labels'])
    valid_df, test_df = train_test_split(dummy_df, train_size=.5, shuffle=True, random_state=123,
                                         stratify=dummy_df['labels'])
    classes = sorted(train_df['labels'].unique())
    class_count = len(classes)
    sample_df = train_df.sample(n=50, replace=False)

    ht = 0
    wt = 0
    count = 0
    for i in range(len(sample_df)):
        fpath = sample_df['filepaths'].iloc[i]
        try:
            img = cv2.imread(fpath)
            h = img.shape[0]
            w = img.shape[1]
            wt += w
            ht += h
            count += 1
        except:
            pass
    have = int(ht / count)
    wave = int(wt / count)
    aspect_ratio = have / wave
    print('number of classes in processed dataset= ', class_count)
    counts = list(train_df['labels'].value_counts())
    print('the maximum files in any class in train_df is ', max(counts),
          ' the minimum files in any class in train_df is ', min(counts))
    print('train_df length: ', len(train_df), '  test_df length: ', len(test_df), '  valid_df length: ', len(valid_df))
    print('average image height= ', have, '  average image width= ', wave, ' aspect ratio h/w= ', aspect_ratio)
    return train_df, test_df, valid_df


train_df, test_df, valid_df = make_dataframes(sdir=sdir)


# ### Defining ImageDataGenerator

# In[5]:


batch_size = 64
img_height, img_width = 300, 300
img_size = (img_height, img_width)
input_shape = (img_height, img_width, 3)

def create_data(batch_size, train_df, test_df, valid_df, img_size):
    train_gen = ImageDataGenerator(rescale=1/255.0, rotation_range=20, vertical_flip=True, shear_range=0.2, dtype=float)
    test_gen = ImageDataGenerator(rescale=1/255.0, dtype=float)
    
    train_set = train_gen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical', batch_size=batch_size)
    test_set = test_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical', batch_size=batch_size)
    val_set = test_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical', batch_size=batch_size)
    
    return train_set, test_set, val_set


train_set, test_set, val_set = create_data(batch_size, train_df, test_df, valid_df, img_size)


# ### Label Mapping

# In[6]:


label_dict = {v:k for k,v in train_set.class_indices.items()}
label_dict


# ### Generating Model

# In[7]:


def generate_model(pretrained_base,name=None):

    pretrained_base.trainable=False
    model = Sequential([
        pretrained_base,
        Flatten(),
        Dense(units=512, activation='relu'),
        Dense(units=256, activation='relu'),
        Dense(units=128, activation='relu'),
        Dense(units=64, activation='relu'),
        Dense(units=4, activation='softmax')
    ], name=name)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy', F1Score(num_classes=4)])
    
    return model


# ### Generating Callbacks

# In[8]:


def generate_callbacks(ResNet50):
    checkpoint = ModelCheckpoint(filepath=f'ResNet50.h5', save_best_only=True, save_weights_only=True)
    return [checkpoint]


# ### Loading ResNet50 model

# In[9]:


import tensorflow as tf
pretrained_params = {
    'weights':'imagenet',
    'include_top':False,
    'input_shape':input_shape
}


pretrained_models = {
    'ResNet50':tf.keras.applications.ResNet50(**pretrained_params)
}


fit_args = {
    'epochs':10,
    'steps_per_epoch':train_set.n//train_set.batch_size,
    'validation_data':val_set,
    'validation_steps':val_set.n//val_set.batch_size
}


# In[10]:


ResNet50_model = generate_model(pretrained_models['ResNet50'], name='ResNet50')
callbacks = generate_callbacks('ResNet50')
history = ResNet50_model.fit(train_set, callbacks=callbacks, **fit_args)
ResNet50_model.save('ResNet50.h5')


# In[13]:


resnet = load_model('ResNet50.h5')


# In[14]:


inputs = Input(shape=input_shape)
resnet_preds = resnet(inputs)


# In[15]:


# Averaging for ResNet predictions
outputs = resnet_preds


# ### Model Evaluation 

# In[19]:


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')


plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.show()


# ### Generating Classificarion report and confusion report

# In[20]:


from sklearn.metrics import confusion_matrix, classification_report

test_predictions = resnet.predict(test_set)


# Convert predictions to class labels
test_pred_labels = np.argmax(test_predictions, axis=1)

# Use the original labels from the generator
test_true_labels = test_set.labels

# If the true labels are not one-hot encoded, convert them
if isinstance(test_true_labels[0], list):
    test_true_labels = np.argmax(test_true_labels, axis=1)

# Confusion Matrix
conf_matrix = confusion_matrix(test_true_labels, test_pred_labels)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
class_report = classification_report(test_true_labels, test_pred_labels, target_names=class_names)
print("\nClassification Report:")
print(class_report)


# In[ ]:




