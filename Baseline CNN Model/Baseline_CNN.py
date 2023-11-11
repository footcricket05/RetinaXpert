#!/usr/bin/env python
# coding: utf-8

# # Basic CNN Model using Keras for Eye disease Prediction
# 
# ### Author - Shaurya Singh Srinet and Charvi Jain

# ### Importing necessary libraries

# In[1]:


import numpy as np # linear algebra
import pandas as pd 
import matplotlib.pyplot as plt # visualization
import seaborn as sns # visualization
import cv2
import tensorflow as tf 
from tensorflow import keras
from pathlib import Path
import PIL
import os


# ### Reading Input from local Machine

# In[4]:


glaucoma = Path(r"C:\Users\Charvi Jain\Downloads\dataset\glaucoma")
cataract = Path(r"C:\Users\Charvi Jain\Downloads\dataset\cataract")
normal = Path(r"C:\Users\Charvi Jain\Downloads\dataset\normal")
diabetic_retinopathy = Path(r"C:\Users\Charvi Jain\Downloads\dataset\diabetic_retinopathy")


# ### Creating a dataframe with the file path and the labels
# 

# In[5]:


disease_type = [glaucoma, cataract,normal,diabetic_retinopathy]
df = pd.DataFrame()
from tqdm import tqdm
for types in disease_type:
    for imagepath in tqdm(list(types.iterdir()), desc= str(types)):
        df = pd.concat([df, pd.DataFrame({'image': [str(imagepath)],'disease_type': [disease_type.index(types)]})], ignore_index=True)


# In[6]:


df


# In[7]:


df.disease_type.value_counts()


# ### Defining function to plot sample images
# 

# In[8]:


def plot_image(n, num_samples=3):
    disease_labels = ['glaucoma', 'cataract', 'normal', 'diabetic_retinopathy']
    images = df[df['disease_type'] == n].sample(num_samples)['image']

    plt.figure(figsize=(12, 12))

    for i, path in enumerate(images, 1):
        img = (plt.imread(path) - plt.imread(path).min()) / plt.imread(path).max()
        plt.subplot(3, 3, i)
        plt.imshow(img)
        plt.axis('off')
        plt.title(disease_labels[n])

    plt.show()


# In[9]:


plot_image(1)


# In[10]:


plot_image(0)


# In[11]:


plot_image(2)


# In[12]:


plot_image(3)


# ### Mapping the labels to the type of eye disease
# 

# In[13]:


df['disease_type'] = df['disease_type'].map({0:'glaucoma',1:'cataract',2:'normal',3:'diabetic_retinopathy'})


# ### Checking the label count to verify if it has been mapped

# In[14]:


df.disease_type.value_counts()


# ### Randomising the dataset

# In[15]:


df1 = df.sample(frac=1).reset_index(drop=True)


# ### Creating a countplot of the disease type
# 

# In[16]:


sns.countplot(x = 'disease_type', data = df1)
plt.xlabel("Disease type")
plt.show()


# ### Importing tensorflow libraries

# In[17]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix


# ### Augumentating the images

# In[18]:


datagen = ImageDataGenerator(preprocessing_function=preprocess_input,validation_split=0.2)


# ### Creating the train data
# 

# In[19]:


train_data = datagen.flow_from_dataframe(dataframe=df1,
                                          x_col ='image',
                                          y_col = 'disease_type',
                                          target_size=(224,224),
                                          class_mode = 'categorical',
                                          batch_size = 32,
                                          shuffle = True,
                                          subset = 'training')


# ### Creating the validation data
# 

# In[20]:


valid_data = datagen.flow_from_dataframe(dataframe=df1,
                                          x_col ='image',
                                          y_col = 'disease_type',
                                          target_size=(224,224),
                                          class_mode = 'categorical',
                                          batch_size = 32,
                                          shuffle = False,
                                          subset = 'validation')


# In[21]:


labels=[key for key in train_data.class_indices]
num_classes = len(disease_type)

model = keras.Sequential([ 
    layers.Rescaling(1./255, input_shape=(224,224, 3)), 
    layers.Conv2D(16, 3, padding='same', activation='relu'), 
    layers.MaxPooling2D(), 
    layers.Conv2D(32, 3, padding='same', activation='relu'), 
    layers.MaxPooling2D(), 
    layers.Conv2D(64, 3, padding='same', activation='relu'), 
    layers.MaxPooling2D(), 
    layers.Flatten(), 
    layers.Dense(128, activation='relu'), 
    layers.Dense(num_classes,activation='softmax') 
]) 


# In[22]:


model.compile(optimizer='adam', 
              loss=tf.keras.losses.categorical_crossentropy, 
              metrics=['accuracy']) 
model.summary()


# ### Fitting the model
# 

# In[23]:


his = model.fit( 
  train_data,
    validation_data=valid_data, 
  epochs=15 
)


# ### Creating a plot of accuracy and val_acuracy for each epoch
# 

# In[24]:


plt.plot(his.history['accuracy'])
plt.plot(his.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Val'])
plt.show()


# ### Creating a plot of loss and validation loss for each epoch

# In[25]:


plt.plot(his.history['loss'])
plt.plot(his.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Val'])
plt.show()


# ### Evaluating the model

# In[26]:


y_test = valid_data.classes
y_pred = model.predict(valid_data)
y_pred = np.argmax(y_pred,axis=1)


# ### Generating classification report

# In[29]:


print(classification_report(y_test,y_pred,target_names = labels))


# ### Saving the model

# In[33]:


model.save("cnnkeras.h5")

