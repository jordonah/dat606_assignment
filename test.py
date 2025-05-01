import os 
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout 
from tensorflow.keras.callbacks import ModelCheckpoint 
# Set random seed for reproducibility 
seed_value = 42 
tf.random.set_seed(seed_value) 
# Define dataset path 
# Image parameters 
img_width, img_height = 224, 224 
batch_size = 32 
# Data preprocessing 
datagen = ImageDataGenerator(rescale=1./255, 
validation_split=0.2) 
# train_generator = datagen.flow_from_directory( 
# dataset_path, 
# target_size=(img_width, img_height), 
# batch_size=batch_size, 
# class_mode='binary', 
# subset='training', 
# seed=seed_value 
# ) 
# val_generator = datagen.flow_from_directory( 
# dataset_path, 
# target_size=(img_width, img_height), 
# batch_size=batch_size, 
# class_mode='binary', 
# subset='validation', 
# seed=seed_value 
# ) 
# print("Class indices:", 
# train_generator.class_indices) 
# Define CNN model 
model = Sequential([ 
Conv2D(32, (3,3), activation='relu', 
input_shape=(img_width, img_height, 3)), 
MaxPooling2D(2,2), 
Conv2D(64, (3,3), activation='relu'), 
MaxPooling2D(2,2), 
Conv2D(128, (3,3), activation='relu'), 
MaxPooling2D(2,2), 
Flatten(), 
Dense(128, activation='relu'), 
Dropout(0.5), 
Dense(1, activation='sigmoid') 
]) 
# Print model summary 
model.summary() 
# Compile model 
model.compile(optimizer='adam', 
loss='binary_crossentropy', metrics=['accuracy']) 
# Define checkpoint callback to save best model based on validation loss 
checkpoint = ModelCheckpoint("best_fire_detection_model.keras", monitor='val_loss', save_best_only=True, mode='min', verbose=1) 
# Train model 
epochs = 10 
model.fit(train_generator, 
validation_data=val_generator, epochs=epochs, 
callbacks=[checkpoint]) 
# Evaluate model on validation set 
val_loss, val_accuracy = model.evaluate(val_generator) 
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%") 
print("Model training complete. Best model saved as 'best_fire_detection_model.keras'.") 