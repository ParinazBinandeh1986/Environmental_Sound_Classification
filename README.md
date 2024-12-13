# Environmental_Sound_Classification 
!pip install numpy tqdm librosa pandas resampy 

import numpy as np
from tqdm import tqdm
import os
import librosa
import pandas as pd
import resampy 


metadata_path = '/content/drive/MyDrive/Colab Notebooks/Dataset/UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv'
audio_dataset_path = '/content/drive/MyDrive/Colab Notebooks/Dataset/UrbanSound8K/UrbanSound8K/audio'


# Define the features extractor function
def features_extractor(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfcc_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc_scaled_features = np.mean(mfcc_features.T, axis=0)
    return mfcc_scaled_features

# Test the feature extractor with a sample file
sample_file = os.path.join(audio_dataset_path, 'fold1', '7061-6-0-0.wav')
print("Testing feature extraction on:", sample_file)
features = features_extractor(sample_file)
print("Extracted features:", features) 



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.keras.optimizers import Adam
from sklearn import metrics 

# Load the metadata
metadata_path = '/content/drive/MyDrive/Colab Notebooks/Dataset/UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv'
metadata = pd.read_csv(metadata_path)

# Extract labels from metadata
labels = metadata['classID'].values  # 'classID' contains the labels

# Define the number of unique classes
num_labels = len(np.unique(labels))

# Build the model
model = Sequential()

# First layer (with Input layer explicitly defined)
model.add(Input(shape=(40,)))  # Replace 40 with the feature dimension
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Second layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Third layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Final layer (output layer)
model.add(Dense(num_labels))
model.add(Activation('softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()


model.compile (loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer = 'adam') 

# Feature extraction function
def features_extractor(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfcc_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfcc_scaled_features = np.mean(mfcc_features.T, axis=0)
        return mfcc_scaled_features
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
        return None

# Extract features and labels
features = []
labels = []

for index, row in tqdm(metadata.iterrows(), total=metadata.shape[0], desc="Processing audio files"):
    fold = f"fold{row['fold']}"
    file_name = row['slice_file_name']
    class_label = row['classID']
    file_path = os.path.join(audio_dataset_path, fold, file_name)

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    feature = features_extractor(file_path)
    if feature is not None:
        features.append(feature)
        labels.append(class_label)

# Convert to NumPy arrays
features = np.array(features)
labels = np.array(labels)

print(f"Features shape: {features.shape}")
print(f"Labels shape: {labels.shape}")

# Save for reuse
np.save("features.npy", features)
np.save("labels.npy", labels)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

# Reload the saved features and labels
features = np.load("features.npy")
labels = np.load("labels.npy")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Encode labels
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# One-hot encoding
y_train = to_categorical(y_train, num_classes=len(np.unique(labels)))
y_test = to_categorical(y_test, num_classes=len(np.unique(labels)))

# Now train your model
model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))



from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('/content/drive/MyDrive/Colab Notebooks/Models/UrbanSound8k_audio_classification.keras')

# Test the model (using X_test and y_test)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}") 


test_accuracy=model.evaluate(X_test, y_test,verbose=0)
print (test_accuracy[1]) 


# Save the entire model
model_path = '/content/drive/MyDrive/Colab Notebooks/Models/UrbanSound8k_audio_classification.keras'  # Specify the path and file name
model.save(model_path)

print(f"Model saved to {model_path}") 


from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('/content/drive/MyDrive/Colab Notebooks/Models/UrbanSound8k_audio_classification.keras')

# Test the model (using X_test and y_test)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}") 


import numpy as np
import librosa
from tensorflow.keras.models import load_model

# Load the saved model
model_path = '/content/drive/MyDrive/Colab Notebooks/Models/UrbanSound8k_audio_classification.keras'
model = load_model(model_path)
print("Model loaded successfully.")

# Class mapping (update this if your dataset has different classes)
class_mapping = {
    0: "Air Conditioner",
    1: "Car Horn",
    2: "Children Playing",
    3: "Dog Bark",
    4: "Drilling",
    5: "Engine Idling",
    6: "Gunshot",
    7: "Jackhammer",
    8: "Siren",
    9: "Street Music"
}
# Function to extract features from a sound file
def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfcc_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfcc_scaled_features = np.mean(mfcc_features.T, axis=0)
        return mfcc_scaled_features
    except Exception as e:
        print(f"Error extracting features from {file_name}: {e}")
        return None

# Path to the audio file
audio_file_path = '/content/drive/MyDrive/Colab Notebooks/Dataset/UrbanSound8K/UrbanSound8K/audio/fold8/103076-3-1-0.wav'  # Replace with the actual file path

# Extract features from the audio file
features = extract_features(audio_file_path)

if features is not None:
    # Reshape features for prediction
    features = np.array(features).reshape(1, -1)  # Reshape to (1, 40)

    # Predict the class
    predictions = model.predict(features)
    predicted_class_id = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_mapping.get(predicted_class_id, "Unknown Class")

    print(f"Predicted Class ID: {predicted_class_id}")
    print(f"Predicted Class Name: {predicted_class_name}")
else:
    print("Failed to extract features from the audio file.")



import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the saved model
from tensorflow.keras.models import load_model
model_path = '/content/drive/MyDrive/Colab Notebooks/Models/UrbanSound8k_audio_classification.keras'  # Replace with actual model path
model = load_model(model_path)

# Reload test data
X_test = np.load("features.npy")  # Replace with actual test features file
y_test = np.load("labels.npy")    # Replace with actual test labels file

# Check if y_test is one-hot encoded
if len(y_test.shape) == 1:  # Labels are integers
    y_true = y_test
else:  # One-hot encoded
    y_true = np.argmax(y_test, axis=1)

# Predict on the test set
y_pred_probs = model.predict(X_test)  # Predicted probabilities
y_pred = np.argmax(y_pred_probs, axis=1)  # Convert probabilities to class labels

# Evaluate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()



#############   CNN_MelSpectrogram ############# 



# Load the metadata
import pandas as pd
metadata = pd.read_csv(metadata_path)

# Define feature extraction function for Mel Spectrogram
def mel_spectrogram_extractor(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128, fmax=8000)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)  # Convert to dB scale
        mel_scaled_features = np.mean(mel_spectrogram_db.T, axis=0)  # Take the mean across time
        return mel_scaled_features
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
        return None

# Extract features and labels
features = []
labels = []

for index, row in tqdm(metadata.iterrows(), total=metadata.shape[0], desc="Processing audio files"):
    fold = f"fold{row['fold']}"
    file_name = row['slice_file_name']
    class_label = row['classID']
    file_path = os.path.join(audio_dataset_path, fold, file_name)

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    feature = mel_spectrogram_extractor(file_path)
    if feature is not None:
        features.append(feature)
        labels.append(class_label)

# Convert to NumPy arrays
features = np.array(features)
labels = np.array(labels)

print(f"Features shape: {features.shape}")
print(f"Labels shape: {labels.shape}")

# Save for reuse
np.save("mel_features.npy", features)
np.save("labels.npy", labels)



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
import numpy as np

# Load preprocessed data
features = np.load("mel_features.npy")  # Mel spectrogram features
labels = np.load("labels.npy")         # Corresponding labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Encode labels
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=len(np.unique(labels)))
y_test = to_categorical(y_test, num_classes=len(np.unique(labels)))

# Reshape data for CNN (Conv1D input shape)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  # Shape: (samples, features, channels)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the CNN model
model = Sequential()

# First Conv Layer
model.add(Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

# Second Conv Layer
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

# Flatten and Fully Connected Layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(np.unique(labels)), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy}")

# Save the model
model.save("UrbanSound8k_MelSpectrogram_CNN.h5")



import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Save the model
model.save("UrbanSound8k_MelSpectrogram_CNN.h5")
print("Model saved successfully!")

# Load the saved model
model = load_model("UrbanSound8k_MelSpectrogram_CNN.h5")
print("Model loaded successfully!")

# Predict on the test set
y_pred_probs = model.predict(X_test)  # Predicted probabilities
y_pred = np.argmax(y_pred_probs, axis=1)  # Convert probabilities to class labels
y_true = np.argmax(y_test, axis=1)  # Convert one-hot encoded true labels to class indices

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Generate classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()



