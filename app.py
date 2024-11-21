import streamlit as st
import librosa
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model

# Load your existing model
model = load_model("D:/Audio_classification/models/audio_classification_cnn.h5")

# Save in SavedModel format
model.save("D:/Audio_classification/models/audio_classification_cnn_fixed", save_format='tf')

# Load models
# Load the model using tf.saved_model.load (after converting to .pb format if necessary)
model_2d_cnn = tf.saved_model.load("D:/Audio_classification/models/audio_classification_cnn_fixed")  # Load 2D CNN model in .h5 format
model_1d_cnn = load_model("D:/Audio_classification/models/my_model.h5", compile=False)  # Load 1D CNN model in .h5 format
model_1d_cnn.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load the meta-model (pickle)
meta_model = pickle.load(open("D:/Audio_classification/models/meta_classifier.pkl", "rb"))

# Helper function: Extract Mel spectrogram
def extract_mel_spectrogram(audio, sr):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Resize mel spectrogram to match the training shape (400, 1000, 3)
    mel_resized = np.resize(mel_db, (400, 1000))
    
    # Create 3 channels (You can stack the same Mel spectrogram across 3 channels, or use other features)
    mel_3_channels = np.stack([mel_resized, mel_resized, mel_resized], axis=-1)  # Shape (400, 1000, 3)
    
    return mel_3_channels

# Helper function: Extract MFCCs
def extract_mfccs(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=18)
    return mfccs.T  # Return transposed to fit input shape

# Calculate Shift Mean and RMS Mean
def calculate_shift_rms(audio):
    # Calculate Shift Mean
    shift_mean = np.mean(np.diff(audio))  # Shift is the mean difference between consecutive samples
    # Calculate RMS Mean
    rms_mean = np.sqrt(np.mean(audio ** 2))  # Root mean square of the signal
    return shift_mean, rms_mean

# Streamlit interface
st.title("Cry Sound Classification")
st.write("Upload a .wav file to classify the cry reason into one of five classes.")

uploaded_file = st.file_uploader("Choose a .wav file", type="wav")

if uploaded_file is not None:
    # Load audio
    audio, sr = librosa.load(uploaded_file, sr=None)

    # Extract features
    mel = extract_mel_spectrogram(audio, sr)
    mfccs = extract_mfccs(audio, sr)
    shift_mean, rms_mean = calculate_shift_rms(audio)

    # Reshape mel for 2D CNN: Add batch size and channel dimensions
    mel_input = mel[np.newaxis, :, :, :]  # Shape (1, 400, 1000, 3)

    # Prepare input for 1D CNN
    mfccs_input = mfccs[np.newaxis, :, :]  # Add batch dimension
    additional_features = np.array([[shift_mean, rms_mean]])  # Combine shift and RMS into a single array
    features_1d = np.concatenate((mfccs_input.reshape(mfccs_input.shape[1], -1), additional_features), axis=1)
    features_1d = features_1d[np.newaxis, :, :]  # Add batch dimension

    # Predict with 2D CNN
    output_2d_cnn = model_2d_cnn.predict(mel_input)

    # Predict with 1D CNN
    output_1d_cnn = model_1d_cnn.predict(features_1d)

    # Concatenate outputs and predict with meta-model
    combined_features = np.concatenate((output_2d_cnn, output_1d_cnn), axis=1)
    prediction = meta_model.predict(combined_features)

    # Display result
    class_labels = ["Belly Pain", "Burping", "Discomfort", "Hungry", "Tired"]
    st.write(f"Predicted Class: {class_labels[int(prediction[0])]}")  # Assuming prediction returns an integer
