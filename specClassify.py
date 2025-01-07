from tensorflow.keras.models import load_model
import librosa
from tensorflow.image import resize
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import threading

class SpecClassify():
    def __init__(self,file_path):
        # Load the saved model
        self.model = load_model(r"C:\MyUserContents\Uni Bamberg Dars\KMU\kmu-ki-ez\Explainable_Audio_Classifier\audio_model_2.h5")

        # Define the target shape for input spectrograms
        self.target_shape = (128, 128)

        # Define your class labels
        self.classes = ['abnormal', 'normal']

        # Load and preprocess the audio file
        self.audio_data, self.sample_rate = librosa.load(file_path, sr=None)
        self.mel_spectrogram = librosa.feature.melspectrogram(y=self.audio_data, sr=self.sample_rate)
        self.mel_spectrogram = resize(np.expand_dims(self.mel_spectrogram, axis=-1), self.target_shape)
        self.mel_spectrogram = tf.reshape(self.mel_spectrogram, (1,) + self.target_shape + (1,))

    # Functions to preprocess and classify an audio file
    def show_spec(self):
        # plotting the spectogram    
        S = librosa.feature.melspectrogram(y=self.audio_data, sr=self.sample_rate)  
        plt.figure(figsize=(6, 2)) # change it for the size
        plt.axis('off') # comment out for turning the axis on
        librosa.display.specshow(librosa.power_to_db(S),sr=self.sample_rate, x_axis='time', y_axis='mel')
        # plt.colorbar(format='%+2.0f dB') # for showing the db colorbar
        plt.tight_layout()
        plt.show()
        
    def classify_spec(self):
        # Make predictions
        predictions = self.model.predict(self.mel_spectrogram)

        # Get the class probabilities
        class_probabilities = predictions[0]

        # Get the predicted class index
        predicted_class_index = np.argmax(class_probabilities)

        return class_probabilities, predicted_class_index



if __name__ == "__main__":
    test_audio_file = r'C:\Users\karam\Valve audio data\abnormal\\1_abnormal.wav'
    sc = SpecClassify(test_audio_file)

    # Test an audio file
    class_probabilities, predicted_class_index = sc.classify_spec()
    # Display results for all classes
    for i, class_label in enumerate(sc.classes):
        probability = class_probabilities[i]
        print(f'Class: {class_label}, Probability: {probability:.4f}')

    # Calculate and display the predicted class and accuracy
    predicted_class = sc.classes[predicted_class_index]
    accuracy = class_probabilities[predicted_class_index]
    print(f'The audio is classified as: {predicted_class}')
    print(f'Accuracy: {accuracy:.4f}')

    sc.show_spec()


# t1 = threading.Thread(target=sc.show_spec,args=([test_audio_file]))
# t2 = threading.Thread(target=sc.classify_spec,args=())
# t1.start()
# t2.start()
# t1.join()
# class_probabilities, predicted_class_index = t2.join()

