from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.uic import loadUi
import sys
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.image import resize
import tensorflow as tf
from tensorflow.keras.models import load_model
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QPushButton, QFileDialog, QWidget
from Explainable_Audio_Classifier.nearmiss import NearMissImageFinder, calc_cosine_similarities


class NearMissUI(QtWidgets.QMainWindow):
    def __init__(self,limitBoxValue,audioFile,parent=None):
        super(NearMissUI,self).__init__(parent=parent)
        loadUi("KMU-GUI-NearMiss.ui", self)
        self.audioFile = audioFile
        self.top_images = None        
        self.limitBoxValue = limitBoxValue
        self.currentNearMiss = 0
        # setting special Audio Classes for playing the audio
        self.audioOutput = QAudioOutput()
        self.mediaPlayer = QMediaPlayer()
        self.mediaPlayer.setAudioOutput(self.audioOutput)
        # calculate NearMisses
        self.calculate_nearMisses()
        # calling event handlers methods
        self.setup_event_handlers()
        

    # handling the events
    def setup_event_handlers(self):
        self.show_nearMiss(self.top_images[self.currentNearMiss][0]) # By default first show the first nearMiss
        self.set_audio_source(self.top_images[self.currentNearMiss][0])
        self.nextBtn.clicked.connect(self.show_next)
        self.previousBtn.clicked.connect(self.show_previous)
        self.playAudioBtn.clicked.connect(self.play_audio)
        self.pauseBtn.clicked.connect(self.pause_audio)

    def show_next(self):
        # first remove the content of QVBoxLayout:
        self.spectVerticalLayout.takeAt(0)
        # then show next:
        self.currentNearMiss += 1
        if(self.currentNearMiss >= len(self.top_images)):
            self.currentNearMiss = len(self.top_images) - 1
        self.show_nearMiss(self.top_images[self.currentNearMiss][0])
        self.set_audio_source(self.top_images[self.currentNearMiss][0])
        
    def show_previous(self):
        # first remove the content of QVBoxLayout:
        self.spectVerticalLayout.takeAt(0)
        # then show previous:
        self.currentNearMiss -= 1
        if(self.currentNearMiss < 0):
            self.currentNearMiss = 0
        self.show_nearMiss(self.top_images[self.currentNearMiss][0])
        self.set_audio_source(self.top_images[self.currentNearMiss][0])
        
    def set_audio_source(self,audioPath):
        self.audioFileList.clear()
        self.audioFileList.addItem(os.path.basename(audioPath))
        audioFileUrl = QtCore.QUrl.fromLocalFile(os.path.abspath(audioPath))
        self.mediaPlayer.setSource(audioFileUrl)

    def play_audio(self):
        self.mediaPlayer.play()
        self.pauseBtn.setDisabled(False)
        self.playAudioBtn.setDisabled(True)

    def pause_audio(self):
        self.mediaPlayer.pause()
        self.pauseBtn.setDisabled(True)
        self.playAudioBtn.setDisabled(False)
    # calculate cosine similarity and find nearmisses:
    def calculate_nearMisses(self):
        ns_Img_finder = NearMissImageFinder(r"C:\MyUserContents\Uni Bamberg Dars\KMU\kmu-ki-ez\Explainable_Audio_Classifier\audio_model_2.h5")
        predicted_test_class, test_feature_vector, preprocessed_image, class_probabilities = ns_Img_finder.preprocess_image(os.path.abspath(self.audioFile))
        # Determine the opposite class based on the predicted test class
        opposite_class = "abnormal" if predicted_test_class == "normal" else "normal"
        train_data = ns_Img_finder.load_and_preprocess_training_data(r'C:\Users\karam\Valve_audio_data',opposite_class)
        # calculate cosine similarities
        cosine_similarities = calc_cosine_similarities(test_feature_vector, train_data)

        # print('cosine_similarities: ')
        # print(cosine_similarities)
        # Sort the training images by cosine similarity in descending order
        sorted_images = sorted(cosine_similarities.items(), key=lambda x: x[1],reverse=True)
        # Display the top 5 "near miss" images and their similarity scores
        self.top_images = sorted_images[:self.limitBoxValue] # top_images[0...n][0]
        # print(f'{self.limitBoxValue} top near misses: ')
        print(self.top_images) 
        
    def show_nearMiss(self,currentNearMissFilePath):
        # load audio file
        audio_data, sample_rate = librosa.load(os.path.abspath(currentNearMissFilePath), sr=None)
         # Matplotlib Canvas for Spectrogram
        self.figure, self.ax = plt.subplots(figsize=(6, 2))
        self.canvas = FigureCanvas(self.figure)
        # setting the layout
        self.spectVerticalLayout.addWidget(self.canvas)
        self.mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        # Clear the previous plot and draw the new spectrogram
        self.ax.clear()
        self.ax.set_axis_off()
        librosa.display.specshow(librosa.power_to_db(self.mel_spectrogram), sr=sample_rate, x_axis='time', y_axis='mel', ax=self.ax)
        self.ax.set(title="Spektrogramm")
        self.figure.tight_layout()
        self.canvas.draw()
        


class SpectrogrammUI(QtWidgets.QMainWindow):
    def __init__(self,file_path,parent=None):
        super(SpectrogrammUI,self).__init__(parent=parent)
        self.setWindowTitle('Spektrogramm')
        self.setGeometry(500,150,600,500)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layout
        self.layout = QVBoxLayout(central_widget)

        # load audio file
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        # calculate class probabilities and accuracy
        class_probabilities, predicted_class_index = self.calculate_class_probibilities(audio_data, sample_rate)
        classes = ['normal','abnormal']
        predicted_class = classes[predicted_class_index]
        accuracy = class_probabilities[predicted_class_index]
        predicted_class_text = f'Das Audio wird klassifiziert als: {predicted_class}'
        accuracy_text = f'Genauigkeit: {accuracy:.4f}'
        # Matplotlib Canvas for Spectrogram
        self.figure, self.ax = plt.subplots(figsize=(6, 2))
        self.canvas = FigureCanvas(self.figure)
        # setting the layout
        self.layout.addWidget(self.canvas)
        predicted_class_lbl = QtWidgets.QLabel(self)
        predicted_class_lbl.setText(predicted_class_text)
        accuracy_lbl = QtWidgets.QLabel(self)
        accuracy_lbl.setText(accuracy_text)
        self.layout.addWidget(predicted_class_lbl)
        self.layout.addWidget(accuracy_lbl)
        self.setLayout(self.layout)
        # showing the spectrogramm
        
        self.mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        # Clear the previous plot and draw the new spectrogram
        self.ax.clear()
        self.ax.set_axis_off()
        librosa.display.specshow(librosa.power_to_db(self.mel_spectrogram), sr=sample_rate, x_axis='time', y_axis='mel', ax=self.ax)
        self.ax.set(title="Spektrogramm")
        self.figure.tight_layout()
        self.canvas.draw()
        # sc = SpecClassify(file_path)
        

    def calculate_class_probibilities(self,audio_data, sample_rate):
             # Load the saved model
            target_shape = (128, 128)
            model = load_model(r"C:\MyUserContents\Uni Bamberg Dars\KMU\kmu-ki-ez\Explainable_Audio_Classifier\audio_model_2.h5")
            # Load and preprocess the audio file
            mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate) 
            mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
            mel_spectrogram = tf.reshape(mel_spectrogram, (1,) + target_shape + (1,)) 
            # Make predictions
            predictions = model.predict(mel_spectrogram)

            print(predictions)
            # Get the class probabilities
            class_probabilities = predictions[0]
            print(type(class_probabilities))
            # Get the predicted class index
            predicted_class_index = np.argmax(class_probabilities)
            print(predicted_class_index)

            return class_probabilities, predicted_class_index
             

class MainUI(QtWidgets.QMainWindow):
    # loading the UI Setup
    def __init__(self):
        super().__init__()
        loadUi("KMU-GUI.ui", self)
        # setting special Audio Classes for playing the audio
        self.audioOutput = QAudioOutput()
        self.mediaPlayer = QMediaPlayer()
        self.mediaPlayer.setAudioOutput(self.audioOutput)
        # calling event handlers methods
        self.WaitLblNearMiss.setVisible(False)
        self.setup_event_handlers()        

    # handling the events
    def setup_event_handlers(self):
        self.chooseFileBtn.clicked.connect(self.open_file)
        self.playAudioBtn.clicked.connect(self.play_audio)
        self.pauseBtn.clicked.connect(self.pause_audio)
        self.showAndClassifyBtn.clicked.connect(self.show_spectrogramm_and_classify)
        self.showNearMissBtn.clicked.connect(self.show_near_misses)

    # event handlers
    def open_file(self):
        audioFile, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Audiodatei auswählen", filter="Audio Files (*.mp3 *.wav *.m4a)")
        if audioFile:
            self.audioFile = audioFile
            self.audioFileList.clear()
            self.audioFileList.addItem(os.path.basename(self.audioFile))
            self.playAudioBtn.setDisabled(False)
            self.showAndClassifyBtn.setDisabled(False)
            self.showNearMissBtn.setDisabled(False)
            

    def play_audio(self):
        audioFileUrl = QtCore.QUrl.fromLocalFile(os.path.abspath(self.audioFile))
        self.mediaPlayer.setSource(audioFileUrl)
        self.mediaPlayer.play()
        self.pauseBtn.setDisabled(False)
        self.playAudioBtn.setDisabled(True)

    def pause_audio(self):
        self.mediaPlayer.pause()
        self.pauseBtn.setDisabled(True)
        self.playAudioBtn.setDisabled(False)

    def show_spectrogramm_and_classify(self):
        SpectrogrammUI(file_path=os.path.abspath(self.audioFile),parent=self).show()

    def show_near_misses(self):
        self.WaitLblNearMiss.setVisible(True)
        timer = QtCore.QTimer(self)
        timer.setSingleShot(True)  # Set to run only once
        timer.timeout.connect(self.run_show_near_misses)
        # Start the timer for 1.2 seconds this is for first updating the lable and then running the NearMissUI after 1.2 seconds
        timer.start(1200)
    def run_show_near_misses(self):
        NearMissUI(limitBoxValue=self.nearMissLimitBox.value(),audioFile=self.audioFile,parent=self).show()
        self.WaitLblNearMiss.setVisible(False)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = MainUI()
    ui.show()
    app.exec()