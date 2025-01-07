# ##%%
# import tensorflow as tf
# from sklearn.metrics.pairwise import cosine_similarity
# import matplotlib.pyplot as plt
# import os
# import numpy as np
# from PIL import Image
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.applications.resnet50 import preprocess_input
# from tensorflow.keras.preprocessing import image
# from keras.utils import load_img, img_to_array
# from keras.models import load_model
# # from tqdm import tqdm
# import pickle
# import pandas as pd


# class NearMissImageFinder:

#     def __init__(self, model_path, test_image):
#         """
#         Initialize the NearMissImageFinder class
#         Args:
#             model_path (str): Path to the pre-trained model.
#             test_image (str): Path to the test image for analysis.
#         """
#         self.model = tf.keras.models.load_model(model_path)
#         self.data_folder = test_image
#         self.feature_extraction_model = tf.keras.Model(inputs=self.model.input, outputs=self.model.layers[-3].output)
#         self.image_paths = []
#         self.labels = []
 
#     def preprocess_image(self, image_path):
#         """
#         Preprocess image for prediction and feature extraction.

#         Args:
#             image_path (str): Path of image to preprocess.

#         Returns:
#             predicted_class (str): The predicted class for the image.
#             test_feature_vector (numpy.ndarray): Feature vector extracted from the image.
#             img (numpy.ndarray): Preprocessed image data.
#             class_probabilities (numpy.ndarray): Class probabilities for the image.
#         """
#         img = load_img(image_path, target_size=(224, 224))
#         img = img_to_array(img)
#         img = preprocess_input(img)
#         img = np.expand_dims(img, axis=0)
#         # Extract the feature vector for the provided image
#         test_feature_vector = self.feature_extraction_model.predict(img)
#         print(self.feature_extraction_model.summary())
#         # Predict the class probability for the provided image
#         class_probabilities = self.model.predict(img)
#         predicted_probability = class_probabilities[0][0] 
#         predicted_class = 'Good' if predicted_probability >= 0.5 else 'Bad'
#         print(test_feature_vector)

#         return predicted_class, test_feature_vector, img, class_probabilities

# model_path = 'audio_model_2.h5'
# test_image = 'final_dataset/test/bad/IMG-20170615-WA0023.jpg'
# # test_image='final_dataset/test/good/IMG_20170614_122033442.jpg'
# image_finder = NearMissImageFinder(model_path, test_image)
# predicted_test_class, test_feature_vector, preprocessed_image, class_probabilities = image_finder.preprocess_image(test_image)

# print(f'Predicted class: {predicted_test_class}')
# print(f'Feature vector shape: {test_feature_vector.shape}, {class_probabilities}')

# #Load and preprocess the training set images and labels
# train_folder = 'final_dataset/train'


# def load_and_preprocess_training_data(train_folder, opposite_class):
#     """
#     Load and preprocess training data.

#     Args:
#         train_folder (str): Path to the training dataset.
#         opposite_class (str): The opposite train class to the predicted test class.

#     Returns:
#         train_data (dict): dictionary containing image paths, labels, and feature vectors.
#     """
#     train_data = {'image_paths': [], 'labels': [], 'feature_vectors': []}


#     for root, dirs, files in os.walk(train_folder):
#         for file in files:
#             if file.endswith(".jpg"):
#                 image_path = os.path.join(root, file)

#                 # Determine the label based on the folder structure (Good or Bad)
#                 label = "Good" if "good" in root.lower() else "Bad"

#                 if label == opposite_class:
#                     # Preprocess the image and extract feature vectors
#                     img = load_img(image_path, target_size=(224, 224))
#                     img = img_to_array(img)
#                     img = preprocess_input(img)
#                     img = np.expand_dims(img, axis=0)
#                     feature_vector = image_finder.feature_extraction_model.predict(img)

#                     train_data['image_paths'].append(image_path)
#                     train_data['labels'].append(label)
#                     train_data['feature_vectors'].append(feature_vector)

#     return train_data

# # Determine the opposite class based on the predicted test class
# opposite_class = "Bad" if predicted_test_class == "Good" else "Good"

# # Load and preprocess training data of the opposite class
# train_data= load_and_preprocess_training_data(train_folder, opposite_class)

# print(opposite_class)
# print(predicted_test_class)

# features=train_data['feature_vectors']
# labels=train_data['labels']

# output_file=f'extracted_feature{opposite_class}.pickle'

# cosine_similarities = {}
# for image_path, image_data in zip(train_data['image_paths'],train_data['feature_vectors']):

#         cosine_sim = cosine_similarity(test_feature_vector, image_data)
#         print(cosine_sim)


#         cosine_similarities[image_path] = cosine_sim[0][0]

# # Sort the training images by cosine similarity in descending order
# sorted_images = sorted(cosine_similarities.items(), key=lambda x: x[1],reverse=True)
# print(sorted_images[:5])

# # Display the test image
# plt.figure()
# plt.imshow(Image.open(test_image))
# plt.title(f"Test Image: {predicted_test_class}")
# plt.axis('off')
# plt.show()

# # Display the top 5 "near miss" images and their similarity scores
# top_k = 5
# top_images = sorted_images[:top_k]

# for i, (image_path, similarity_score) in enumerate(top_images):
#     img = Image.open(image_path)
    
#     label = train_data['labels'][train_data['image_paths'].index(image_path)]
#     plt.figure()
#     plt.imshow(img)
#     plt.title(f"Label: {label}, Similarity: {similarity_score:.2f}")
#     plt.axis('off')
#     plt.show()

# with open(output_file, 'wb') as f:
#     data = {
#         'features': features,
#         'labels': labels
#     }
#     pickle.dump(data, f)

# print(f"Extracted features and labels saved to {output_file}")

# # obj = pd.read_pickle(r'/Users/satyampant/Desktop/kogsys_project/kogsys-15-ects-proj-ss23-pant/extracted_feature.pickle')

import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np
from keras.utils import img_to_array
import librosa
from tensorflow.image import resize

class NearMissImageFinder:

    def __init__(self, model_path):
        """
        Initialize the NearMissImageFinder class
        Args:
            model_path (str): Path to the pre-trained model.
            test_image (str): Path to the test image for analysis.
        """
        self.model = tf.keras.models.load_model(model_path)
        # self.data_folder = test_image
        self.feature_extraction_model = tf.keras.Model(inputs=self.model.input, outputs=self.model.layers[-3].output)
        self.image_paths = []
        self.labels = []
 
    def preprocess_image(self, audio_path):
        """
        Preprocess image for prediction and feature extraction.

        Args:
            audio_path (str): Path of image to preprocess.

        Returns:
            predicted_class (str): The predicted class for the image.
            test_feature_vector (numpy.ndarray): Feature vector extracted from the image.
            img (numpy.ndarray): Preprocessed image data.
            class_probabilities (numpy.ndarray): Class probabilities for the image.
        """

        target_shape = (128, 128)
        audio_data, sample_rate = librosa.load(audio_path, sr=None)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        mel_spectrogram = tf.reshape(mel_spectrogram, (1,) + target_shape + (1,))
        # img = np.expand_dims(mel_spectrogram.numpy(), axis=0)

        # Extract the feature vector for the provided image
        test_feature_vector = self.feature_extraction_model(mel_spectrogram)
        # test feature vector if it's null
        # print("test")
        # # print(test_feature_vector[0].shape[0])
        # for i in range(test_feature_vector[0].shape[0]):
        #     if test_feature_vector[0][i].numpy() != 0:
        #         print(i)
        #         print(test_feature_vector[0][i])
        print(self.feature_extraction_model.summary())
        # Predict the class probability for the provided image
        class_probabilities = self.model.predict(mel_spectrogram)
        print("calss Probibilities:")
        print(class_probabilities)
        predicted_probability = class_probabilities[0][0] 
        predicted_class = 'normal' if predicted_probability >= 0.5 else 'abnormal'
        print(test_feature_vector)

        return predicted_class, test_feature_vector, mel_spectrogram, class_probabilities


    def load_and_preprocess_training_data(self,train_folder, opposite_class):
        """
        Load and preprocess training data.

        Args:
            train_folder (str): Path to the training dataset.
            opposite_class (str): The opposite train class to the predicted test class.

        Returns:
            train_data (dict): dictionary containing image paths, labels, and feature vectors.
        """
        train_data = {'audio_paths': [], 'labels': [], 'feature_vectors': []}


        for root, dirs, files in os.walk(train_folder):
            label = "abnormal" if "abnormal" in root.lower() else "normal"
            if label == opposite_class:
                for file in files:
                    if file.endswith(".wav"):
                        audio_path = os.path.join(root, file)

                        target_shape = (128, 128)
                        audio_data, sample_rate = librosa.load(audio_path, sr=None)
                        mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
                        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
                        mel_spectrogram = tf.reshape(mel_spectrogram, (1,) + target_shape + (1,))
                        # Extract the feature vector for the provided image
                        feature_vector = self.feature_extraction_model(mel_spectrogram)

                        train_data['audio_paths'].append(audio_path)
                        train_data['labels'].append(label)
                        train_data['feature_vectors'].append(feature_vector)

        return train_data


def calc_cosine_similarities(test_feature_vector,train_data):
    cosine_similarities = {}
    for audio_path, image_data in zip(train_data['audio_paths'],train_data['feature_vectors']):

            cosine_sim = cosine_similarity(test_feature_vector, image_data)
            print(cosine_sim)


            cosine_similarities[audio_path] = cosine_sim[0][0]
            
    return cosine_similarities
