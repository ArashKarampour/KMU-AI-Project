import argparse
from pathlib import Path

import cv2
import librosa
import numpy as np
from tqdm import tqdm
import skimage.io


def list_files(source):
    """
    List all files in the given source directory and its subdirectories.

    Args:
        source (str): The source directory path.

    Returns:
        list: A list of `Path` objects representing the files.
    """
    path = Path(source)
    files = [file for file in path.rglob('*') if file.is_file()]
    return files


def audio_to_spectrogram(audio_path, save_path):
    """
    Convert an audio file to a spectrogram and save it as an image.

    Args:
        audio_path (str): The path to the audio file.
        save_path (str): The path to save the spectrogram image.
        duration (int): Duration of the audio file to process in seconds.

    Returns:
        None
    """
    # Load audio file
    y, sr = librosa.load(audio_path,sr=None)

    # Compute spectrogram
    D = librosa.stft(y)
    S = librosa.amplitude_to_db(abs(D), ref=np.max)

    # Normalize values to 0-255 range and convert to uint8
    S = cv2.normalize(S, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Convert to RGB and save as PNG
    S = cv2.cvtColor(S, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(save_path, S)

# def scale_minmax(X, min=0.0, max=1.0):
#     X_std = (X - X.min()) / (X.max() - X.min())
#     X_scaled = X_std * (max - min) + min
#     return X_scaled

# def spectrogram_image(y, sr, out, hop_length, n_mels):
#     # use log-melspectrogram
#     mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
#                                             n_fft=hop_length*2, hop_length=hop_length)
#     mels = np.log(mels + 1e-9) # add small number to avoid log(0)

#     # min-max scale to fit inside 8-bit range
#     img = scale_minmax(mels, 0, 255).astype(np.uint8)
#     img = np.flip(img, axis=0) # put low frequencies at the bottom in image
#     img = 255-img # invert. make black==more energy

#     # save as PNG
#     skimage.io.imsave(out, img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='/data/knoblach/Audio_Classification/test/train/abnormal', help='source folder')
    parser.add_argument('--output', type=str, default='/data/karampour/trainSpectogrammImages/abnormal', help='folder output')
    opt = parser.parse_args()
    source, output = opt.source, opt.output

    file_list = list_files(source)

    for file in tqdm(file_list):
        # Output path
        new_path = Path(str(file).replace(str(source), output))

        # Create output directory
        new_path.parent.mkdir(parents=True, exist_ok=True)

        # Replace suffix
        new_path = new_path.with_suffix('.png')

        # Convert
        audio_to_spectrogram(str(file), str(new_path))