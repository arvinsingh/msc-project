
import librosa
import matplotlib.pyplot as plt
import librosa.display



def visualize_mfcc_feature(audio_file, title, sample_rate=16000, n_mfcc=13):
    """
    Visualize the MFCC feature for a given label and index.
    """
    # load MFCC features for audio
    mfcc_features = librosa.feature.mfcc(y=librosa.load(audio_file, sr=sample_rate)[0], sr=sample_rate, n_mfcc=n_mfcc)
    # plot the MFCC features
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(mfcc_features, x_axis='time')
    plt.colorbar()
    plt.title(title)
    plt.show()