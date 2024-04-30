import numpy as np
import librosa
import sounddevice as sd
import matplotlib.pyplot as plt

def record_audio(duration, sr=16000):
    """
    Records audio from the default microphone for the specified duration at the given sample rate.

    Parameters:
    - duration : float
        Duration of the recording in seconds.
    - sr : int, optional
        Sample rate of the recording (default is 16000 Hz).

    Returns:
    - np.ndarray
        Recorded audio as a numpy array.
    """
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    print("Recording...")
    sd.wait()  # Wait until recording is finished
    return recording.flatten()  # Return as a 1D array

def audio_to_embedding(audio, sr=16000, plot=False):
    """
    Converts audio samples into 32-dimensional log-mel spectrogram features and optionally plots the spectrogram.

    Parameters:
    - audio : np.ndarray
        A numpy array containing the audio sample.
        Each sample should be a 32-bit float between -1 and 1.
    - sr : int, optional
        The sample rate of the audio (default is 16000 Hz).
    - plot : bool, optional
        Whether to plot the log-mel spectrogram (default is False).

    Returns:
    - np.ndarray
        A numpy array containing the log-mel spectrogram features of shape (time_steps, 32).
    """
    # Constants for the STFT and Mel transformation
    n_fft = int(0.025 * sr)  # 25ms window size
    hop_length = int(0.010 * sr)  # 10ms step size
    fmin = 60  # minimum frequency for mel scale
    fmax = 3800  # maximum frequency for mel scale
    n_mels = 32  # number of mel frequency bins

    # Compute the magnitude spectrogram from the audio
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    
    # Convert to mel scale
    mel_filter = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel_spec = np.dot(mel_filter, S**2)
    
    # Compute the logarithm of the mel spectrogram
    log_mel_spec = librosa.power_to_db(mel_spec)
    
    if plot:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(log_mel_spec, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', fmin=fmin, fmax=fmax)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Log-Mel Spectrogram')
        plt.savefig('log_mel_spectrogram.png')
    
    return log_mel_spec.T  # transpose to have shape (time_steps, 32)

# Example usage:
if __name__ == "__main__":
    # Record 1.98 seconds of audio
    duration = 1.980
    sr = 16000
    audio = record_audio(duration, sr)
    embeddings = audio_to_embedding(audio, sr, plot=True)
    print("Shape of embeddings:", embeddings.shape)
    print("Embeddings data example:", embeddings)
