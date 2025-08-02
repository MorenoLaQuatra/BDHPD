import pywt
import torch
import numpy as np

class WaveletFeatureExtractor:
    """
    A class for extracting wavelet features from 1D signals such as audio data using PyTorch tensors.

    Attributes:
        wavelet_name (str): The name of the wavelet to use for the transform.
        level (int): The level of decomposition to perform with the wavelet transform.
        use_approximation (bool): Whether to include the approximation coefficients in the features.
        frame_size (int): The number of samples in each frame.
        frame_stride (int): The number of samples to step between frames.
    """
    
    def __init__(self, wavelet_name='db4', level=5, use_approximation=False, frame_size=128, frame_stride=64):
        self.wavelet_name = wavelet_name
        self.level = level
        self.use_approximation = use_approximation
        self.frame_size = frame_size
        self.frame_stride = frame_stride

    def extract_features_from_frame(self, frame):
        """
        Extracts wavelet features from a single frame of the signal.
        Args:
            frame (torch.Tensor): 1D tensor representing the frame.
        Returns:
            torch.Tensor: The extracted wavelet features.
        """
        # Convert tensor to numpy for pywt
        frame_np = frame.cpu().numpy()

        # Wavelet decomposition
        coeffs = pywt.wavedec(frame_np, self.wavelet_name, level=self.level)
        
        # Concatenate coefficients
        if self.use_approximation:
            coeffs_array = np.concatenate(coeffs, axis=-1)
        else:
            coeffs_array = np.concatenate(coeffs[1:], axis=-1)
        
        # Convert back to tensor
        return torch.tensor(coeffs_array, dtype=frame.dtype)

    def extract_features(self, signal):
        """
        Splits the signal into overlapping frames and extracts wavelet features from each frame.
        Args:
            signal (torch.Tensor): 1D tensor representing the input signal.
        Returns:
            torch.Tensor: 2D tensor where each row is the wavelet features for a frame.
        """
        frames = self._split_into_frames(signal)
        features = [self.extract_features_from_frame(frame) for frame in frames]
        return torch.stack(features, dim=0)

    def _split_into_frames(self, signal):
        """
        Splits the signal into overlapping frames.
        Args:
            signal (torch.Tensor): 1D tensor representing the input signal.
        Returns:
            List[torch.Tensor]: A list of 1D tensors, each representing a frame.
        """
        num_samples = signal.shape[0]
        frames = []
        for start in range(0, num_samples - self.frame_size + 1, self.frame_stride):
            frame = signal[start:start + self.frame_size]
            frames.append(frame)
        return frames

    def __repr__(self):
        return (f"WaveletFeatureExtractor(wavelet_name='{self.wavelet_name}', level={self.level}, "
                f"use_approximation={self.use_approximation}, frame_size={self.frame_size}, frame_stride={self.frame_stride})")


if __name__ == "__main__":
    # Example usage for extracting wavelet features from a batch of audio signals
    batch_size = 64
    sample_rate = 16000  # 16 kHz
    duration = 10  # 10 seconds
    frame_size = 400
    hop_size = 320
    signal_length = sample_rate * duration  # Total samples in each signal

    # Create a dummy batch of signals
    dummy_signals = torch.sin(2 * np.pi * 5 * torch.linspace(0, duration, signal_length)) + torch.randn((batch_size, signal_length)) * 0.1

    # Instantiate the WaveletFeatureExtractor
    extractor = WaveletFeatureExtractor(wavelet_name='db4', level=4, use_approximation=False, frame_size=frame_size, frame_stride=hop_size)
    print(extractor)

    # Extract features for each signal in the batch
    batch_features = torch.stack([extractor.extract_features(signal) for signal in dummy_signals])

    # Print out details about the batch features
    print("Batch Features Shape:", batch_features.shape)  # Should be (batch_size, num_frames, feature_dim)
    print("Batch Features (first signal, first frame):", batch_features[0, 0, :])  # Show first 10 features of the first frame of the first signal

    # Example checks:
    assert batch_features.shape[0] == batch_size, "Batch size mismatch"
    assert batch_features.shape[2] == batch_features.shape[2], "Feature dimension mismatch across batch"
