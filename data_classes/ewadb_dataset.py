import os
import torch
import torchaudio
import librosa
import pandas as pd
from transformers import AutoFeatureExtractor
from data_classes.wavelets_feature_extractor import WaveletFeatureExtractor
from audiomentations import Compose, PitchShift, PolarityInversion, Gain

class EWADBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        metadata_path,
        metadata_type,
        dataset_root_path,
        audio_path_key,
        label_key,
        model_name_or_path,
        label2id={ 'Healthy': 0, 'Parkinson': 1, 'HC': 0, 'PD': 1 },
        domain_id=None,
        is_test=False,
    ):
        '''
        Initializes the dataset.
        Args:
            metadata_path: Path to the metadata file.
            metadata_type: Type of the metadata file (csv, json).
            dataset_root_path: Root path of the dataset.
            audio_path_key: Key to the audio path in the metadata.
            label_key: Key to the label in the metadata.
            model_name_or_path: Name or path of the model.
            label2id: Mapping of label to id.
            domain_id: Id of the domain (if domain classification).
            is_test: ...
        '''
        self.config = config
        self.metadata_path = metadata_path
        self.metadata_type = metadata_type
        self.dataset_root_path = dataset_root_path
        self.audio_path_key = audio_path_key
        self.label_key = label_key
        self.model_name_or_path = model_name_or_path
        self.label2id = label2id
        self.domain_id = domain_id
        self.is_test = is_test
        self.data = self.load_metadata()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name_or_path)
        if self.config.data.wavelets:
            self.wavelet_feature_extractor = WaveletFeatureExtractor(
                frame_size=self.config.data.stft_params.win_length,
                frame_stride=self.config.data.stft_params.hop_length
            )
        self.type_mapping = {
            'speech': 0,
            'diadochokinetic': 1
        }
        
        if self.config.data.use_data_augmentation: 
            prob = self.config.data.augmentation_probability
            self.apply_augmentations = Compose([
                # pitch shift
                PitchShift(min_semitones=-4, max_semitones=4, p=prob),
                # polarity inversion
                PolarityInversion(p=prob),
                # random gain
                Gain(min_gain_in_db=-2, max_gain_in_db=2, p=prob),
            ])
        
    def load_metadata(self):
        '''
        Loads the metadata file.
        '''
        if self.metadata_type == 'csv':
            data = pd.read_csv(self.metadata_path)
        elif self.metadata_type == 'json':
            data = pd.read_json(self.metadata_path)
        elif self.metadata_type == 'tsv':
            data = pd.read_csv(self.metadata_path, sep='\t')
        else:
            raise ValueError(f'Invalid metadata type: {self.metadata_type}')
        
        return data
        
    def __len__(self):
        '''
        Returns the length of the dataset.
        '''
        return len(self.data)
    
    def _augment_data(self, audio, sample_rate):
        '''
        Augments the audio data.
        Args:
            audio: Audio tensor.
        '''
        audio = audio.unsqueeze(0)
        # to numpy
        audio = audio.numpy()
        audio = self.apply_augmentations(audio, sample_rate=sample_rate)
        # to tensor
        audio = torch.from_numpy(audio)
        return audio.squeeze(0)
    
    def load_audio(self, audio_path):
        '''
        Loads the audio file.
        Args:
            audio_path: Path to the audio file.
        '''
        audio_path = os.path.join(self.dataset_root_path, audio_path)
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != self.feature_extractor.sampling_rate:
            # resample the audio
            waveform = torchaudio.transforms.Resample(sample_rate, self.feature_extractor.sampling_rate)(waveform)

        # if stereo, convert to mono
        if waveform.shape[0] == 2:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        # remove first dimension
        waveform = waveform.squeeze()
            
        return waveform
    
    def get_random_crop(self, audio, max_length):
        '''
        Randomly crops the audio.
        Args:
            audio: Audio tensor.
            max_length: Maximum length of the audio.
        '''
        if audio.size(0) > max_length:
            start = torch.randint(0, audio.size(0) - max_length, (1,)).item()
            audio = audio[start:start + max_length]
            
        return audio
    
    def get_list_labels(self):
        labels = self.data[self.label_key]
        labels = [self.label2id[label] if label in self.label2id else label for label in labels]
        return labels
    
    def _extract_stft_features(self, audio):
        """
        Extracts STFT, MFCC, or Mel-spectrogram features from the audio signal.

        Args:
            audio (torch.Tensor): The audio signal.

        Returns:
            torch.Tensor: The extracted features, possibly with delta and delta-delta features concatenated.
        """
        # Define the common parameters
        audio_np = audio.numpy()
        stft_params = self.config.data.stft_params
        common_args = {
            'y': audio_np,
            'n_fft': stft_params.n_fft,
            'hop_length': stft_params.hop_length,
            'win_length': stft_params.win_length,
            'window': "hamming",
            'center': False
        }
        
        if stft_params.type == "standard":
            features = self._compute_stft_features(common_args)
        elif stft_params.type == "mfcc":
            features = self._compute_mfcc_features(common_args)
        elif stft_params.type == "mel":
            features = self._compute_mel_spectrogram_features(common_args)
        else:
            raise ValueError(f"Unsupported STFT feature type: {stft_params.type}")

        if stft_params.use_delta_and_delta_delta:
            features = self._add_delta_features(features)

        return features.permute(1, 0)

    def _compute_stft_features(self, common_args):
        """Computes the STFT features."""
        stft = librosa.stft(**common_args)
        magnitude = torch.abs(torch.from_numpy(stft))
        return magnitude

    def _compute_mfcc_features(self, common_args):
        """Computes the MFCC features."""
        common_args['sr'] = self.config.data.sample_rate
        common_args['n_mfcc'] = self.config.data.stft_params.n_mfcc
        mfcc = librosa.feature.mfcc(**common_args)
        return torch.from_numpy(mfcc)

    def _compute_mel_spectrogram_features(self, common_args):
        """Computes the Mel-spectrogram features."""
        common_args['sr'] = self.config.data.sample_rate
        common_args['n_mels'] = self.config.data.stft_params.n_mels
        mel_spectrogram = librosa.feature.melspectrogram(**common_args)
        return torch.from_numpy(mel_spectrogram)

    def _add_delta_features(self, features):
        """Adds delta and delta-delta features to the base features."""
        delta = compute_deltas(features)
        delta_delta = compute_deltas(delta)
        return torch.cat((features, delta, delta_delta), dim=0)
    
    def get_sample_type(self, idx):
        '''
        Returns the sample type.
        Args:
            idx: Index of the item.
        '''
        sample = self.data[self.audio_path_key][idx]
        # if readtext, read_text, sentence, monologue, sentence, picture - type: speech
        filters = ['readtext', 'read_text', 'sentence', 'monologue', 'picture']
        if any(f in sample for f in filters):
            return 'speech'
        else:
            # DDK, pataka, diadochokinetic - type: diadochokinetic
            return 'diadochokinetic'
    
    def __getitem__(self, idx):
        '''
        Returns the item at the given index.
        Args:
            idx: Index of the item.
        '''
        audio_path = self.data[self.audio_path_key][idx]
        audio = self.load_audio(audio_path)
        label = self.data[self.label_key][idx]
        
        # randomly crop the audio if longer
        if not self.is_test:
            audio = self.get_random_crop(audio, self.config.data.max_length_in_seconds * self.feature_extractor.sampling_rate)
            
        if self.config.data.use_data_augmentation and not self.is_test:
            audio = self._augment_data(audio, self.feature_extractor.sampling_rate)

        features = self.feature_extractor(
            audio,
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors='pt',
            max_length=self.config.data.max_length_in_seconds * self.feature_extractor.sampling_rate,
            padding=self.config.data.padding,
            truncation=self.config.data.truncation
        )
        
        # if config.data.magnitude true, extract features using stft
        if self.config.data.magnitude:
            stft_features = self._extract_stft_features(features['input_values'].squeeze())
        else: stft_features = None
        
        if self.config.data.wavelets:
            wavelet_features = self.wavelet_feature_extractor.extract_features(features['input_values'].squeeze())
            # print(f"Wavelet features shape: {wavelet_features.shape}")
        
        if label in self.label2id:
            label = self.label2id[label]
        else:
            label = label # maybe already an integer label
            
        return_dict = {
            'input_values': features['input_values'].squeeze(),
            # 'attention_mask': features['attention_mask'].squeeze(),
            'labels': torch.tensor(label)
        }
        
        if stft_features is not None:
            return_dict['magnitudes'] = stft_features
            # print(f"STFT features shape: {stft_features.shape}")
            
        if self.config.data.wavelets:
            return_dict['magnitudes'] = wavelet_features
            
        if stft_features is not None and self.config.data.wavelets:
            print(f"[INFO] Both STFT and wavelet features are present. Wavelet features will override STFT features. [CHANGE IT WHEN POSSIBLE]")
        
        if self.domain_id is not None:
            return_dict['domain_labels'] = torch.tensor(self.domain_id)
            
        # add type
        sample_type = self.get_sample_type(idx)
        return_dict['sample_type'] = torch.tensor(self.type_mapping[sample_type])
            
        return return_dict
        
if __name__ == '__main__':
    
    from addict import Dict
    
    config = {
        'data': {
            'max_length_in_seconds': 10
        }
    }
    config = Dict(config)
    
    dataset = EWADBDataset(
        config,
        metadata_path='/mnt/disk3/datasets/EWADB/standard_split/test_metadata.tsv',
        metadata_type='tsv',
        dataset_root_path='/mnt/disk3/datasets/EWADB/S0489/',
        audio_path_key='AUDIOFILE',
        label_key='DIAGNOSIS',
        model_name_or_path='facebook/wav2vec2-base-960h'
    )
    print(len(dataset))
    print(dataset[0])