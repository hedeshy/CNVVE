import os
import re
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import torchaudio

class NVVIDataset(Dataset):

    def __init__(self,
                 annotations_file,
                 audio_dir,
                 target_sample_rate,
                 device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.target_sample_rate = target_sample_rate
        self.samples = []

        self.load_samples()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        return self.samples[index]

    def _resample_if_necessary(self, signal, sr, device):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            resampler.to(device)
            signal = resampler(signal)
        return signal

    def _to_mono_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        path = self.annotations.iloc[index, 3]
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 1]
    
    def load_samples(self):
        for index in range(len(self.annotations)):
            audio_sample_path = self._get_audio_sample_path(index)
            label = self._get_audio_sample_label(index)
            signal, sr = torchaudio.load(audio_sample_path)
            signal = signal.to(self.device)
            signal = self._resample_if_necessary(signal, sr, self.device)
            signal = self._to_mono_if_necessary(signal)
            self.samples.append((signal, label))
        
        random.shuffle(self.samples)
    
    def transform(self, wav, sr, device):
        n_fft = 512
        win_length = None
        hop_length = 256
        n_mels = 96
        n_mfcc = 64

        
        # mfcc_transform = torchaudio.transforms.MFCC(
        #     sample_rate=sr,
        #     n_mfcc=n_mfcc,
        #     melkwargs={
        #         "n_fft": n_fft,
        #         "n_mels": n_mels,
        #         "hop_length": hop_length,
        #         "mel_scale": "htk",
        #     },
        # )
        # mfcc_transform = mfcc_transform.to(device)
        # # mfcc = mfcc_transform(wav)

        tform = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                             n_fft=n_fft,
                                             hop_length=hop_length,
                                             n_mels=n_mels,
                                             norm = "slaney"),
        torchaudio.transforms.AmplitudeToDB()).to(device)  
        # The transform needs to live on the same device as the model and the data.
        feature = tform(wav)


        return feature

if __name__ == "__main__":
    ANNOTATIONS_FILE = "./metadata.csv"
    AUDIO_DIR = "./data/padded"
    SAMPLE_RATE = 16000

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    usd = NVVIDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            SAMPLE_RATE,
                            device)
    print(f"There are {len(usd)} samples in the dataset.")


