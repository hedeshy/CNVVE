import os
import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import argparse

from tqdm import tqdm
import random

from torchaudio import sox_effects

import librosa
from librosa.core import resample
from scipy.io import wavfile

sox = False

def read_file(path):
    wav, sr = torchaudio.load(path)
    return wav, sr

def save_sample(wavetensor, rate, target_dir, fn, ix):
    fn = fn.split('.wav')[0]
    if (ix > 0):
        dst_path = os.path.join(target_dir.split('.')[0], fn+'_{}.wav'.format(str(ix)))
    else:
        dst_path = os.path.join(target_dir.split('.')[0], fn+'.wav')
    torchaudio.save(dst_path, wavetensor, rate)

def save_sample_librosa(waveform, rate, target_dir, fn, ix):
    fn = fn.split('.wav')[0]
    if (ix > 0):
        dst_path = os.path.join(target_dir.split('.')[0], fn+'_{}.wav'.format(str(ix)))
    else:
        dst_path = os.path.join(target_dir.split('.')[0], fn+'.wav')
    if os.path.exists(dst_path):
        return
    wavfile.write(dst_path, rate, waveform)



def check_dir(path):
    if os.path.exists(path) is False:
        os.mkdir(path)


def normalize(args):
    src_root = args.src_root
    dst_root = args.dst_root
    delta_sample = args.length * args.sr # 1 sec of 16khz sr 

    check_dir(dst_root)
    classes = os.listdir(src_root)

    for _cls in classes:
        target_dir = os.path.join(dst_root, _cls)
        check_dir(target_dir)
        src_dir = os.path.join(src_root, _cls)

        for fn in tqdm(os.listdir(src_dir)):
            src_fn = os.path.join(src_dir, fn)

            waveform, sample_rate = read_file(src_fn)
            length_signal = waveform.shape[1]
            duration = length_signal / args.sr

            if length_signal < delta_sample:
                num_missing_samples = delta_sample - length_signal
                last_dim_padding = (0, num_missing_samples)
                waveform = torch.nn.functional.pad(waveform, last_dim_padding)
                save_sample(waveform, args.sr, target_dir, fn, 0)
            else:

                if _cls == 'continuous':
                    offset = random.randint(0, length_signal - delta_sample)
                    waveform = waveform[..., offset:offset + delta_sample]
                    save_sample(waveform, args.sr, target_dir, fn, 0)
                else:    
                    
                    if sox == True : 
                        waveform, sr = sox_effects.apply_effects_tensor(waveform, [['speed', duration], ['rate',  args.sr]])
                        save_sample(waveform, args.sr, target_dir, fn, 0)
                    else:
                        waveform, sample_rate = librosa.load(src_fn, sr = args.sr)
                        length_signal = waveform.shape[0]
                        duration = length_signal / args.sr
                        waveform = librosa.effects.time_stretch(y=waveform, rate=duration)
                        save_sample_librosa(waveform, args.sr, target_dir, fn, 0)
                        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Cleaning audio data')
    parser.add_argument('--src_root', type=str, default='data/cleaned',
                        help='directory of audio files in total duration')
    parser.add_argument('--dst_root', type=str, default=None ,
                        help='directory to put the normalized audio files')
    parser.add_argument('--sr', type=int, default=16000, #default=16000
                        help='rate to downsample audio')
    parser.add_argument('--length', type=str, default=1, #1sec of 16ks sample rate
        help='length of the signal. default is 1sec')
    
    args, _ = parser.parse_known_args()
