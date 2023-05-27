import os
import re

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import argparse

from tqdm import tqdm
import random

import librosa
from librosa.core import resample
from scipy.io import wavfile

from helper.audio_cleaner import clean


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


def check_dir(path):
    if os.path.exists(path) is False:
        os.mkdir(path)


def save_sample_librosa(waveform, rate, target_dir, fn, ix):
    fn = fn.split('.wav')[0]
    if (ix > 0):
        dst_path = os.path.join(target_dir.split('.')[0], fn+'_{}.wav'.format(str(ix)))
    else:
        dst_path = os.path.join(target_dir.split('.')[0], fn+'.wav')
    if os.path.exists(dst_path):
        return
    wavfile.write(dst_path, rate, waveform)

def prepare(args):
    src_root = args.src_root
    dst_root = args.dst_root
    tl = args.tl

    check_dir(dst_root)
    classes = os.listdir(src_root)

    for _cls in classes:
        target_dir = os.path.join(dst_root, _cls)
        check_dir(target_dir)

        for sub_dir in os.listdir(os.path.join(src_root, _cls)):
            src_dir = os.path.join(src_root, _cls, sub_dir)

            for fn in tqdm(os.listdir(src_dir)):
                src_fn = os.path.join(src_dir, fn)

                if args.mode == 'sox' :
                    waveform, sample_rate = read_file(src_fn)
                    # Torch Vad only trim the end of the signal 
                    waveform_reversed, sample_rate = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, [["reverse"]])
                    vad = torchaudio.transforms.Vad(sample_rate=sample_rate, trigger_level=tl)
                    waveform_reversed_front_trim = vad(waveform_reversed)
                    waveform_end_trim, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
                        waveform_reversed_front_trim, sample_rate, [["reverse"]]
                    )
                    waveform = vad(waveform_end_trim)

                    resampler = torchaudio.transforms.Resample(sample_rate, args.sr)
                    resampler.to('cpu')
                    waveform = resampler(waveform)

                    save_sample(waveform, args.sr, target_dir, fn, 0)
                else:
                    waveform, sample_rate = librosa.load(src_fn, sr = args.sr)
                    if args.mode == 'librosa' :
                        clean_wav, index = librosa.effects.trim(y=waveform, top_db=args.threshold, ref = 1)
                    else:
                        clean_wav = clean(y=waveform, sr = sample_rate , intensity = 1)
                    if clean_wav.any():
                        if clean_wav.shape[0] < args.sr:
                            clean_wav = resample(clean_wav, orig_sr = sample_rate,  target_sr = args.sr)
                            save_sample_librosa(clean_wav, args.sr, target_dir, fn, 0)
                        else:
                            # We try a higher intensity
                            clean_wav = clean(y=clean_wav, sr = sample_rate , intensity = 3)
                            if clean_wav.any():
                                clean_wav = resample(clean_wav, orig_sr = sample_rate,  target_sr = args.sr)
                                save_sample_librosa(clean_wav, args.sr, target_dir, fn, 0)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Cleaning audio data')
    parser.add_argument('--mode', type=str, default='librosa',
                    help='vad, sox, or librosa')
    parser.add_argument('--src_root', type=str, default='data/raw',
                        help='directory of audio files in total duration')
    parser.add_argument('--dst_root', type=str, default='data/cleaned',
                        help='directory to put audio files split by delta_time')
    parser.add_argument('--sr', type=int, default=16000, #default=16000
                        help='rate to downsample audio')

    parser.add_argument('--tl', type=str, default=7.5, #default=7.5
                        help='trigger level for torch vad')
    parser.add_argument('--threshold', type=str, default=55, 
        help='threshold top db for librosa trim')
    args, _ = parser.parse_known_args()

    prepare(args)

