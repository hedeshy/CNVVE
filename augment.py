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


def add_noise(wav, noise, noise_sr, snr_db = 10 ):
    audio_length = wav.shape[-1]
    noise_length = noise.shape[-1]
    if noise_length > audio_length:
        offset = random.randint(0, noise_length-audio_length)
        noise = noise[..., offset:offset+audio_length]
    elif noise_length < audio_length:
        noise = torch.cat([noise, torch.zeros((noise.shape[0], audio_length-noise_length))], dim=-1)

    wav_rms = wav.norm(p=2)
    noise_rms = noise.norm(p=2)
    
    # Scale the loudness of the noise level (dB)
    snr = 10 ** (snr_db / 20)
    scale = snr * noise_rms / wav_rms
    augmented = (scale * wav + noise) / 2
    return augmented


def augment(args):
    src_root = args.src_root
    dst_root = args.dst_root
    dt = args.delta_time
    delta_sample = int(dt*args.sr)

    check_dir(dst_root)
    classes = os.listdir(src_root)

    for _cls in classes:
        target_dir = os.path.join(dst_root, _cls)
        check_dir(target_dir)
        src_dir = os.path.join(src_root, _cls)
        for fn in tqdm(os.listdir(src_dir)):
            src_fn = os.path.join(src_dir, fn)
            wav, rate = read_file(src_fn)

            resampler = torchaudio.transforms.Resample(rate, args.sr)
            resampler.to('cpu')
            wav = resampler(wav)

            save_sample(wav, args.sr, target_dir, fn, 0)

            fn = fn.split('.wav')[0]

            # Augmentation 1: Background Noise
            # 
            # Taken from torchaudio assets, people talking in the background
            # https://download.pytorch.org/torchaudio/tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav
            noise1Wav, noise1Rate = read_file('./assets/noise/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav')
            augmented1_1 = add_noise(wav, noise1Wav, noise1Rate, 10)
            save_sample(augmented1_1, args.sr, target_dir, fn+'_{}.wav'.format(str('noise1')) , 0)

            
            # Urban ambience
            noise2Wav, noise2Rate = read_file('./assets/noise/urban-city-ambience_94bpm.wav')
            augmented1_2 = add_noise(wav, noise2Wav, noise2Rate, 10)
            save_sample(augmented1_2, args.sr, target_dir, fn+'_{}.wav'.format(str('noise2')) , 0)


            # Nature background noise
            noise3Wav, noise3Rate = read_file('./assets/noise/ambient-backyard-noise-filed-recording_81bpm_G_major.wav')
            augmented1_3 = add_noise(wav, noise3Wav, noise3Rate, 10)
            save_sample(augmented1_3, args.sr, target_dir, fn+'_{}.wav'.format(str('noise3')) , 0)


            # Fan noise
            noise4Wav, noise4Rate = read_file('./assets/noise/fan-in-background-fx_139bpm_F_major.wav')
            augmented1_4 = add_noise(wav, noise4Wav, noise4Rate, 10)
            save_sample(augmented1_4, args.sr, target_dir, fn+'_{}.wav'.format(str('noise4')) , 0)

            # Rain background
            noise5Wav, noise5Rate = read_file('./assets/noise/background-rain-fx_76bpm_F_minor.wav')
            augmented1_5 = add_noise(wav, noise5Wav, noise5Rate, 10)
            save_sample(augmented1_5, args.sr, target_dir, fn+'_{}.wav'.format(str('noise5')) , 0)

            # White noise background
            noise6Wav, noise6Rate = read_file('./assets/noise/white-noise-vinyl_C_minor.wav')
            augmented1_6 = add_noise(wav, noise6Wav, noise6Rate, 10)
            save_sample(augmented1_6, args.sr, target_dir, fn+'_{}.wav'.format(str('noise6')) , 0)


            # Augmentation 2: Pitch shifting
            # Number of steps needs to be tuned search space [4,5,6] should be a good starting point
            # seems to be a bit slow, might not be suitable during training
            for n_step in [3, 4, 5, 6]:
                augmented2_shifted_up = torchaudio.functional.pitch_shift( waveform =wav, sample_rate = rate, n_steps = n_step)
                save_sample(augmented2_shifted_up, args.sr, target_dir, fn+'_{}.wav'.format(str('shifted_up') + str(n_step)) , 0)

                augmented2_shifted_down = torchaudio.functional.pitch_shift( waveform =wav, sample_rate = rate, n_steps = -n_step)
                save_sample(augmented2_shifted_down, args.sr, target_dir, fn+'_{}.wav'.format(str('shifted_down') + str(n_step)) , 0)


            # Augmentation 3: Room Impulse Response
            # https://download.pytorch.org/torchaudio/tutorial-assets/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo-8000hz
            rir_raw, rirSR = read_file('./assets/noise/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav')
            rir = rir_raw[:, int(rirSR * 1.01) : int(rirSR * 1.3)]
            rir = rir / torch.norm(rir, p=2)
            RIR = torch.flip(rir, [1])
            wav_ = torch.nn.functional.pad(wav, (RIR.shape[1] - 1, 0))
            augmented3 = torch.nn.functional.conv1d(wav_[None, ...], RIR[None, ...])[0]

            save_sample(augmented3, args.sr, target_dir, fn+'_{}.wav'.format(str('rir')) , 0)


            # Augmentation 4: Loudness
            # Valuse to be checked: [  ±5,  ±10,  ±15]
            for gain in [5, 10 ,15]:
                quieter = torchaudio.transforms.Vol(gain=-gain, gain_type="db")
                quieter_waveform = quieter(wav)
                save_sample(quieter_waveform, args.sr, target_dir, fn+'_{}.wav'.format(str('quieter')+str(gain)) , 0)

                louder = torchaudio.transforms.Vol(gain=gain, gain_type="db")
                louder_waveform = louder(wav)
                save_sample(louder_waveform, args.sr, target_dir, fn+'_{}.wav'.format(str('louder')+str(gain)) , 0)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Augmenting audio data')
    parser.add_argument('--src_root', type=str, default='data/stretched',
                        help='directory of audio files to be augmented. data/stretched or data/padded')
    parser.add_argument('--dst_root', type=str, default=None,
                        help='directory to output the audio augmented data. data/stretched_augmented or data/padded_augmented')
    parser.add_argument('--delta_time', '-dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=16000, #default=16000
                        help='rate to downsample audio')

    args, _ = parser.parse_known_args()

    if args.src_root == 'data/stretched':
        args.dst_root = 'data/stretched_augmented'
    else:
        args.dst_root = 'data/padded_augmented'

    augment(args)

