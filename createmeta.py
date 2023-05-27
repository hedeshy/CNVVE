import os
from glob import glob
import csv
import argparse
import json

with open("config.json") as json_file:
    config = json.load(json_file)

def createMeta(args):


    AUDIO_DIR = "./data/" + args.mode

    header = ['filename', 'classID', 'class', 'filepath']

    class_mapping = config["CLASS_MAPPING"]

    wav_paths = glob('{}/**'.format(AUDIO_DIR), recursive=True)
    wav_paths = sorted([x.replace(os.sep, '/') for x in wav_paths if '.wav' in x])

    classes = sorted(os.listdir(AUDIO_DIR))

    data = []
    for wav_fn in wav_paths:
        filename = wav_fn.split('/')[-1]
        classname = os.path.dirname(wav_fn).split('/')[-1]
        classID = class_mapping.index(classname)
        path = wav_fn
        data.append([filename, classID, classname, path  ])

    with open('./data/'+ args.mode +'_metadata.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Creating the meta')
    parser.add_argument('--mode', type=str, default='padded',
                        help='padding, augmented, etc.')
    args, _ = parser.parse_known_args()

    createMeta(args)
