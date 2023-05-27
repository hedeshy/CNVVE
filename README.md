## CNVVE: Dataset and Benchmark for Classifying Non-verbal Voice Expressions


The CNVVE Dataset is consists of 950 audio samples encompassing six distinct classes of voice expressions. These expressions were collected from 42 generous individuals who donated their voice recordings for the study. By making the dataset publicly accessible, we hope to facilitate further research and development of computational methods for non-verbal voice-based interactions.

### Setting up the Environment

```
pip install virtualenv
virtualenv CNVVE
activate CNVVE
pip install -r requirements.txt
```

### Steps to train and make the model

1. Create a new folder call it data and copy the raw audio files in it under raw folder.

1. Run `python clean.py` in the terminal
    - This script will trim the empty trailing signals and place the cleaned samples under cleaned folder.

1. Run `python normalize.py`in the terminal
    - By default this script normalized the data using zero padding technique. Run `python normalize.py --mode=padding` to normlize the data using padding technique.

1. Run `python augment.py` in the terminal
    - By by default it uses the padded data. Run `python normalize.py --src_root=data/padded --dst_root=data/padded_augmented` to augment the padded data.
    - Note: The created virtualenv ought to be active. 

1. Run `python createmeta.py --mode=padding` to create the metadata.csv

1. Run `python train.py` to start training the model. Note the training modes in training configuration down below.



#### Training Configurations

Various modes of training can be used by modifying the `config.json`.
- The  variable `train_mode`  has to be modified for
    - `tn`: Normal training mode.
    - `thp`: Hyperparameter search training mode, also modify `search_space.json` accordingly.
    - `tkf`: Training with kfold evaluation, modify `kfold_num` for number of folds accordingly.
    - `tp`: Training in production mode so the dataset is not split.
    - `tea`: Training in evaluation for augmentation, for this also do the following below steps. 
- For training with augmented dataset the following variables have to be set
    - `is_aug`: This has to point to `true`.
    - `AUDIO_DIR`: This has to point to the new augmented dataset path. 
    - `ANNOTATIONS_FILE`: This has to point to the newly generated metadata csv file. 
