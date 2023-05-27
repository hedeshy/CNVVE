import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader

import json
import sys
import os
import datetime
import random
from pathlib import Path
import pandas as pd


import matplotlib.pyplot as plt
import IPython.display as ipd
import numpy as np
from sklearn.model_selection import KFold


from tqdm import tqdm
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from itertools import cycle

from sklearn.metrics import confusion_matrix
import seaborn as sns
from dataset import NVVIDataset
from models import CNN2D


DEVICE = "cuda:3"

training_dict = {}
dataset_dict = {}

with open("config.json") as json_file:
    config = json.load(json_file)

random.seed(config["random_seed"])
torch.manual_seed(config["random_seed"])

def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

def train(model, epoch, log_interval, train_loader,  loss_fn, optimizer, pbar_update):
    
    model.train()
    batch_losses = []
    t_correct = 0

    for batch_idx, (data, target) in enumerate(dataset_dict["train_loader"]):

        data = data.to(DEVICE)
        target = target.to(DEVICE)

        # apply transform and model on whole batch directly on device
        data = transform(data)

        output = model(data)

        loss = loss_fn(output, target)

        pred  = get_likely_index(output)
        t_correct += number_of_correct(pred, target)

        if config["EPOCHS"] == epoch + 1:
            training_dict["y_pred_train"].extend(pred.detach().tolist())
            training_dict["y_train"].extend(target.detach().tolist())


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        # update progress bar
        training_dict["pbar"].update(pbar_update)
        # record loss
        batch_losses.extend([loss.item()]*target.shape[0])

    training_dict["train_losses"].append(np.mean(batch_losses))
    training_dict["train_accuracies"].append(t_correct/len(train_loader))


def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


def test(model, epoch):
    model.eval()
    correct = 0
    batch_losses = []
    for data, target in dataset_dict["test_loader"]:

        data = data.to(DEVICE)
        target = target.to(DEVICE)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        loss = training_dict["loss_fn"](output, target)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

        if config["EPOCHS"] == epoch + 1:
            training_dict["y_pred_test"].extend(pred.detach().tolist())
            training_dict["y_test"].extend(target.detach().tolist())

        # update progress bar
        training_dict["pbar"].update(training_dict["pbar_update"])
        batch_losses.extend([loss.item()]*target.shape[0])

    training_dict["test_losses"].append(np.mean(batch_losses))
    training_dict["test_accuracies"].append(correct/len(dataset_dict["test_loader"]))
    print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(dataset_dict['test_loader'])} ({100. * correct / len(dataset_dict['test_loader']):.0f}%)\n")


def transform(wav):

    tform = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=config["SAMPLE_RATE"],
                                         n_fft=config["N_FFT"],
                                         hop_length=config["HOP_LEN"],
                                         n_mels=config["N_MELS"]),
    torchaudio.transforms.AmplitudeToDB()).to(DEVICE)  
    # The transform needs to live on the same device as the model and the data.
    feature = tform(wav)
    return feature


def plot_losses(metric="loss", save_path=None):

    epochs = [epoch for epoch in range(len(training_dict["train_losses"]))]

    plt.close('all')
    fig, ax = plt.subplots(tight_layout=True)

    if metric == "loss":
        p1,  = ax.plot(epochs, training_dict["train_losses"],  color='#00017a', marker='o', label='Train Loss')
        p2,  = ax.plot(epochs, training_dict["test_losses"],  color='#eecc16', marker='o', label='Test Loss')
        plt_title =  'Loss Plot'
        ax.set_xlabel("Epoch Count")
        ax.set_ylabel("cross entropy loss")
    elif metric == "accuracy":
        p1,  = ax.plot(epochs, training_dict["train_accuracies"],  color='#00017a', marker='o', label='Train Accuracy')
        p2,  = ax.plot(epochs, training_dict["test_accuracies"],  color='#eecc16', marker='o', label='Test Accuracy')
        plt_title =  'Accuracy Plot'
        ax.set_xlabel("Epoch Count")
        ax.set_ylabel("Accuracy")
    
    
    ax.set_title(plt_title)

    ax.legend(handles=[p1, p2], title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)

    plt.show()

    fig.savefig(save_path / f"{metric}.png")

def plot_confusion_matrix(metric="confusion_matrix", save_path=None):
    c = 0
    expected_labels = []
    predicted_labels = []
    for i in range(0, len(test_dataset)):
        wav, target = test_dataset[i][0], test_dataset[i][1]
        wav = wav.to(DEVICE)
        input = transform(wav)
        input.unsqueeze_(0)
        output, expected = predict(training_dict["model"], input, target )
        if output != expected:
            c = c + 1

        expected_labels.append(expected)
        predicted_labels.append(output)

        print(f"Data point #{i}. Expected: {expected}. Predicted: {output}. Accuracy: { (((i+1)-c)/(i+1))*100}")
    print("number of wrong predictions:", c)

    cm = confusion_matrix(expected_labels, predicted_labels)
    cm_df = pd.DataFrame(cm,
                     index = config["CLASS_MAPPING"], 
                     columns = config["CLASS_MAPPING"])
    
    fig, ax = plt.subplots(tight_layout=True)

    sns.heatmap(cm_df, annot=True, fmt='.0f',)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()
    fig.savefig(save_path / f"{metric}.png")

def predict(model, input, target):
    with torch.no_grad():
        predictions = model(input)
        predicted_index  = get_likely_index(predictions)
        predicted = config["CLASS_MAPPING"][predicted_index]
        expected = config["CLASS_MAPPING"][target]
    return predicted, expected

def init_and_train_model(*params_list):
    global training_dict, config

    if not params_list is None and len(params_list) > 0:
        params = params_list[0]

        for key in params:
            if key in config:
                config[key] = params[key]

    training_dict["config_num"] += 1
    config_path = reports_path / f"config{training_dict['config_num']}"
    config_path.mkdir(exist_ok=True, parents=True)

    # model = CNN1D(n_input=1, n_output=len(labels))
    #CNN2D, CRNN
    training_dict["model"] = CNN2D()
    training_dict["model"].to(DEVICE)
    print(training_dict["model"])

    n = count_parameters(training_dict["model"])
    print("Number of parameters: %s" % n)

    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?highlight=crossentropyloss 
    training_dict["loss_fn"]= nn.CrossEntropyLoss()

    training_dict["optimizer"] = optim.Adam(training_dict["model"].parameters(), lr=config["LEARNING_RATE"], weight_decay=0.0001)
    training_dict["scheduler"] = optim.lr_scheduler.StepLR(training_dict["optimizer"], step_size=20, gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10
   
    training_dict["log_interval"] = 20
 
    training_dict["pbar_update"] = 1 / (len(dataset_dict["train_loader"]) + len(dataset_dict["test_loader"]))
    training_dict["train_losses"] = []
    training_dict["test_losses"] = []
    training_dict["train_accuracies"] = []
    training_dict["test_accuracies"] = []

    with tqdm(total=config["EPOCHS"]) as training_dict["pbar"]:
        for epoch in range(1, config["EPOCHS"] + 1):
            train(training_dict["model"], epoch, training_dict["log_interval"], dataset_dict["train_loader"],  training_dict["loss_fn"], training_dict["optimizer"], training_dict["pbar_update"])
            test(training_dict["model"], epoch)
            training_dict["scheduler"].step()

    plot_losses(metric="loss", save_path=config_path)
    plot_losses(metric="accuracy", save_path=config_path)

    # get a sample to pass to the jit trace
    wav, target = ds[0][0], ds[0][1]
    wav = wav.to(DEVICE)
    input = transform(wav)
    input.unsqueeze_(0)

    traced_model = torch.jit.trace(training_dict["model"], input)
    traced_model.save(config_path /  "torch_model_traced.pt")

    with open(config_path / "hyperparams.json", "w") as json_file:
        json_file.write(json.dumps(config, indent=2))
    
    with open(config_path / "acc.txt", "w") as f:
        f.write(f"The test accuracy is {training_dict['test_accuracies'][-1]}")

    return training_dict["train_losses"][-1]


def perf_kfold_validation():
    
    kfold = KFold(n_splits=config["kfold_num"], shuffle=True)
    kfold_results = {}

    for fold, (train_ids, test_ids) in enumerate(kfold.split(ds)):

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        kfold_results[fold] = {}

        # Define data loaders for training and testing data in this fold
        dataset_dict["train_loader"] = torch.utils.data.DataLoader(
                        ds, 
                        batch_size=config["BATCH_SIZE"],
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        sampler=train_subsampler)

        dataset_dict["test_loader"] = torch.utils.data.DataLoader(
                        ds,
                        batch_size=config["BATCH_SIZE"],
                        num_workers=num_workers,
                        pin_memory=pin_memory, 
                        sampler=test_subsampler)
        
        init_and_train_model()
        
        kfold_results[fold]["train_losses"] = training_dict["train_losses"]
        kfold_results[fold]["test_losses"] = training_dict["test_losses"]
        kfold_results[fold]["train_accuracies"] = training_dict["train_accuracies"]
        kfold_results[fold]["test_accuracies"] = training_dict["test_accuracies"]

    training_dict["train_losses"] = np.zeros(len(training_dict["train_losses"] ))
    training_dict["test_losses"] = np.zeros(len(training_dict["train_losses"] ))
    training_dict["train_accuracies"] = np.zeros(len(training_dict["train_accuracies"]))
    training_dict["test_accuracies"] = np.zeros(len(training_dict["test_accuracies"]))


    for i in range(len(training_dict["train_losses"] )):
        
            
        tot_train_l = 0
        tot_test_l = 0
        tot_train_acc = 0
        tot_test_acc = 0


        for k1, v in kfold_results.items():
            tot_train_l += v["train_losses"][i]
            tot_test_l +=  v["test_losses"][i]
            tot_train_acc +=  v["train_accuracies"][i]
            tot_test_acc +=  v["test_accuracies"][i]

        training_dict["train_losses"][i] += tot_train_l
        training_dict["test_losses"][i] += tot_test_l
        training_dict["train_accuracies"][i] += tot_train_acc
        training_dict["test_accuracies"][i] += tot_test_acc

     
    training_dict["train_losses"] /= len(kfold_results)
    training_dict["test_losses"] /= len(kfold_results)
    training_dict["train_accuracies"] /= len(kfold_results)
    training_dict["test_accuracies"] /= len(kfold_results) 

    plot_losses(metric="loss", save_path=reports_path)
    plot_losses(metric="accuracy", save_path=reports_path)


    print(f"The average train loss is: {tot_train_l / len(kfold_results.items())} ")
    print(f"The average test loss is: {tot_test_l / len(kfold_results.items())} ")
    print(f"The average train accuracy is: {tot_train_acc / len(kfold_results.items())} ")
    print(f"The average test accuracy is: {tot_test_acc / len(kfold_results.items())} ")


def eval_augs():
    met_og_df = pd.read_csv(config["ANNOTATIONS_FILE"])  
    met_df = met_og_df
    aug_types = ["rir"]

    for n_step in [3, 4, 5, 6]:
        aug_types.append(f"shifted_up{n_step}")
        aug_types.append(f"shifted_down{n_step}")
    
    for gain in [5, 10 ,15]:
        aug_types.append(f"quieter{gain}")
        aug_types.append(f"louder{gain}")

    for i in range(1, 7):
        aug_types.append(f"noise{i}")
    
    result_dict = {}

    met_df["include"] = [False for _ in range(len(met_og_df))]
    met_df["non_aug"] = [not(any(x in row["filename"] for x in aug_types)) for index, row in met_df.iterrows()]

    aug_types.append("none")
    aug_types.append("all")
    aug_types.append("test")


    test_list = random.sample(list(met_df[met_df["non_aug"] == True].filename), int(0.3*len(met_df[met_df["non_aug"] == True])))
    met_df["is_test"] = [any([fname_test[0:fname_test.find('_0.wav')] in row["filename"] for fname_test in test_list]) for index, row in met_df.iterrows()]
    met_df["is_train"] = [not value for value in met_df["is_test"]]

    # Test set without aug
    met_df["include"] = met_df["is_test"] & met_df["non_aug"]
    met_og_df[met_df["include"]].to_csv("test.csv", index = False, encoding='utf-8')
    test_dataset = NVVIDataset("test.csv",
                    config["AUDIO_DIR"],
                    config["SAMPLE_RATE"],
                    DEVICE)

    # Test set with aug
    met_df["include"] = met_df["is_test"]
    met_og_df[met_df["include"]].to_csv("testaug.csv", index = False, encoding='utf-8')
    test_dataset_aug = NVVIDataset("testaug.csv",
                    config["AUDIO_DIR"],
                    config["SAMPLE_RATE"],
                    DEVICE)



    for aug_type in aug_types:
        
        config["aug_type"] = aug_type

        # Train dataset
        met_df["include"] = [aug_type in fname or aug_type == "all" for fname in met_df["filename"]]
        met_df["include"] = (met_df["include"] |  met_df["non_aug"] ) &  met_df["is_train"]
        met_og_df[met_df["include"]].to_csv("train.csv", index = False, encoding='utf-8')
        train_dataset = NVVIDataset("train.csv",
                        config["AUDIO_DIR"],
                        config["SAMPLE_RATE"],
                        DEVICE)

        dataset_dict["train_loader"] = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config["BATCH_SIZE"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        if aug_type == "test":
            dataset_dict["test_loader"] = torch.utils.data.DataLoader(
                test_dataset_aug,
                batch_size=config["BATCH_SIZE"],
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        else:
            dataset_dict["test_loader"] = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=config["BATCH_SIZE"],
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        init_and_train_model()

        result_dict[config["aug_type"]] = training_dict["test_accuracies"][-1]

if __name__ == "__main__":

    if DEVICE == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    training_dict["model"] = None
    training_dict["loss_fn"] = None
    training_dict["optimizer"] = None
    training_dict["scheduler"] = None 
    training_dict["log_interval"] = None
    training_dict["pbar"] = None
    training_dict["pbar_update"] = None
    training_dict["train_losses"] = []
    training_dict["test_losses"] = []
    training_dict["train_accuracies"] = []
    training_dict["test_accuracies"] = []
    training_dict["config_num"] = -1


    training_dict["y_pred_train"] = []
    training_dict["y_train"] = []
    training_dict["y_pred_test"] = []
    training_dict["y_test"] = []

    # Instantiating our dataset object and create data loader.
    ds = NVVIDataset(config["ANNOTATIONS_FILE"],
                     config["AUDIO_DIR"],
                     config["SAMPLE_RATE"],
                     DEVICE)


    # Create training and testing split of the data.
    train_count = int(0.8 * len(ds))
    test_count = len(ds) - train_count

    train_dataset, test_dataset = torch.utils.data.random_split(
        ds, (train_count, test_count)
    )


    waveform, label = train_dataset[0]


    print("Shape of waveform: {}".format(waveform.size()))

    labels = sorted(list(set(datapoint[1] for datapoint in train_dataset)))
    print("lables: {}".format(labels))

    dataset_dict["train_loader"] = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    dataset_dict["test_loader"] = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    reports_path = Path("reports") / f"{config['exp_name']}{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}"
    reports_path.mkdir(exist_ok=True, parents=True)


    if config["train_mode"] == "tn":
        init_and_train_model()
        plot_confusion_matrix(metric="confusion_matrix", save_path=reports_path)
    elif config["train_mode"] == "thp":
        with open("search_space.json") as json_file:
            search_space_config = json.load(json_file)

        with open(reports_path / "search_space.json", "w") as json_file:
            json_file.write(json.dumps(search_space_config, indent=2))

        search_space = {}
        for key in search_space_config:
            search_space[key] = hp.choice(key, search_space_config[key])

        trials = Trials()

        # Perform optimal hyper-parameter search. 
        best = fmin(fn=init_and_train_model,
            space=search_space,
            algo=tpe.suggest,
            trials=trials,
            max_evals=12)
    
    elif config["train_mode"] == "tkf":
        perf_kfold_validation()
    elif config["train_mode"] == "tp":
        dataset_dict["train_loader"] = torch.utils.data.DataLoader(
        ds,
        batch_size=config["BATCH_SIZE"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory)

        init_and_train_model()
    elif config["train_mode"] == "tea":
        eval_augs()
    
