
import configurations as cnf
from torch import nn
import torch
import tqdm
from sklearn.metrics import average_precision_score, recall_score, precision_score, f1_score
from model import Transcriptor
from dataset import MusicNetDataset
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from os import path
import os
import sys
import numpy as np
from torch.nn.utils import clip_grad_value_
from metrics import precision, recall
from torch.optim.radam import RAdam


def train_epoch(model, epoch, optimizer, train_dataloader, validation_dataloader):
    print(f" --------------- Epoch {epoch}/{cnf.epochs} Training Start --------------- ", file=cnf.logs_file)

    loss_criterion = nn.BCEWithLogitsLoss()

    model.train()
    epoch_loss = 0
    optimizer.zero_grad()

    with torch.set_grad_enabled(True):
        for batch_idx, (batch_audio, batch_target_labels) in tqdm.tqdm(enumerate(train_dataloader)):
            batch_audio = batch_audio.to(cnf.device)  # shape = (batch_size, cnf.unit_duration * cnf.sampling_rate)
            batch_target_labels = batch_target_labels.to(cnf.device)  # shape = (batch_size, cnf.bins, cnf.pitch_classes)

            model_logits = model(batch_audio)  # shape = (batch_size, cnf.bins, cnf.pitch_classes)
            loss = loss_criterion(model_logits, batch_target_labels.float())

            loss = loss / cnf.update_every_n_batches
            loss.backward()

            if ((batch_idx + 1) % cnf.update_every_n_batches == 0) or (batch_idx == len(train_dataloader) - 1):
                if cnf.clip_grad:
                    clip_grad_value_(model.parameters(), clip_value=cnf.grad_clip_value)

                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

            epoch_loss += loss.item() * cnf.update_every_n_batches

            if (batch_idx + 1) % cnf.train_print_every == 0:
                print(f" ------------ Epoch {epoch}/{cnf.epochs} Batch {batch_idx + 1}/{len(train_dataloader)} Training Results ------------ ",
                      file=cnf.logs_file)
                print(f"Total Loss: {epoch_loss / (batch_idx + 1)}", file=cnf.logs_file)

            if (batch_idx + 1) % cnf.checkpoint_every == 0:
                validate(model=model,
                         epoch=epoch,
                         validation_dataloader=validation_dataloader)

                torch.save(model.state_dict(), f'{cnf.models_dir}/model_epoch={epoch}.pth')
                model.train()

    epoch_loss /= len(train_dataloader)

    print(f" --------------------- Epoch {epoch}/{cnf.epochs} Final Training Results ------------------------ ", file=cnf.logs_file)
    print(f"Training Loss: {epoch_loss}", file=cnf.logs_file)


def validate(model, epoch, validation_dataloader):
    print(f" --------------- Epoch {epoch + 1}/{cnf.epochs} Validation Start --------------- ", file=cnf.logs_file)
    model.eval()

    loss_criterion = nn.BCEWithLogitsLoss()

    loss = 0
    aps_accuracy = 0
    recall_accuracy = np.zeros(shape=(len(cnf.pitch_prediction_thresholds), ))
    precision_accuracy = np.zeros(shape=(len(cnf.pitch_prediction_thresholds), ))

    with torch.set_grad_enabled(False):
        for batch_idx, (batch_audios, batch_target_labels) in tqdm.tqdm(enumerate(validation_dataloader)):
            batch_audios = batch_audios.to(cnf.device)  # shape = (batch_size, cnf.unit_duration * cnf.sampling_rate)
            batch_target_labels = batch_target_labels.to(cnf.device)  # shape = (batch_size, cnf.bins, cnf.pitch_classes)

            batch_logits = model(batch_audios)  # shape = (batch_size, cnf.bins, cnf.pitch_classes)
            batch_loss = loss_criterion(batch_logits, batch_target_labels.float())

            batch_probs = model.get_probs_from_logits(batch_logits)
            # ^ : shape = (batch_size, cnf.bins, cnf.pitch_classes)

            # ----- Calculating Recall and Precision for every threshold -----
            batch_recall = np.empty(shape=(len(cnf.pitch_prediction_thresholds), ))
            batch_precision = np.empty(shape=(len(cnf.pitch_prediction_thresholds), ))

            for i, threshold in enumerate(cnf.pitch_prediction_thresholds):
                batch_predictions = batch_probs > threshold
                batch_recall[i] = recall(target=batch_target_labels, prediction=batch_predictions)
                batch_precision[i] = precision(target=batch_target_labels, prediction=batch_predictions)

            # ----- Calculating APS -----
            numpyed_batch_targets = batch_target_labels.cpu().detach().numpy().astype(int)
            numpyed_batch_probs = batch_probs.cpu().detach().numpy().astype(float)

            batch_aps_accuracy = 0
            for target, probs in zip(numpyed_batch_targets, numpyed_batch_probs):
                batch_aps_accuracy += average_precision_score(y_true=target, y_score=probs)

            batch_aps_accuracy /= target.shape[0]  # averaging over the elements in the batch

            # ----- Aggregating -----
            loss += batch_loss.item()
            aps_accuracy += batch_aps_accuracy
            recall_accuracy += batch_recall
            precision_accuracy += batch_precision

    loss /= len(validation_dataloader)
    aps_accuracy /= len(validation_dataloader)

    recall_accuracy /= len(validation_dataloader)
    precision_accuracy /= len(validation_dataloader)
    f1_accuracy = recall_accuracy * precision_accuracy / (recall_accuracy + precision_accuracy)

    print("Results: --------------- ", file=cnf.logs_file)

    print(f"Test Loss: {loss}", file=cnf.logs_file)
    print(f"Test APS Accuracy: {aps_accuracy}", file=cnf.logs_file)
    print(f"Test F1: {f1_accuracy}", file=cnf.logs_file)
    print(f"Test Recall: {recall_accuracy}", file=cnf.logs_file)
    print(f"Test Precision: {precision_accuracy}", file=cnf.logs_file)


if __name__ == "__main__":
    model = Transcriptor().to(cnf.device)

    if not (cnf.model_checkpoint is None):
        model.load_state_dict(torch.load(cnf.model_checkpoint, map_location=cnf.device))

    train_dataset = MusicNetDataset(train=True)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=cnf.batch_size,
                                  shuffle=True,
                                  num_workers=cnf.num_workers)

    optimizer = AdamW(params=model.parameters(),
                      lr=cnf.lr,
                      weight_decay=cnf.weight_decay)

    if not (cnf.optimizer_checkpoint is None):
        optimizer.load_state_dict(torch.load(cnf.optimizer_checkpoint, map_location=cnf.device))

    validation_dataset = MusicNetDataset(train=False)
    validation_dataloader = DataLoader(dataset=validation_dataset,
                                       batch_size=1,  # when validating we should have batch_size set to 1 because of some bug in wav2vec
                                       shuffle=False,
                                       num_workers=cnf.num_workers)

    for epoch in range(cnf.current_epoch_num, cnf.current_epoch_num + cnf.epochs):
        train_epoch(model=model,
                    epoch=epoch,
                    optimizer=optimizer,
                    train_dataloader=train_dataloader,
                    validation_dataloader=validation_dataloader)

        if not path.exists(cnf.models_dir):
            os.mkdir(cnf.models_dir)

        torch.save(model.state_dict(), f'{cnf.models_dir}/model_epoch={epoch + 1}.pth')
        torch.save(optimizer.state_dict(), f'{cnf.models_dir}/optimizer_epoch={epoch + 1}.pth')

        validate(model=model,
                 epoch=epoch,
                 validation_dataloader=validation_dataloader)
