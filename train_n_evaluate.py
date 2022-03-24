
import configurations as cnf
from torch import nn
import torch
import tqdm
from sklearn.metrics import average_precision_score
from model import Transcriptor
from dataset import MusicNetDataset
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
import pathlib
import os


def train_epoch(model, epoch, optimizer, train_dataloader, validation_dataloader, loss_criterion):
    print(f" --------------- Epoch {epoch + 1}/{cnf.epochs} Training Start --------------- ", file=cnf.logs_file)

    epoch_loss = 0
    model.train()

    with torch.set_grad_enabled(True):
        for batch_idx, (batch_audio, batch_target_labels) in tqdm.tqdm(enumerate(train_dataloader)):
            batch_audio = batch_audio.to(cnf.device)  # shape = (batch_size, cnf.unit_duration * cnf.sampling_rate)
            batch_target_labels = batch_target_labels.to(cnf.device)  # shape = (batch_size, cnf.bins, cnf.pitch_classes)

            model_logits = model(batch_audio)  # shape = (batch_size, cnf.bins, cnf.pitch_classes)
            loss = loss_criterion(model_logits, batch_target_labels)

            loss = loss / cnf.update_every_n_batches
            loss.backward()

            if ((batch_idx + 1) % cnf.update_every_n_batches == 0) or (batch_idx == len(train_dataloader) - 1):
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

            epoch_loss += loss.item() * cnf.update_every_n_batches

            if (batch_idx + 1) % cnf.train_print_every == 0:
                print(f" ------------ Epoch {epoch + 1}/{cnf.epochs} Batch {batch_idx + 1}/{len(train_dataloader)} Training Results ------------ ", file=cnf.logs_file)
                print(f"Total Loss: {epoch_loss / (batch_idx + 1)}", file=cnf.logs_file)

            if (batch_idx + 1) % cnf.checkpoint_every == 0:
                validate(model=model,
                         epoch=epoch,
                         validation_dataloader=validation_dataloader,
                         loss_criterion=loss_criterion)

                torch.save(model.state_dict(), f'{cnf.models_dir}/model_epoch={epoch + 1}.pth')
                model.train()

    epoch_loss /= len(train_dataloader)

    print(f" --------------------- Epoch {epoch + 1}/{cnf.epochs} Final Training Results ------------------------ ", file=cnf.logs_file)
    print(f"Training Loss: {epoch_loss}", file=cnf.logs_file)


def validate(model, epoch, validation_dataloader, loss_criterion):
    print(f" --------------- Epoch {epoch + 1}/{cnf.epochs} Validation Start --------------- ", file=cnf.logs_file)

    model.eval()

    loss = 0
    aps_accuracy = 0

    with torch.set_grad_enabled(False):
        for batch_idx, (batch_audios, batch_target_labels) in enumerate(validation_dataloader):
            batch_audios = batch_audios.to(cnf.device)  # shape = (batch_size, cnf.unit_duration * cnf.sampling_rate)
            batch_target_labels = batch_target_labels.to(cnf.device)  # shape = (batch_size, cnf.bins, cnf.pitch_classes)

            batch_logits = model(batch_audios)  # shape = (batch_size, cnf.bins, cnf.pitch_classes)
            batch_loss = loss_criterion(batch_logits, batch_target_labels)

            batch_probs = nn.functional.sigmoid(batch_logits)  # shape = (batch_size, cnf.bins, cnf.pitch_classes)

            batch_aps_accuracy = average_precision_score(y_true=batch_target_labels.cpu().detach().numpy(),
                                                         y_score=batch_probs.cpu().detach().numpy())

            loss += batch_loss.item()
            aps_accuracy += batch_aps_accuracy

    print("Results: --------------- ", file=cnf.logs_file)

    print(f"Test Loss: {loss / (len(validation_dataloader))}")
    print(f"Test APS Accuracy: {aps_accuracy / len(validation_dataloader)}")


def train(cnf):
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
                                       batch_size=cnf.batch_size,
                                       shuffle=False,
                                       num_workers=cnf.num_workers)

    loss_criterion = nn.BCEWithLogitsLoss()

    for epoch_idx in range(cnf.current_epoch_num, cnf.current_epoch_num + cnf.epochs):
        train_epoch(model=model,
                    epoch=epoch_idx,
                    optimizer=optimizer,
                    train_dataloader=train_dataloader,
                    validation_dataloader=validation_dataloader,
                    loss_criterion=loss_criterion)

        if not path.exists(cnf.models_dir):
            os.mkdir(cnf.models_dir)

        torch.save(model.state_dict(), f'{cnf.models_dir}/model_epoch={epoch_idx + 1}.pth')
        torch.save(optimizer.state_dict(), f'{cnf.models_dir}/optimizer_epoch={epoch_idx + 1}.pth')

        validate(model=model,
                 epoch=epoch_idx,
                 validation_dataloader=validation_dataloader,
                 loss_criterion=loss_criterion)

