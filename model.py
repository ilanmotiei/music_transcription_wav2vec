
import torch
from torch import nn
from transformers import AutoModel, Wav2Vec2PreTrainedModel
import configurations as cnf


class Transcriptor(nn.Module):

    def __init__(self):
        super().__init__()

        self.wav2vec2 = AutoModel.from_pretrained(cnf.wav2vec_model)
        self.linear = nn.Linear(in_features=cnf.wav2vec_model_embedding_dim, out_features=cnf.pitch_classes)

    def forward(self, audio):
        """
        :param audio: A tensor of shape (batch_size, cnf.unit_duration * cnf.sampling_rate)
        :return: A tensor of shape (batch_size, cnf.bins, cnf.pitch_classes) containing the *- logits -* of the model.
        """

        embeddings = self.wav2vec2(audio).last_hidden_state
        # ^ : shape = (batch_size, cnf.bins, cnf.wav2vec_model_embedding_dim)

        logits = self.linear(embeddings)  # shape = (batch_size, cnf.bins, cnf.pitch_classes)

        return logits

    def predict_from_logits(self, logits, pred_threshold):
        """
        :param logits: A tensor of shape (batch_size, cnf.bins, cnf.pitch_classes) containing the logits of the model.
        :param pred_threshold: The minimum confidence score required for a pitch to be predicted.
        :return: A boolean tensor of shape (batch_size, cnf.bins, cnf.pitch_classes) that indicates which pitches
                 are present at which bins.
        """

        probs = nn.functional.sigmoid(logits)

        return probs > pred_threshold

