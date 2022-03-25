
import torch
from torch import nn
from transformers import Wav2Vec2Model, AutoModel
import configurations as cnf


class Transcriptor(nn.Module):

    def __init__(self):
        super().__init__()

        self.wav2vec2 = AutoModel.from_pretrained(cnf.wav2vec_model)

        if cnf.freeze_entire_wav2vec:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
        else:
            if cnf.freeze_wav2vec_feature_extractor:
                self.wav2vec2.freeze_feature_encoder()

        self.linear = nn.Linear(in_features=cnf.wav2vec_model_embedding_dim, out_features=5 * cnf.pitch_classes)
        # self.linear = nn.Linear(in_features=cnf.wav2vec_model_embedding_dim, out_features=cnf.pitch_classes)

        torch.nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0.01)

    def forward(self, audio):
        """
        :param audio: A tensor of shape (batch_size, cnf.unit_duration * cnf.sampling_rate)
        :return: A tensor of shape (batch_size, cnf.bins, cnf.pitch_classes) containing the *- logits -* of the model.
        """

        var, mean = torch.var_mean(audio, dim=1, keepdim=True)  # shapes = (batch_size, 1)
        audio = (audio - mean) / torch.sqrt(var + 1e-08)

        embeddings = self.wav2vec2(audio).last_hidden_state
        # ^ : shape = (batch_size, cnf.bins / 5, cnf.wav2vec_model_embedding_dim)

        logits = self.linear(embeddings)  # shape = (batch_size, cnf.bins / 5, 5 * cnf.pitch_classes)

        logits = logits.view(-1, cnf.bins, cnf.pitch_classes)
        # shape = (batch_size, cnf.bins, cnf.pitch_classes)

        return logits

    def predict_from_logits(self, logits, pred_threshold):
        """
        :param logits: A tensor of shape (batch_size, cnf.bins, cnf.pitch_classes) containing the logits of the model.
        :param pred_threshold: The minimum confidence score required for a pitch to be predicted.
        :return: A boolean tensor of shape (batch_size, cnf.bins, cnf.pitch_classes) that indicates which pitches
                 are present at which bins.
        """
        probs = self.get_probs_from_logits(logits)
        return probs > pred_threshold

    def get_probs_from_logits(self, logits):
        return torch.sigmoid(logits)