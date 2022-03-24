
import transformers
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel, Wav2Vec2Config, AutoFeatureExtractor, AutoModel
import torch


if __name__ == "__main__":
    model = Wav2Vec2Model(Wav2Vec2Config())
    model = AutoModel.from_pretrained('facebook/wav2vec2-base')

    print(sum(p.numel() for p in model.parameters() if p.requires_grad is True))  # num of parameters of the model
    model.freeze_feature_extractor()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad is True))  # num of parameters of the model

    input = torch.randint(low=-100, high=100, size=(1, 16000))  # sample input (1 second of a 16KHz audio file)
    input = input.float()

    outputs = model(input_values=input)  # shape = (1, 49, 768) for wav2vec2-base
    print(outputs.last_hidden_state.shape)
