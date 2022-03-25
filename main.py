
import transformers
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel, Wav2Vec2Config, AutoFeatureExtractor, AutoModel
import torch


if __name__ == "__main__":
    # model = Wav2Vec2Model(Wav2Vec2Config())
    # model = AutoModel.from_pretrained('facebook/wav2vec2-base')
    #
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad is True))  # num of parameters of the model
    # model.freeze_feature_extractor()
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad is True))  # num of parameters of the model
    #
    input = torch.randint(low=-100, high=100, size=(1, 16000))  # sample input (1 second of a 16KHz audio file)
    input = input.float()
    #
    # outputs = model(input_values=input)  # outputs.last_hidden_state.shape = (1, 49, 768) for wav2vec2-base
    # print(outputs.last_hidden_state.shape)

    from transformers import Wav2Vec2FeatureExtractor

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                 do_normalize=True, return_attention_mask=False)

    print(type(feature_extractor))

    print(feature_extractor(input).input_values)
    var, mean = torch.var_mean(input, dim=1, keepdim=True)  # shapes = (batch_size, 1)
    input = (input - mean) / (torch.sqrt(var))

    print(input)