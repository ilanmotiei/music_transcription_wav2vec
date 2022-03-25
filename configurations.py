import torch


original_musicnet_sampling_rate = 44100
sampling_rate = 16000
bins = 49 * 5  # amount of predictions in a single unit of audio (#samples = sampling_rate * unit_duration)
pitch_classes = 128
unit_duration = 1  # duration in seconds of each unit for which we divide the audio and the labels files
num_samples_in_unit = int(unit_duration * sampling_rate)  # amount of audio samples in a unit

wav2vec_model = 'facebook/wav2vec2-base'
wav2vec_model_embedding_dim = 768

musicnet_data_path = '../../music-translation/musicnet'
device = torch.device('cuda')
epochs = 50000
logs_file = open('logs.txt', 'w')
update_every_n_batches = 1  # for gradient accumulation - amount of steps between each .step() call
train_print_every = 10  # amount of batches between each print
checkpoint_every = 8000  # amount of batches processed between each checkpoint. # In each checkpoint we're doing validation and saving the model
models_dir = 'models'
batch_size = 100
lr = 3e-04
weight_decay = 0.01
num_workers = 20
clip_grad = True
grad_clip_value = 1
freeze_wav2vec_feature_extractor = False
freeze_entire_wav2vec = False

current_epoch_num = 6
model_checkpoint = 'models/model_epoch=5.pth'
# place here ^ a model's checkpoint path if you want to continue to train from there
optimizer_checkpoint = 'models/optimizer_epoch=5.pth'
# place here ^ an optimizer's checkpoint path if you want to continue to train from it

pitch_prediction_thresholds = [.05, .1, .2, .3, .4, .5, .6, .7, .8, .85, .9, .95]
