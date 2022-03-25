
import pretty_midi
import torch
from preprocess import load_audio
import torchaudio
import configurations as cnf
from model import Transcriptor
import tqdm
import pandas as pd


INSTRUMENTS = [
    'Acoustic Grand Piano',
    'Bright Acoustic Piano',
    'Electric Grand Piano',
    'Honky-tonk Piano',
    'Electric Piano 1',
    'Electric Piano 2',
    'Harpsichord',
    'Clavi',
    'Celesta',
    'Glockenspiel',
    'Music Box',
    'Vibraphone',
    'Marimba',
    'Xylophone',
    'Tubular Bells',
    'Dulcimer',
    'Drawbar Organ',
    'Percussive Organ',
    'Rock Organ',
    'Church Organ',
    'Reed Organ',
    'Accordion',
    'Harmonica',
    'Tango Accordion',
    'Acoustic Guitar (nylon)',
    'Acoustic Guitar (steel)',
    'Electric Guitar (jazz)',
    'Electric Guitar (clean)',
    'Electric Guitar (muted)',
    'Overdriven Guitar',
    'Distortion Guitar',
    'Guitar harmonics',
    'Acoustic Bass',
    'Electric Bass (finger)',
    'Electric Bass (pick)',
    'Fretless Bass',
    'Slap Bass 1',
    'Slap Bass 2',
    'Synth Bass 1',
    'Synth Bass 2',
    'Violin',
    'Viola',
    'Cello',
    'Contrabass',
    'Tremolo Strings',
    'Pizzicato Strings',
    'Orchestral Harp',
    'Timpani',
    'String Ensemble 1',
    'String Ensemble 2',
    'SynthStrings 1',
    'SynthStrings 2',
    'Choir Aahs',
    'Voice Oohs',
    'Synth Voice',
    'Orchestra Hit',
    'Trumpet',
    'Trombone',
    'Tuba',
    'Muted Trumpet',
    'French Horn',
    'Brass Section',
    'SynthBrass 1',
    'SynthBrass 2',
    'Soprano Sax',
    'Alto Sax',
    'Tenor Sax',
    'Baritone Sax',
    'Oboe',
    'English Horn',
    'Bassoon',
    'Clarinet',
    'Piccolo',
    'Flute',
    'Recorder',
    'Pan Flute',
    'Blown Bottle',
    'Shakuhachi',
    'Whistle',
    'Ocarina',
    'Lead 1 (square)',
    'Lead 2 (sawtooth)',
    'Lead 3 (calliope)',
    'Lead 4 (chiff)',
    'Lead 5 (charang)',
    'Lead 6 (voice)',
    'Lead 7 (fifths)',
    'Lead 8 (bass + lead)',
    'Pad 1 (new age)',
    'Pad 2 (warm)',
    'Pad 3 (polysynth)',
    'Pad 4 (choir)',
    'Pad 5 (bowed)',
    'Pad 6 (metallic)',
    'Pad 7 (halo)',
    'Pad 8 (sweep)',
    'FX 1 (rain)',
    'FX 2 (soundtrack)',
    'FX 3 (crystal)',
    'FX 4 (atmosphere)',
    'FX 5 (brightness)',
    'FX 6 (goblins)',
    'FX 7 (echoes)',
    'FX 8 (sci-fi)',
    'Sitar',
    'Banjo',
    'Shamisen',
    'Koto',
    'Kalimba',
    'Bag pipe',
    'Fiddle',
    'Shanai',
    'Tinkle Bell',
    'Agogo',
    'Steel Drums',
    'Woodblock',
    'Taiko Drum',
    'Melodic Tom',
    'Synth Drum',
    'Reverse Cymbal',
    'Guitar Fret Noise',
    'Breath Noise',
    'Seashore',
    'Bird Tweet',
    'Telephone Ring',
    'Helicopter',
    'Applause',
    'Gunshot'
]


CORRESPONDING_WAV_SAMPLE_RATE = 44100


def instrument_number_to_instrument_name(instrument_number: str) -> int:
    return INSTRUMENTS[instrument_number - 1]


def df2midi(midi_df, output_file_name) -> None:

    """
    :param midi_df: A dataframe of the notes, FROM THE 'Musicnet' DATASET !!
    :param output_file_name: The file to which we should output the data.
    :return: None, saves the extracted midi file to the 'output_file_name'.
    """

    midi_file = pretty_midi.PrettyMIDI()
    instruments_ids = midi_df['instrument'].drop_duplicates()

    instruments = {instrument_id:
                   pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program(instrument_number_to_instrument_name(instrument_number=instrument_id)))
                   for instrument_id in instruments_ids}

    for _, row in midi_df.iterrows():
        note_start_time = row['start_time'] / CORRESPONDING_WAV_SAMPLE_RATE
        note_end_time = row['end_time'] / CORRESPONDING_WAV_SAMPLE_RATE

        note_instrument_id = row['instrument']
        note_number = row['note']

        note = pretty_midi.Note(velocity=100, pitch=note_number, start=note_start_time, end=note_end_time)
        instruments[note_instrument_id].notes.append(note)

    midi_file.instruments += instruments.values()
    midi_file.write(output_file_name)


def transcribe(filepath: str, model: Transcriptor, instrument_name=None) -> None:

    """
    """

    midi_file = pretty_midi.PrettyMIDI()

    if instrument_name is None:
        instrument_name = 'Acoustic Grand Piano'

    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program(instrument_name))

    audio = load_audio(filepath=filepath, torch_resampler=torchaudio.transforms.Resample(cnf.original_musicnet_sampling_rate,
                                                                                         cnf.sampling_rate,
                                                                                         dtype=torch.float32))

    # ^ : audio - a list of units of audio

    curr_absolute_time = 0
    with torch.set_grad_enabled(False):
        for audio_unit in tqdm.tqdm(audio):
            notes = model.predict_from_logits(model(audio_unit.unsqueeze(0)),
                                              pred_threshold=cnf.pitch_prediction_threshold)

            # ^ : notes.shape = (1, cnf.bins, cnf.pitch_classes)

            for time in range(cnf.bins):
                for pitch_class in range(cnf.pitch_classes):
                    if notes[0][time][pitch_class] > 0:
                        note = pretty_midi.Note(velocity=50,
                                                pitch=pitch_class + 1,
                                                start=curr_absolute_time,
                                                end=curr_absolute_time + cnf.unit_duration / cnf.bins)
                        instrument.notes.append(note)

                curr_absolute_time += cnf.unit_duration / cnf.bins

    midi_file.instruments.append(instrument)
    midi_file.write('.'.join(filepath.split('.')[:-1]) + '_transcribed.mid')


if __name__ == "__main__":
    #
    # model = Transcriptor()
    # model.load_state_dict(torch.load(cnf.model_checkpoint, map_location=cnf.device))
    #
    # transcribe('../../music-translation/musicnet/test_data/2556.wav', model, 'Acoustic Grand Piano')

    midi_df = pd.read_csv('../../music-translation/musicnet/test_labels/2556.csv')

    df2midi(midi_df=midi_df,
            output_file_name='2556_gt.mid')
