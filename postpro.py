""" This module generates notes for a midi file using the
    trained neural network """
import os
import lstm
from music21 import instrument, note, chord, stream, pitch
import dawdreamer as daw
from scipy.io import wavfile
SAMPLE_RATE = 44100


# Given a list of notes (lists of ints) generates a MIDI file.
def create_midi(prediction_output, output_fp, offset_notes):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        pattern = [x for x in pattern if x != 0]

        if len(pattern) == 1:  # is a single note

            new_pitch = pitch.Pitch(midi=pattern[0])
            new_note = note.Note(pitch=new_pitch)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        elif len(pattern) == 0:  # is a rest
            rest_note = note.Rest
            rest_note.offset = offset
            output_notes.append(rest_note)
        else:  # is a chord
            notes = []
            for current_note in pattern:
                new_pitch = pitch.Pitch(midi=current_note)
                new_note = note.Note(pitch=new_pitch)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # increase offset each iteration so that notes do not stack
        offset += offset_notes

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp=output_fp)


# Method that generates everything and sintitizes the MIDI to a .WAV file.
def sintetize(midi_fp, wav_fp, n_notes, offset, duration=100):
    BUFFER_SIZE = 128  # Parameters will undergo automation at this buffer/block size.
    SYNTH_PLUGIN = "C:\\Program Files\\Common Files\\VST3\\LABS (64 Bit).vst3"  # extensions: .dll, .vst3, .vst, .component
    VINYL_PLUGIN = "C:\\Program Files\\Common Files\\VST3\\iZotope\\Vinyl.vst3"  # extensions: .dll, .vst3, .vst, .component
    BPM = 88

    WEIGHTS = os.path.join('weights', 'run_10_weights-1118-1.3864.hdf5')

    engine = daw.RenderEngine(SAMPLE_RATE, BUFFER_SIZE)
    engine.set_bpm(BPM)

    #Generating the midi
    print("tiempo generar notas: ", end=' ')
    generated_notes = lstm.generate_notes(n_notes, WEIGHTS)
    create_midi(generated_notes, midi_fp, offset)

    # Make a processor and give it the unique name "my_synth", which we use later.
    synth = engine.make_plugin_processor("my_synth", SYNTH_PLUGIN)
    assert synth.get_name() == "my_synth"

    vinyl = engine.make_plugin_processor("my_vinyl", VINYL_PLUGIN)
    assert vinyl.get_name() == "my_vinyl"

    if not os.path.isfile('piano_vst_config') or not os.path.isfile('vinyl_vst_config') :
        synth.open_editor()  # Open the editor, make changes, and close
        synth.save_state('piano_vst_config')
        vinyl.open_editor()  # Open the editor, make changes, and close
        vinyl.save_state('vinyl_vst_config')
    else:
        synth.load_state('piano_vst_config')
        vinyl.load_state('vinyl_vst_config')

    synth.load_midi(midi_fp, clear_previous=True, beats=True)

    graph = [  # plugging , input
        (synth, []),
        (vinyl, [synth.get_name()])
    ]

    engine.load_graph(graph)
    print("rendering", end='')
    engine.render(duration)
    print(" done")
    audio = engine.get_audio()  # shaped (2, N samples)
    wavfile.write(wav_fp, SAMPLE_RATE, audio.transpose())


if __name__ == '__main__':
    print("running postpro!")
