from collections import defaultdict
from mido import MidiFile
from pydub import AudioSegment
from pydub.generators import Sine

tempo = 100  # bpm
def note_to_freq(note, concert_A=440.0):
    '''
    from wikipedia: http://en.wikipedia.org/wiki/MIDI_Tuning_Standard#Frequency_values
    '''
    return (2.0 ** ((note - 69) / 12.0)) * concert_A

def ticks_to_ms(mid, ticks):
    tick_ms = (60000.0 / tempo) / mid.ticks_per_beat
    return ticks * tick_ms

def convert_midi_to_wav(input_path, output_path):
    mid = MidiFile(input_path)
    output = AudioSegment.silent(mid.length * 1000.0)


    for track in mid.tracks:
        current_pos = 0.0

        current_notes = defaultdict(dict)
        i = 0
        for msg in track:
            i+= 1
            current_pos += ticks_to_ms(mid, msg.time)

            if msg.type == 'note_on':
                current_notes[msg.channel][msg.note] = (current_pos, msg)

            if msg.type == 'note_off':
                if msg.note in current_notes[msg.channel]:
                    start_pos, start_msg = current_notes[msg.channel].pop(msg.note)

                    duration = current_pos - start_pos

                    signal_generator = Sine(note_to_freq(msg.note))
                    rendered = signal_generator.to_audio_segment(duration=duration - 50, volume=-20).fade_out(100).fade_in(30)

                    output = output.overlay(rendered, start_pos)

    output.export(output_path, format="wav")