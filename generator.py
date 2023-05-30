import postpro
import os
from pydub import AudioSegment


# Method that generate a name for the songs to save, based on the number of midis
def song_namer(n_notes, note_offset, directory=os.path.join("generated_songs", "midis")):
    existing_files = [1 for _ in os.listdir(directory)]
    new_file_name = f"song{len(existing_files)+1}_{n_notes}_{note_offset}"
    return new_file_name


# Method that overlaps two given audios
def overlap_audios(audio1, audio2):
    if len(audio1) > len(audio2):
        audio1 = audio1[:len(audio2)]
    elif len(audio1) < len(audio2):
        audio2 = audio2[:len(audio1)]
    combined_audio = audio2.overlay(audio1)
    return combined_audio

# Method that generates a song with a duration in seconds and n_notes notes
def generate(duration=100, n_notes=120):
    note_offset = 1.5
    song_name = song_namer(n_notes, note_offset)
    generated_midi_fp = os.path.join("generated_songs", "midis", (song_name + ".midi"))
    generated_song_fp = os.path.join("generated_songs", (song_name + ".mp3"))
    print(f"Generating lofi song with name {generated_midi_fp} , {n_notes} notes and {note_offset} note offset ")

    postpro.sintetize(generated_midi_fp, os.path.join("temp", "temp_notes.wav"), n_notes, note_offset, duration=duration)

    audio_notes = AudioSegment.from_file(os.path.join("temp", "temp_notes.wav"), format="wav")
    audio_rain = AudioSegment.from_file(os.path.join("data", "extra", "Rain_Audio.wav"), format="wav") - 17
    audio_beat = AudioSegment.from_file(os.path.join("data", "extra", "CymaticsEternity Drum Loop 10 - 88 BPM.wav"),
                                        format="wav") - 3
    for _ in range(4):
        audio_rain = audio_rain + audio_rain
        audio_beat = audio_beat + audio_beat

    notes_and_rain = overlap_audios(audio_notes, audio_rain)
    total_combination = overlap_audios(audio_beat, notes_and_rain)

    print("Melody generated, starting postproduction")

    total_combination.export(generated_song_fp, format="mp3")
    print(f"Song saved: {generated_song_fp}")
    return

if __name__ == '__main__':
    generate(duration=10, n_notes=15)
