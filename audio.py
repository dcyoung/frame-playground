# Beat tracking example
from __future__ import print_function
import os
import os.path as osp
import librosa
import numpy as np
from moviepy.editor import *
from dummy_frames import generate_video_with_target_times_seconds

if __name__ == '__main__':
    # 1. Get the file path to the included audio example
    # filename = "C:\\Users\\dcyoung\\Music\\Montage Music\\Pink Floyd - Learning To Fly.mp3"
    filename = "C:\\Users\\dcyoung\\Music\\Montage Music\\Bar-Kays - Too Hot To Stop.mp3"
    # filename = librosa.util.example_audio_file()

    # 2. Load the audio as a waveform `y`
    #    Store the sampling rate as `sr`
    y, sr = librosa.load(filename)

    # 3. Run the default beat tracker
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

    # Let's make and display a mel-scaled power (energy-squared) spectrogram
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)

    y_harmonic, y_percussive = librosa.effects.hpss(y)
    S_harmonic = librosa.feature.melspectrogram(y_harmonic, sr=sr)
    S_percussive = librosa.feature.melspectrogram(y_percussive, sr=sr)

    # Convert to log scale (dB). We'll use the peak power as reference.
    log_Sh = librosa.power_to_db(S_harmonic, ref=np.max)
    log_Sp = librosa.power_to_db(S_percussive, ref=np.max)

    C = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)

    # Let's pad on the first and second deltas while we're at it
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    M = np.vstack([mfcc, delta_mfcc, delta2_mfcc])

    # tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr)
    tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)

    audio_file = osp.join("output", "audio.wav")
    librosa.output.write_wav(audio_file, y, sr)

    # generate the video
    duration_sec = librosa.get_duration(y, sr)
    generate_video_with_target_times_seconds(beat_times, duration_sec)

    video_file = osp.join("output", "test.avi")

    audioclip = AudioFileClip(audio_file)
    videoclip = VideoFileClip(video_file)
    videoclip_with_audio = videoclip.set_audio(audioclip)

    output_video_file = osp.join("output", "test2.mp4")
    videoclip_with_audio.write_videofile(output_video_file,  fps=24)
