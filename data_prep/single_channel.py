import soundfile as sf
import os
import numpy as np

iemo_root='IEMOCAP'
wav_dir_all = [f'{iemo_root}/Session{i}/dialog/wav' for i in range(1,6)]
for i in range(1,6):
    SingleChannel_dir=f'{iemo_root}/Session{i}/dialog/wav-SingleChannel'
    if not os.path.exists(SingleChannel_dir):
        os.mkdir(SingleChannel_dir)

cnt=0
for wav_dir in wav_dir_all:
    for wav_path in os.listdir(wav_dir):
        if wav_path.endswith('.wav'):
            wav_path = os.path.join(wav_dir,wav_path)
            data, sr = sf.read(wav_path)
            # print(data.shape)
            assert data.shape[-1]==2
            data = np.mean(data,-1)
            wav_path_new = wav_path.replace('dialog/wav','dialog/wav-SingleChannel')
            print(wav_path_new)
            sf.write(wav_path_new, data, sr)
            cnt+=1
print(cnt)
