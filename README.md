# Time-weighted Emotion Error Rate (TEER)  
Code for "Integrating Emotion Recognition with Speech Recognition and Speaker Diarisation for Conversations". 
This paper proposes a system that integrates emotion recognition with speech recognition and speaker diarisation in a jointly-trained model.  

Two metrics proposed to evaluate emotion classification performance with automatic segmentation:  
  - Time-weighted Emotion Error Rate (TEER)  
    $$\text{TEER} = \frac{\text{MS}+\text{FA}+\text{CONF}_\text{emo}}{\text{TOTAL}}$$
  - speaker-attributed Time-weighted Emotion Error Rate (sTEER) 
    $$\text{sTEER} = \frac{\text{MS}+\text{FA}+\text{CONF}_\text{emo+spk}}{\text{TOTAL}}$$
    
## Setup
- Python == 3.7
- PyTorch == 1.11
- Speechbrain == 0.5.14
- pyannote.core == 4.5
- pyannote.metrics == 3.2.1 

## Data preparation
1. Convert stereo audio to single channel  
  `data_prep/single_channel.py`  

2. Prepare reference transcriptions   
    `data_prep/iemo_trans_raw.py   # generate raw reference transcription from the dataset`  
    `data_prep/iemo_trans_organized.py # remove punctuation and special markers`  

3. Prepare emotion label  
    `data_prep/iemo_lab_AER-cat.py # 6-way emotion classification label`  

4. Prepare VAD label  
    - Label used for training: intra-utterance frame-level speech/non-speech    
    `data_prep/iemo_lab_VAD-utt.py`  
    - Label used for testing: speech segments according to word-level alignment (silence at the beginning, between words and at the end are removed)  
    `data_prep/iemo_lab_VAD-seg.py`
    - Convert speech segments to pyannote Annotation format  
    `data_prep/iemo_lab_VAD-annote.py`

5. Prepare training, validation, testing scp file  
    `data_prep/iemocap_prepare.py`

## Training
`Train.py Train.yaml --output_folder=exp`

## Testing and scoring
1. Forward windowed test dialogue  
   `Train.py Train.yaml --FWD_VAD=True --output_folder=exp`    
2. Evaluate VAD performance     
 `scoring/score_VAD.py`   
3. Diariase based on predicted VAD  
 `fwd_drz.py fwd_drz.yaml --output_folder=exp-eval `  
4. Compute DER   
  `scoring/score_DER.py`   
5. Obtain segments for ASR and AER  
  `Train.py Train.yaml --FWD_DRZ=True --output_folder=exp`  
6. Compute cpWER  
   `scoring/score_cpWER.py`   
7. Compute TEER and sTEER  
 `prepare_emo_rttm.py # prepare rttm file for (s)TEER evaluation`  
 `scoring/score_TEER.py   # compute TEER and sTEER`  

*N.B. Since the CTC loss function of PyTorch (torch.nn.functional.ctc_loss) may produce nondeterministic gradients when given tensors on a CUDA device, users may get slighty different results from those reported in the paper.   
See https://pytorch.org/docs/1.11/generated/torch.nn.functional.ctc_loss.html for details.*

##
Please cite:  
> @inproceedings{wu23_interspeech,  
  author={Wen Wu and Chao Zhang and Philip C. Woodland},  
  title={{Integrating Emotion Recognition with Speech Recognition and Speaker Diarisation for Conversations}},  
  year=2023,  
  booktitle={Proc. INTERSPEECH 2023},  
  pages={3607--3611},  
  doi={10.21437/Interspeech.2023-293}  
}   
