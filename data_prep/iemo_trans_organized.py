import pandas as pd
import jiwer


transformation = jiwer.Compose([
    jiwer.ToUpperCase(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces(),
    jiwer.ReduceToSingleSentence(),
    jiwer.RemoveKaldiNonWords(),
    jiwer.RemoveSpecificWords(['[LAUGHTER]','[GARBAGE]','[BREATHING]','[LIPSMACK]']),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.RemoveEmptyStrings(),
]) 

ref_dic={}
iemocap_ref_file = '../data/iemo_trans_raw_all.csv'
df = pd.read_csv(iemocap_ref_file, delimiter='\t', header=None, names=['utt_name','sentence'])

sentences = df.sentence.values.astype(str)
utt_name = df.utt_name.values.tolist()

organized_output = iemocap_ref_file.replace('raw','organized-noPunc')
with open(organized_output,'w') as csv:
    for i in range(len(sentences)):
        tmp= transformation(sentences[i])
        if not tmp: tmp = '.'
        # if tmp == '.': print(utt_name[i])
        csv.write('{}\t{}'.format(utt_name[i],tmp))
        if i<len(sentences)-1:
            csv.write('\n')

csv.close()