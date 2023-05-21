#!/usr/bin/python3
"""
Diarisation.
Code modified from https://github.com/speechbrain/speechbrain/blob/develop/recipes/AMI/Diarization/experiment.py
"""

import os
import sys
import torch
import logging
import json
import glob
import shutil
from types import SimpleNamespace
import numpy as np
import ruamel.yaml
import speechbrain as sb
from tqdm.contrib import tqdm
from hyperpyyaml import load_hyperpyyaml
from speechbrain.processing import diarization as diar
from speechbrain.utils.DER import DER
from speechbrain.dataio.dataio import read_audio
# from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from pyannote.core import Annotation

from utils import *

SAMPLERATE = 16000

# Logger setup
logger = logging.getLogger(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

def dataio_prep(json_file):
    dataset = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=json_file
    )
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig","start_s","stop_s")
    def audio_pipeline(wav):
        sig = read_audio(wav)
        if len(sig)<=5120:  #wavlm frame rate 320ms *16000 = 5120
            sig = torch.cat((sig,sig))
        start = wav['start']
        stop = wav['stop']
        start_s = round(start/16000,4)
        stop_s = round(stop/16000,4)
        return sig,start_s,stop_s

    sb.dataio.dataset.add_dynamic_item([dataset], audio_pipeline)
    sb.dataio.dataset.set_output_keys([dataset], ["id", "sig","start_s","stop_s"])
    dataloader = sb.dataio.dataloader.make_dataloader(
        dataset, **params["dataloader_opts"]
    )
    return dataloader

def compute_embeddings(wavs, lens):
    with torch.no_grad():
        wavs = wavs.to(params["device"])
        embeddings = embedding_model(wavs)
    return embeddings

def embedding_computation_loop_full(set_loader, stat_file):
    # Extract embeddings (skip if already done).
    if not os.path.isfile(stat_file):
        logger.debug("Extracting deep embeddings and diarizing")
        embeddings = np.empty(shape=[0, params["emb_dim"]], dtype=np.float64)
        
        starts =[]
        stops=[]
        modelset = []
        segset = []

        for batch in set_loader:
            wavs, lens = batch.sig
            starts += [x for x in batch.start_s]
            stops += [x for x in batch.stop_s]
            mod = [x for x in batch.id]
            seg = [x for x in batch.id]
            modelset = modelset + mod
            segset = segset + seg

            emb = compute_embeddings(wavs, lens).cpu().numpy()
            if np.isnan(emb).any():
                print(emb,batch.id)

            embeddings = np.concatenate((embeddings, emb), axis=0)

        starts = np.array(starts)
        stops = np.array(stops)
        # Intialize variables for start, stop and stat0.
        s = np.array([None] * embeddings.shape[0])
        b = np.array([[1.0]] * embeddings.shape[0])
        modelset = np.array(modelset, dtype="|O")
        segset = np.array(segset, dtype="|O")

        stat_obj = {
            'start':starts,
            'stop':stops,
            'stat0':b,
            'stat1':embeddings,
            'modelset':modelset,
            'segset':segset,
            }
        np.save(stat_file,stat_obj)

    else:
        logger.debug("Loading previously saved embeddings.")
        stat_obj=np.load(stat_file,allow_pickle=True).item()

    return stat_obj

def prepare_subset_json(full_meta_data, rec_id, out_meta_file):
    subset = {}
    for key in full_meta_data:
        k = str(key)
        if k.startswith(rec_id):
            subset[key] = full_meta_data[key]

    with open(out_meta_file, mode="w") as json_f:
        json.dump(subset, json_f, indent=2)

def cos_spec_clustering_full(diary_obj,out_rttm_file,rec_id,k_oracle,pval):
    assert k_oracle==2

    clust_obj = diar.Spec_Clust_unorm(min_num_spkrs=2, max_num_spkrs=2)
    try:
        sim_mat = clust_obj.get_sim_mat(diary_obj['stat1'])
    except:
        print(diary_obj['stat1'])
        exit()

    pruned_sim_mat = clust_obj.p_pruning(sim_mat, pval)
    # Symmetrization
    sym_pruned_sim_mat = 0.5 * (pruned_sim_mat + pruned_sim_mat.T)
    # Laplacian calculation: the un-normalized laplacian for the given affinity matrix.
    laplacian = clust_obj.get_laplacian(sym_pruned_sim_mat)
    # Get Spectral Embeddings
    emb, num_of_spk = clust_obj.get_spec_embs(laplacian, k_oracle)
    assert num_of_spk == 2

    # Perform clustering
    clust_obj.cluster_embs(emb, num_of_spk)
    labels = clust_obj.labels_

    # Convert labels to speaker boundaries
    lol = []
    for i in range(labels.shape[0]):
        spkr_id = rec_id + "_" + str(labels[i])
        a = [rec_id, diary_obj['start'][i], diary_obj['stop'][i], spkr_id]
        lol.append(a)

    lol.sort(key=lambda x: float(x[1]))
    lol = diar.merge_ssegs_same_speaker(lol)
    diar.write_rttm(lol, out_rttm_file)

def cos_do_spec_clustering_VAD(
    diary_obj, out_rttm_file, rec_id, k_oracle, pval
    ):

    assert k_oracle==2
    clust_obj = diar.Spec_Clust_unorm(min_num_spkrs=2, max_num_spkrs=10)
    clust_obj.do_spec_clust(diary_obj['stat1'], k_oracle, pval)
    labels = clust_obj.labels_

    # Convert labels to speaker boundaries
    lol = []

    for i in range(labels.shape[0]):
        spkr_id = rec_id + "_" + str(labels[i])
        a = [rec_id, diary_obj['start'][i], diary_obj['stop'][i], spkr_id]
        lol.append(a)

    # Sorting based on start time of sub-segment
    lol.sort(key=lambda x: float(x[1]))
    lol = diar.merge_ssegs_same_speaker(lol)
    lol = diar.distribute_overlap(lol)
    diar.write_rttm(lol, out_rttm_file)

def diarize_dataset_full(full_meta, split_type, pval):
    full_ref_rttm_file = fullref_name(split_type)
    rttm = diar.read_rttm(full_ref_rttm_file)
    spkr_info = list(filter(lambda x: x.startswith("SPKR-INFO"), rttm))

    # Get all the recording IDs in this dataset.
    all_keys = full_meta.keys()
    A = [word.rstrip().split("-")[0] for word in all_keys] 

    all_rec_ids = list(set(A))
    all_rec_ids.sort()
    split = "IEMO_" + split_type
    i = 1

    embedding_model.eval()
    msg = "Diarizing " + split_type + " set"
    logger.info(msg)

    if len(all_rec_ids) <= 0:
        msg = "No recording IDs found! Please check if meta_data json file is properly generated."
        logger.error(msg)
        sys.exit()

    # Diarizing different recordings in a dataset.
    for rec_id in tqdm(all_rec_ids):
        tag = f"[{split_type}: {i}/{len(all_rec_ids)}]"
        i += 1

        msg = "Diarizing %s : %s " % (tag, rec_id)
        logger.debug(msg)

        emb_file_name = rec_id + ".emb_stat.npy"
        diary_stat_emb_file = os.path.join(params["embedding_dir"], emb_file_name)

        json_file_name = rec_id + ".json"
        meta_per_rec_file = os.path.join(params["embedding_dir"], json_file_name)

        prepare_subset_json(full_meta, rec_id, meta_per_rec_file)
        diary_set_loader = dataio_prep(meta_per_rec_file)
        embedding_model.to(params["device"])
        diary_obj = embedding_computation_loop_full(diary_set_loader, diary_stat_emb_file)

        type_of_num_spkr = "oracle"
        tag = type_of_num_spkr+ "_cos_SC"
        out_rttm_dir = os.path.join(params["sys_rttm_dir"], split, tag)
        if not os.path.exists(out_rttm_dir):
            os.makedirs(out_rttm_dir)
        out_rttm_file = out_rttm_dir + "/" + rec_id + ".rttm"

        num_spkrs = diar.get_oracle_num_spkrs(rec_id, spkr_info)
        assert num_spkrs==2,[num_spkrs,rec_id,spkr_info,]
        # Go for Spectral Clustering (SC).
        cos_spec_clustering_full(diary_obj,out_rttm_file,rec_id,num_spkrs,pval)


    concate_rttm_file = out_rttm_dir + "/sys_output.rttm"
    logger.debug("Concatenating individual RTTM files...")
    with open(concate_rttm_file, "w") as cat_file:
        for f in glob.glob(out_rttm_dir + "/*.rttm"):
            if f == concate_rttm_file:
                continue
            with open(f, "r") as indi_rttm_file:
                shutil.copyfileobj(indi_rttm_file, cat_file)

    msg = "The system generated RTTM file for %s set : %s" % (
        split_type,
        concate_rttm_file,
    )
    logger.debug(msg)

    return concate_rttm_file

def fullref_name(split_type):
    return params["ref_rttm_dir"] + "/fullref_iemo-" + split_type + "-VAD.rttm"


class wavlm_xvector(torch.nn.Module):
    def __init__(self,params=None):
        super().__init__()
        if params is not None:
            self.hparams = SimpleNamespace(**params)
        self.device = params['device']
        self.interface = self.hparams.interface.to(self.device).eval()
        self.wav2vec2 = self.hparams.wav2vec2.to(self.device).eval()
        self.embedding_model=self.hparams.embedding_model.to(self.device).eval()

    def load_pretrain(self,checkpointer,AVERAGE_CKPT=True):
        if AVERAGE_CKPT:
            logger.info("Averaging checkpoints")

            ckpt_list = checkpointer.find_checkpoints(min_key="loss",max_num_checkpoints = 5)
            for k,v in checkpointer.recoverables.items():
                averaged_state = sb.utils.checkpoints.average_checkpoints(ckpt_list, k,device=self.device)
                if k == 'interface_SV':
                    self.interface.load_state_dict(averaged_state)
                elif k == 'embedding_model_SV':
                    self.embedding_model.load_state_dict(averaged_state)
                elif k== 'wav2vec2':
                    self.wav2vec2.load_state_dict(averaged_state)
        else:
            ckpt= checkpointer.find_checkpoint(min_key="loss")
            state_dict = torch.load(str(ckpt.path)+'/interface_SV.ckpt',map_location=self.device)
            self.interface.load_state_dict(state_dict)
            self.embedding_model.load_state_dict(torch.load(str(ckpt.path)+'/embedding_model_SV.ckpt',map_location=self.device))

    def forward(self, wavs):
        with torch.no_grad():
            feats = self.wav2vec2(wavs)
            if len(feats.shape)>3: 
                feats=feats.permute(1,2,0,3)    ##[B, T, 12, 768]
            else:
                feats=feats.transpose(0,1)
            feats = self.interface(feats)
            embeddings = self.embedding_model(feats)
            if torch.isnan(embeddings).any():
                print("embeddings nan:",embeddings)
        return embeddings
        
def eval_meta_json(data_folder,
                    save_folder,
                    test_id,
                    max_subseg_dur,
                    overlap,
                    type_name,
                    ):
    data_folder = data_folder

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    test_list=[diag_id for diag_id in ses_wav_dic.keys() if int(diag_id[4])==test_id]

    meta_rttm_dir=os.path.join(save_folder,'metadata')
    if not os.path.exists(meta_rttm_dir):
            os.makedirs(meta_rttm_dir)

    test_meta_file_name=f"{meta_rttm_dir}/iemo-test-{test_id}-{type_name}_{max_subseg_dur}-{overlap}.json"
    gen_meta(test_list,test_meta_file_name,max_subseg_dur,overlap)
    return test_meta_file_name

def gen_meta(splits_list,meta_path,max_subseg_dur,overlap):
    if os.path.exists(meta_path):
        print("%s already exists, not recreating." % meta_path)
        return

    shift = max_subseg_dur - overlap
    # Create JSON from subsegments
    json_dict = {}
    for diag_id in splits_list:
        CPD_annotation = Annotation()
        CPD_annotation=CPD_annotation.from_json(VAD_dic[diag_id])
        wav_file_path = ses_wav_dic[diag_id]["fea_path"]
        diag_id +='_'    #adding '_' to distinguisth Ses05M_script01_1b from Ses05M_script01_1
        for segment, track, label in CPD_annotation.itertracks(yield_label=True):
            start,stop =segment
            seg_dur = stop-start

            if seg_dur > max_subseg_dur:
                num_subsegs = int(seg_dur / shift)
                # Now divide this segment (new_row) in smaller subsegments
                for i in range(num_subsegs):
                    subseg_start = start + i * shift
                    # make sure each segment is dur length to avoid short segments
                    if subseg_start + max_subseg_dur - 0.01 > stop:
                        subseg_end = stop
                        subseg_start = stop - max_subseg_dur + 0.01
                    else:
                        subseg_end = subseg_start + max_subseg_dur - 0.01

                    json_dict[f"{diag_id}-{round(subseg_start,4)}_{round(subseg_end,4)}"] = {
                        "wav": {
                            "file": wav_file_path,
                            "start": int(subseg_start * SAMPLERATE),
                            "stop": int(subseg_end * SAMPLERATE),
                        },
                    }
                    # Break if exceeding the boundary
                    if subseg_end >= stop:
                        break
            else:
                json_dict[f"{diag_id}-{round(start,4)}_{round(stop,4)}"] = {
                    "wav": {
                        "file": wav_file_path,
                        "start": int(start * SAMPLERATE),
                        "stop": int(stop * SAMPLERATE),
                    },
                }
    with open(meta_path, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    msg = "%s JSON prepared" % (meta_path)
    print(msg)

if __name__ == "__main__":  
    params_file, run_opts, overrides = sb.core.parse_arguments(sys.argv[1:])

    if '--device' not in sys.argv[1:]:
        run_opts['device']= 'cuda' if torch.cuda.is_available() else 'cpu'

    ruamel_yaml = ruamel.yaml.YAML()
    overrides = ruamel_yaml.load(overrides)
    if not overrides:
        overrides={'device': run_opts['device']}
    else:
        overrides.update({'device': run_opts['device']})

    with open(params_file) as fin:
        params = load_hyperpyyaml(fin, overrides)

    set_seed(params['seed'])
    test_id = params['test_id']
    VAD_path = f'exp/VAD_test_outcome-{test_id}-E-1-merge0.25.json'
    VAD_dic=read_json(VAD_path)
    ses_wav_dic=read_json('data/iemo_ses.json')
    ref_dic=read_json('data/iemo_VAD_lab.json')

    # Create experiment directory.
    sb.core.create_experiment_directory(
        experiment_directory=params["output_folder"],
        hyperparams_to_save=params_file,
        overrides=overrides,
    )
    exp_dirs = [
        params["embedding_dir"],
        params["sys_rttm_dir"],
        params["der_dir"],
    ]
    for dir_ in exp_dirs:
        if not os.path.exists(dir_):
            os.makedirs(dir_)
    
    # load model
    checkpointer=params["checkpointer"]
    embedding_model=wavlm_xvector(params)
    embedding_model.load_pretrain(checkpointer,AVERAGE_CKPT=params['AVERAGE_CKPT'])

    eval_meta_file=eval_meta_json(data_folder=params['iemocap_root'],
                                save_folder=params['scp_folder'],
                                test_id = test_id,
                                max_subseg_dur=1,
                                overlap=0.5,
                                type_name = 'VAD-MTV',
                        )
    
    with open(eval_meta_file, "r") as f:
        full_meta = json.load(f)

    split_type = f"test-{test_id}"
    
    type_of_num_spkr = "oracle"
    tag = type_of_num_spkr+ "_cos"

    # Performing diarization.
    out_boundaries = diarize_dataset_full(
        full_meta,
        split_type,
        pval=params['pval'],
    )

    # Computing DER.
    msg = "Computing DERs for " + split_type + " set"
    logger.info(msg)
    ref_rttm = fullref_name(split_type)
    sys_rttm = out_boundaries
    [MS, FA, SER, DER_vals] = DER(
        ref_rttm,
        sys_rttm,
        params["ignore_overlap"],
        params["forgiveness_collar"],
        individual_file_scores=True,
    )

    # Writing DER values to a file. Append tag.
    der_file_name = split_type + "_DER_" + tag
    out_der_file = os.path.join(params["der_dir"], der_file_name)
    msg = "Writing DER file to: " + out_der_file
    logger.info(msg)
    diar.write_ders_file(ref_rttm, DER_vals, out_der_file)
    
    msg = f"IEMOCAP {split_type} set DER = {round(DER_vals[-1], 2)} %%\n"
    logger.info(msg)
    
    ses_wav_dic=read_json('data/iemo_ses.json')
    output_dir='data/'
    rttm_file = f'exp-eval/save/sys_rttms/IEMO_test-{test_id}/oracle_cos_SC/sys_output.rttm'

    f = open(rttm_file,'r')
    lines = f.readlines()
    dic = {}
    for line in lines:
        if line.startswith('SPEAKER'):
            tmp = line.split()
            start = float(tmp[3])
            end = start+float(tmp[4])
            spkr = tmp[-3]
            diag_id = tmp[1][:-1]
            seg_id = f"{diag_id}-{round(start,4)}_{round(end,4)}"
            dic[seg_id] = {
                "wav": ses_wav_dic[diag_id]['fea_path'],
                "start": int(start*SAMPLERATE),
                "stop": int(end*SAMPLERATE),
                "spk_id": spkr,
            }    
    dump_json(dic,output_dir+f'iemo-test-{test_id}-drz.json')
