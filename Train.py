#!/usr/bin/env python3
"""
Code for training the integrated system

Author
 * Wen 2023
"""
import os
import sys
import time
import torch
import json
import random
import logging
import ruamel.yaml
import numpy as np
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from transformers import WavLMForXVector
from speechbrain.dataio.dataio import length_to_mask

from utils import *

logger = logging.getLogger(__name__)


def dataio_prep(hparams,emo_lab_dic,vad_lab_dic):
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    label_encoder_asr = sb.dataio.encoder.CTCTextEncoder()
    label_encoder_asr.add_unk()
    # Define ASR text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides("wrd", "char_list", "tokens_list", "tokens")
    def SR_pipeline(wrd):
        yield wrd
        char_list = list(wrd)
        yield char_list
        tokens_list = label_encoder_asr.encode_sequence(char_list)
        yield tokens_list
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    label_encoder_spk = sb.dataio.encoder.CategoricalEncoder()
    label_encoder_emo = sb.dataio.encoder.CategoricalEncoder()
    # Define sv pipeline
    @sb.utils.data_pipeline.takes("spk_id")
    @sb.utils.data_pipeline.provides("spk_id", "spk_id_encoded")
    def SV_pipeline(spk_id):
        yield spk_id
        spk_id_encoded = label_encoder_spk.encode_sequence_torch([spk_id])
        yield spk_id_encoded

    # Define emo label pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("emo_id","emo_id_encoded","vad_lab")
    def ER_pipeline(wav):
        utt_name = wav.split('/')[-1].split('.')[0]
        emo_id = emo_lab_dic[utt_name]
        yield emo_id
        emo_id_encoded = label_encoder_emo.encode_sequence_torch([emo_id])
        yield emo_id_encoded
        vad_lab = vad_lab_dic[utt_name]
        yield torch.from_numpy(vad_lab).long()

    @sb.utils.data_pipeline.takes("wav","start","stop")
    @sb.utils.data_pipeline.provides("sig")#,"start_s","stop_s")
    def fwd_drz_pipeline(wav,start,stop):
        sig = sb.dataio.dataio.read_audio({"file":wav,"start":start,"stop":stop})

        return sig
    
    datasets = {}
    for dataset in ["train", "valid", "test","fwd_vad"]:
        if dataset == 'test':
            datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
                json_path=hparams[f"{dataset}_annotation"],
                dynamic_items=[audio_pipeline, SR_pipeline,ER_pipeline],
                output_keys=["id", "sig","emo_id_encoded", "wrd", "char_list", "tokens","vad_lab"],
            )
        elif dataset == 'fwd_vad':
            datasets[dataset]= sb.dataio.dataset.DynamicItemDataset.from_json(
                json_path=hparams[f"{dataset}_annotation"],
                dynamic_items=[audio_pipeline],
                output_keys=["id", "sig"],
            )
        else:
            datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
                json_path=hparams[f"{dataset}_annotation"],
                dynamic_items=[audio_pipeline, SR_pipeline,SV_pipeline,ER_pipeline],
                output_keys=["id", "sig","emo_id_encoded", "wrd", "char_list", "tokens","vad_lab", "spk_id_encoded"],
            )
        if dataset != 'fwd_vad':
            if hparams["sorting"] == "ascending":
                datasets[dataset] = datasets[dataset].filtered_sorted(sort_key="duration")
                hparams["dataloader_options"]["shuffle"] = False

            elif hparams["sorting"] == "descending":
                datasets[dataset] = datasets[dataset].filtered_sorted(sort_key="duration", reverse=True)
                hparams["dataloader_options"]["shuffle"] = False
            elif hparams["sorting"] == "random":
                pass
            else:
                raise NotImplementedError(
                    "sorting must be random, ascending or descending"
                )
    datasets['fwd_drz'] = sb.dataio.dataset.DynamicItemDataset.from_json(
                json_path=hparams["fwd_drz_annotation"],
                dynamic_items=[fwd_drz_pipeline],
                output_keys=["id", "sig"],
            )

    lab_enc_file_spk = os.path.join(hparams["save_folder"], "label_encoder_spk.txt")
    label_encoder_spk.load_or_create(
        path=lab_enc_file_spk, from_didatasets=[datasets["train"]], output_key="spk_id",
    )

    label_encoder_emo.load(hparams["lab_enc_file_emo"])
    label_encoder_asr.load_or_create(
        path=hparams["lab_enc_file_asr"],
        output_key="char_list",
        sequence_input=True,
    )

    return datasets,label_encoder_asr,lab_enc_file_spk,label_encoder_emo

class MTL_Brain(sb.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig     

        # Forward pass
        feats = self.modules.wav2vec2(wavs, wav_lens)
        feats=self.select_layer(feats)
        
        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            wavs_aug = self.hparams.augmentation(wavs, wav_lens)
            feats_aug = self.modules.wav2vec2(wavs_aug, wav_lens)
            feats_aug=self.select_layer(feats_aug)
            feats_SR = self.modules.interface_SR(feats_aug)
            feats_SV = self.modules.interface_SV(feats_aug)
        else:
            feats_SR = self.modules.interface_SR(feats)
            feats_SV = self.modules.interface_SV(feats)
        feats_ER = self.modules.interface_ER(feats)
        

        # SR_head
        logits = self.modules.enc_SR(feats_SR)
        p_tokens = None
        p_ctc = self.hparams.log_softmax(logits)
        if stage != sb.Stage.TRAIN:
            p_tokens = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )

        # ER_head
        src_key_padding_mask = self.make_masks(feats_ER,wav_len=wav_lens)
        pred_ER = self.modules.enc_ER(feats_ER,src_key_padding_mask=~src_key_padding_mask)

        if self.FWD_DRZ:
            return pred_ER,p_ctc, p_tokens,wav_lens

        # SV_head
        embeddings = self.modules.embedding_model_SV(feats_SV)
        output_spk = self.modules.classifier_SV(embeddings)

        # VAD_head
        feats_VAD = self.modules.interface_VAD(feats)
        pred_VAD = self.modules.enc_VAD(feats_VAD)

        return pred_ER,p_ctc, p_tokens,output_spk,wav_lens,pred_VAD

    def make_masks(self, src, wav_len=None, pad_idx=0):
        src_key_padding_mask = None
        if wav_len is not None:
            abs_len = torch.round(wav_len * src.shape[1])
            src_key_padding_mask = length_to_mask(abs_len).bool()
        return src_key_padding_mask
        
    def select_layer(self, feats):
        if len(feats.shape)>3: 
            feats=feats.permute(1,2,0,3)
        else:
            feats=feats.transpose(0,1)
        return feats


    def compute_objectives_ER(self, pred_ER, batch, stage):
        # ER_head
        ids = batch.id
        if not self.FWD_DRZ:
            emo_id = batch.emo_id_encoded
            if isinstance(emo_id,sb.dataio.batch.PaddedData):
                emo_id = emo_id.data
            emo_id = emo_id.squeeze(-1).to(self.device)
            self.error_metrics_emo.append(ids, pred_ER, emo_id)
            loss_ER = self.hparams.compute_cost_CE(pred_ER, emo_id).requires_grad_()
        else:
            loss_ER = torch.tensor(0.0)
        if stage == sb.Stage.TEST:
            assert len(ids)==len(pred_ER),[len(ids),len(pred_ER)]
            for i in range(len(ids)):
                if not self.FWD_DRZ:
                    self.test_outcome_ER[ids[i]]=(emo_id[i].detach().cpu().numpy(),pred_ER[i].detach().cpu().numpy())
                else:
                    self.test_outcome_ER[ids[i]]=(label_encoder_emo.ind2lab[int(torch.argmax(pred_ER[i]).detach().cpu().numpy())], pred_ER[i].detach().cpu().numpy())
        return loss_ER

    def compute_objectives_SR(self, p_ctc,wav_lens, predicted_tokens, batch, stage):
        # SR_head
        ids = batch.id
        
        if stage != sb.Stage.TRAIN:
            # Decode token terms to words
            predicted_words = [
                "".join(self.tokenizer.decode_ndim(utt_seq)).split(" ")
                for utt_seq in predicted_tokens
            ]
            if not self.FWD_DRZ:
                target_words = [wrd.split(" ") for wrd in batch.wrd]
                self.wer_metric.append(ids, predicted_words, target_words)
                self.cer_metric.append(ids, predicted_words, target_words)
        if stage == sb.Stage.TEST:
            for i,x in enumerate(ids):
                self.test_results_SR[ids[i]]=" ".join(predicted_words[i])
            loss_SR = torch.tensor(0.0)
        else:
            tokens, tokens_lens = batch.tokens
            loss_SR = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)
        return loss_SR

    def compute_objectives_SV(self, output_spk, batch, stage):
        # SV_head
        ids = batch.id
        if stage != sb.Stage.TEST:
            spkid = batch.spk_id_encoded
            spkid = spkid.squeeze(-1).to(self.device)
            loss_SV = self.hparams.compute_cost_CE(output_spk, spkid).requires_grad_()
            self.error_metrics_spk.append(ids, output_spk, spkid)
            return loss_SV
        else:
            return torch.tensor(0.0).to(self.device)

    def compute_objectives_VAD(self, pred_VAD ,batch, stage):
        ids = batch.id
        if stage == sb.Stage.TEST:
            assert len(ids)==len(pred_VAD),[len(ids),len(pred_VAD)]
            for i in range(len(ids)):
                self.test_outcome_VAD[ids[i]]=pred_VAD[i].detach().cpu().numpy()
        
        if self.FWD_VAD:
            return torch.tensor(0.0)
        vad_lab = batch.vad_lab
        if isinstance(vad_lab,sb.dataio.batch.PaddedData):
            vad_lab = vad_lab.data
        loss_vad = self.hparams.compute_cost_CE(pred_VAD.reshape(-1,2), vad_lab.reshape(-1,)).requires_grad_()
        self.error_metrics_vad.append(ids, pred_VAD.reshape(-1,2), vad_lab.reshape(-1,))
        return loss_vad



    def compute_objectives(self, predictions, batch, stage):
        if not self.FWD_DRZ:
            pred_ER,p_ctc, predicted_tokens,output_spk,wav_lens,pred_VAD = predictions
        else:
            pred_ER,p_ctc, predicted_tokens,wav_lens = predictions

        if self.FWD_VAD:
            loss_VAD = self.compute_objectives_VAD(pred_VAD ,batch, stage)
            return torch.tensor(0.0)

        loss_ER = self.compute_objectives_ER(pred_ER, batch, stage)
        self.loss_ER += loss_ER
        loss_SR = self.compute_objectives_SR(p_ctc,wav_lens, predicted_tokens, batch, stage)
        self.loss_SR += loss_SR
        if self.FWD_DRZ:
            return loss_ER+loss_SR
        loss_SV = self.compute_objectives_SV(output_spk ,batch, stage)
        self.loss_SV += loss_SV
        loss_VAD = self.compute_objectives_VAD(pred_VAD ,batch, stage)
        self.loss_VAD += loss_VAD

        loss = loss_ER*self.hparams.coeff_ER \
                +loss_SR*self.hparams.coeff_SR \
                +loss_SV*self.hparams.coeff_SV \
                +loss_VAD*self.hparams.coeff_VAD
        
        return loss        

    def fit_batch(self, batch):
        should_step = self.step % self.grad_accumulation_factor == 0

        outputs = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
        (loss / self.grad_accumulation_factor).backward()
        if should_step:
            if self.check_gradients(loss):
                self.optimizer_ER.step()
            self.optimizer_ER.zero_grad()
        if self.check_gradients(loss):
            self.optimizer_SR.step()
        self.optimizer_SR.zero_grad()

        return loss.detach().cpu()

    def on_stage_start(self, stage, epoch=None):

        self.start_time = time.time()

        self.loss_ER = torch.tensor(0.0).to(self.device)
        self.loss_SR = torch.tensor(0.0).to(self.device)
        self.loss_SV = torch.tensor(0.0).to(self.device)
        self.loss_VAD = torch.tensor(0.0).to(self.device)

        self.error_metrics_emo = self.hparams.error_stats_Acc()
        self.error_metrics_vad = self.hparams.error_stats_Acc()

        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.wer_computer()
        if stage == sb.Stage.TEST:
            self.test_results_SR = {}
            self.test_outcome_ER = {}
            self.test_target_ER = {}
            self.test_outcome_VAD = {}

        self.error_metrics_spk = self.hparams.error_stats_Acc()
        

    def on_stage_end(self, stage, stage_loss, epoch=None):
        self.elapse_time = time.time() - self.start_time

        stage_stats = {"loss": stage_loss,
                        "loss_ER": self.loss_ER.item(),
                        "loss_SR": self.loss_SR.item(),
                        "loss_SV": self.loss_SV.item(),
                        "loss_VAD": self.loss_VAD.item(),
                        "duration": self.elapse_time,
                        }
        if not self.FWD_VAD and not self.FWD_DRZ:
            stage_stats['Acc_emo']=self.error_metrics_emo.summarize()
            stage_stats['Acc_vad']=self.error_metrics_vad.summarize()
        if stage != sb.Stage.TEST:
            stage_stats["Acc_spk"] = self.error_metrics_spk.summarize()
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            if not self.FWD_VAD and not self.FWD_DRZ:
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")
    

        if stage == sb.Stage.VALID:
            old_lr_SR, new_lr_SR = self.hparams.lr_annealing_SR(stage_stats["WER"])
            old_lr_SV, new_lr_SV = self.hparams.lr_annealing_SV(stage_stats["loss_SV"])
            old_lr_VAD, new_lr_VAD = self.hparams.lr_annealing_VAD(stage_stats["loss_VAD"])
            old_lr_w2v2, new_lr_w2v2 = self.hparams.lr_annealing_w2v2(stage_stats["loss"])

            if isinstance(self.hparams.lr_annealing_ER, sb.nnet.schedulers.NewBobScheduler):
                old_lr_ER, new_lr_ER = self.hparams.lr_annealing_ER(1-stage_stats["Acc_emo"])
            else:
                old_lr_ER, new_lr_ER = self.hparams.lr_annealing_ER(epoch)

            if self.hparams.diff_lr:
                sb.nnet.schedulers.update_learning_rate(self.optimizer_SR, new_lr_w2v2,[0])
                sb.nnet.schedulers.update_learning_rate(self.optimizer_ER, new_lr_ER,[0])
                sb.nnet.schedulers.update_learning_rate(self.optimizer_SR, new_lr_SR,[1])
                sb.nnet.schedulers.update_learning_rate(self.optimizer_ER, new_lr_SV,[1])
                sb.nnet.schedulers.update_learning_rate(self.optimizer_ER, new_lr_SV,[2])
                sb.nnet.schedulers.update_learning_rate(self.optimizer_ER, new_lr_VAD,[3])
            else:
                sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
                
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_ER": old_lr_ER,
                    "lr_SR": old_lr_SR,
                    "lr_SV": old_lr_SV,
                    "lr_VAD": old_lr_VAD,
                    "lr_w2v2": old_lr_w2v2,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            self.checkpointer.save_and_keep_only(
                meta=stage_stats, 
                min_keys=["loss"],
                num_to_keep=3,
                name=f"E{self.hparams.epoch_counter.current}")

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
            stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
            test_stats=stage_stats,
            )
            
            if self.FWD_VAD:
                np.save(os.path.join(self.hparams.output_folder,f'VAD_test_outcome-{self.hparams.test_id}-E{self.hparams.epoch_counter.current}.npy'),self.test_outcome_VAD)
            elif self.FWD_DRZ:
                json_file=f"{self.hparams.output_folder}/SR_test_outcome-{self.hparams.test_id}-drz.json"
                with open(json_file, mode="w") as json_f:
                    json.dump(self.test_results_SR, json_f, indent=2)
                np.save(os.path.join(self.hparams.output_folder,f'ER_test_outcome-CAT-{self.hparams.test_id}-drz.npy'),self.test_outcome_ER)

            else:
                with open(self.hparams.wer_file, "w") as w:
                    self.wer_metric.write_stats(w)
                json_file=self.hparams.wer_file.replace('.txt','.json').replace('wer',f'SR_test_outcome-E{self.hparams.epoch_counter.current}')
                with open(json_file, mode="w") as json_f:
                    json.dump(self.test_results_SR, json_f, indent=2)
                np.save(os.path.join(self.hparams.output_folder,f'ER_test_outcome-CAT-E{self.hparams.epoch_counter.current}.npy'),self.test_outcome_ER)

    def init_optimizers(self):
        if self.hparams.opt_class_ER is not None:
            if self.hparams.diff_lr:
                params_SV1 = []
                params_SV1.extend(self.modules.classifier_SV.parameters())

                params_SV2 = []
                params_SV2.extend(self.modules.interface_SV.parameters())
                params_SV2.extend(self.modules.embedding_model_SV.parameters())

                params_ER = []
                params_ER.extend(self.modules.interface_ER.parameters())
                params_ER.extend(self.modules.enc_ER.parameters())

                params_SR = []
                params_SR.extend(self.modules.interface_SR.parameters())
                params_SR.extend(self.modules.enc_SR.parameters())

                params_VAD = []
                params_VAD.extend(self.modules.interface_VAD.parameters())
                params_VAD.extend(self.modules.enc_VAD.parameters())

                other_params = [p for p in self.modules.parameters() if p not in set(params_SV1+params_SV2+params_ER+params_SR+params_VAD)]
                assert len(other_params)==len([x for x in self.modules.wav2vec2.parameters()])

                self.optimizer_SR = self.hparams.opt_class_SR([
                    {'params': other_params, 'lr': self.hparams.lr_w2v2},
                    {'params': params_SR, 'lr': self.hparams.lr_SR},      
                ], lr=self.hparams.lr, weight_decay = self.hparams.weight_decay_SR )
                self.optimizer_ER = self.hparams.opt_class_ER([
                    {'params': params_ER, 'lr': self.hparams.lr_ER, 'weight_decay': self.hparams.weight_decay_ER*2.5},
                    {'params': params_SV1, 'lr': self.hparams.lr_SV}, 
                    {'params': params_SV2, 'lr': self.hparams.lr_SV}, 
                    {'params': params_VAD, 'lr': self.hparams.lr_VAD},     
                ], lr=self.hparams.lr, weight_decay = self.hparams.weight_decay_ER )
            else:
                self.optimizer = self.hparams.opt_class(self.modules.parameters())


        if self.hparams.load_pretrained_SV:
            # load parameters from wavlm-base-plus-sv
            pretrained_SV_model = WavLMForXVector.from_pretrained(self.hparams.pretrained_SV_hub)

            state_dict = self.modules.interface_SV.state_dict()
            state_dict['layer_weights'] = pretrained_SV_model.layer_weights.data
            self.modules.interface_SV.load_state_dict(state_dict)
            assert len(self.modules.embedding_model_SV.state_dict().keys()) == len(self.modules.embedding_model_SV.projector.state_dict().keys()) \
                                                                            + len(self.modules.embedding_model_SV.tdnn.state_dict().keys()) \
                                                                            + len(self.modules.embedding_model_SV.feature_extractor.state_dict().keys())
            
            self.modules.embedding_model_SV.projector.load_state_dict(pretrained_SV_model.projector.state_dict())
            self.modules.embedding_model_SV.tdnn.load_state_dict(pretrained_SV_model.tdnn.state_dict())
            self.modules.embedding_model_SV.feature_extractor.load_state_dict(pretrained_SV_model.feature_extractor.state_dict())

            if self.hparams.freeze_embedding_model:
                for param in self.modules.embedding_model_SV.parameters():
                    param.requires_grad = False
            if self.hparams.freeze_interface:
                self.modules.interface_SV.requires_grad = False


        elif self.checkpointer is not None:
            self.checkpointer.add_recoverable("optimizer_SR", self.optimizer_SR)
            self.checkpointer.add_recoverable("optimizer_ER", self.optimizer_ER)

        if not self.hparams.freeze_wav2vec:
            self.checkpointer.add_recoverable("wav2vec2", self.modules.wav2vec2)


    def on_evaluate_start(self, max_key=None, min_key=None):
        # Recover best checkpoint for evaluation
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible(
                max_key=max_key,
                min_key=min_key,
                device=torch.device(self.device),
            )
        if self.AVERAGE:
            logger.info("Averaging checkpoints")
            ckpt_list = self.checkpointer.find_checkpoints(max_key=max_key,
                                                                min_key=min_key,
                                                                max_num_checkpoints = 5)
            for k,v in self.checkpointer.recoverables.items():
                if k in ['interface_ER', 'interface_SR','interface_SV','interface_VAD',
                        'enc_ER', 'enc_SR', 'enc_VAD','embedding_model_SV', 'classifier_SV','wav2vec2']:
                    averaged_state = sb.utils.checkpoints.average_checkpoints(ckpt_list, k)
                    _ = v.load_state_dict(averaged_state)
            self.hparams.epoch_counter.current = -1




if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    if '--device' not in sys.argv[1:]:
        run_opts['device']= 'cuda' if torch.cuda.is_available() else 'cpu'

    ruamel_yaml = ruamel.yaml.YAML()
    with open(hparams_file) as tmp:
        preview = ruamel_yaml.load(tmp)
    cmd_overriden = {k:v for k,v in run_opts.items() if k in preview}   
    overrides = ruamel_yaml.load(overrides)
    if not overrides:
        overrides=cmd_overriden
    else:
        overrides.update(cmd_overriden)
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    emo_lab_dic=np.load(hparams['emo_lab'],allow_pickle=True).item()
    vad_lab_dic=np.load(hparams['vad_lab'],allow_pickle=True).item()

    set_seed(hparams['seed'])

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Create dataset objects "train", "valid", and "test".
    datasets,label_encoder_asr,lab_enc_file_spk,label_encoder_emo = dataio_prep(hparams,emo_lab_dic,vad_lab_dic)
    
    MTL_brain = MTL_Brain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class_ER"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    MTL_brain.tokenizer = label_encoder_asr

    MTL_brain.AVERAGE=True
    MTL_brain.FWD_VAD=False
    MTL_brain.FWD_DRZ=False
    if not hparams['FWD_VAD'] and not hparams['FWD_DRZ']:
        MTL_brain.fit(
            epoch_counter=MTL_brain.hparams.epoch_counter,
            train_set=datasets["train"],
            valid_set=datasets["valid"],
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["dataloader_options"],
        )
    
        
        test_stats = MTL_brain.evaluate(
        test_set=datasets["test"],
        min_key="loss",
        test_loader_kwargs={'batch_size':1},
        )

    # During testing, input dialogue to the system. Windows with width 3 and overlap 2 applies.
    if hparams['FWD_VAD']: 
        MTL_brain.FWD_VAD=True
        test_stats = MTL_brain.evaluate(
        test_set=datasets["fwd_vad"],
        min_key="loss",
        test_loader_kwargs={'batch_size':1},
        )
        MTL_brain.FWD_VAD=False
    
    if hparams['FWD_DRZ']:
        MTL_brain.FWD_DRZ=True
        test_stats = MTL_brain.evaluate(
        test_set=datasets["fwd_drz"],
        min_key="loss",
        test_loader_kwargs={'batch_size':1},
        )
        MTL_brain.FWD_DRZ=False
