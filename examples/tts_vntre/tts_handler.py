import logging
import os
import re
import uuid
import json
import torch
import zipfile
import unicodedata

import numpy as np
from types import SimpleNamespace
from scipy.io.wavfile import write
from preprocess_new import preprocess_text

from models import DurationNet, SynthesizerTrn
from scipy.io import wavfile

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class WaveGlowSpeechSynthesizer(BaseHandler):
    def __init__(self):
        # self.waveglow_model = None
        self.generator_model = None
        # self.tacotron2_model = None
        self.duration_model = None
        self.mapping = None
        self.device = None
        self.initialized = False
        self.metrics = None
        self.hps = None
        self.sil_idx = None
        self.dict_convert = {}
        self.phone_set = None
        self.dict_unit = {}

    # From https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/inference.py
    def _unwrap_distributed(self, state_dict):
        """
        Unwraps model from DistributedDataParallel.
        DDP wraps model in additional "module.", it needs to be removed for single
        GPU inference.
        :param state_dict: model's state dict
        """

        params = {}
        for k, v in state_dict.items():
            k = k[7:] if k.startswith("module.") else k
            params[k] = v
        return params
    
    def _load_duration_model(self, model_dir):
        self.duration_model = DurationNet(self.hps.data.vocab_size, 64, 4).to(self.device)
        self.duration_model.load_state_dict(torch.load(model_dir + "/duration_model.pth", map_location=self.device))
        self.duration_model = self.duration_model.eval()
    
    def initialize(self, ctx):
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        if not torch.cuda.is_available() or properties.get("gpu_id") is None:
            raise RuntimeError("This model is not supported on CPU machines.")
        
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")))

        # Dict convert
        with open(model_dir + "/convert.txt", 'r', encoding='utf-8') as f:
            convert = f.read().strip().split("\n")
        for line  in convert:
            out = re.split(':', line)
            out[0] = ' ' + out[0] + ' '
            out[1] = ' ' + out[1] + ' '
            self.dict_convert[out[0]] = out[1]
        
        # Dict unit
        with open(model_dir + "/units.txt", 'r', encoding='utf-8') as f:
            units = f.read().strip().split("\n")
        for line in units:
            out = re.split(':', line)
            out[0] = out[0].strip()
            out[1] = out[1].strip()
            self.dict_unit[out[0]] = out[1]
        
        # Phone set
        with open(model_dir + "/phone_set.json", 'r') as f:
            self.phone_set = json.load(f)
        
        # sil index
        self.sil_idx = self.phone_set.index("sil")

        # hps
        with open(model_dir + "/config.json", "rb") as f:
            self.hps = json.load(f, object_hook=lambda x: SimpleNamespace(**x))

        # Load model generator
        self.generator = SynthesizerTrn(
            self.hps.data.vocab_size,
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **vars(self.hps.model),
        ).to(self.device)
        del self.generator.enc_q

        ckpt = torch.load(model_dir + '/generator.pth', map_location=self.device)
        ckpt_state_dict = self._unwrap_distributed(ckpt)
        self.generator.load_state_dict(ckpt_state_dict, strict=False)
        del ckpt, ckpt_state_dict
        self.generator = self.generator.eval()

        self._load_duration_model(model_dir)

        logger.debug("TTS VNTre model file loaded successfully")
        self.initialized = True
    
    def segment_text(self, text):
        text = re.sub("['\(\)\"\*\#\&\@\+\=']", " ", text)
        text = re.sub('[.]+', '.', text)

        split_patterns = "\n+|\.\s+"
        paragraphs = re.split(split_patterns, text)

        return paragraphs
        
    def text_to_phone_idx(self, text):
        # lowercase
        text = text.lower()
        # unicode normalize
        text = unicodedata.normalize("NFKC", text)
        text = text.replace(".", " . ")
        text = text.replace(",", " , ")
        text = text.replace(";", " ; ")
        text = text.replace(":", " : ")
        text = text.replace("!", " ! ")
        text = text.replace("?", " ? ")
        text = text.replace("(", " ( ")

        # remove redundant spaces
        text = re.sub(r"\s+", " ", text)
        # remove leading and trailing spaces
        text = text.strip()
        # convert words to phone indices
        tokens = []
        for c in text:
            # if c is "," or ".", add <sil> phone
            if c in ":,.!?;(":
                tokens.append(self.sil_idx)
            elif c in self.phone_set:
                tokens.append(self.phone_set.index(c))
            elif c == " ":
                # add <sep> phone
                tokens.append(0)
        if tokens[0] != self.sil_idx:
            # insert <sil> phone at the beginning
            tokens = [self.sil_idx, 0] + tokens
        if tokens[-1] != self.sil_idx:
            tokens = tokens + [0, self.sil_idx]
        return tokens

    def preprocess(self, data):
        """
        converts text to sequence of IDs using tacatron2 text_to_sequence
        with english cleaners to transform text and standardize input
        (ex: lowercasing, expanding abbreviations and numbers, etc.)
        returns an Numpy array
        """
        text = data[0].get("data")

        if text is None:
            text = data[0].get("body")
        text = text.decode("utf-8")

        # print("Text: ", text)
        
        text_segment = self.segment_text(text.lower())
        text_segment = [preprocess_text(text) for text in text_segment]

        # print("Text segment: ", text_segment)

        paragraphs = []
        for p in text_segment:
            if p.strip() == "":
                continue
            if len(p) > 500:
                sents = []
                word_p = p.split()
                num_s = len(word_p) // 50
                for i in range(num_s):
                    if i == num_s - 1:
                        sents.append(' '.join(word_p[i*50:]))
                    else:
                        sents.append(' '.join(word_p[i*50:(i+1)*50]))
                paragraphs += sents
            else:
                paragraphs.append(p)
        
        # print("Paragraphs: ", paragraphs)

        
        iterations = []
        for p in paragraphs:
            phone_idx = self.text_to_phone_idx(p)
            batch = {
                "phone_idx": np.array([phone_idx]),
                "phone_length": np.array([len(phone_idx)]),
                "text": p
            }    

            iterations.append(batch)

        # print("Interation: ", iterations)

        return iterations

    def inference(self, data):
        output = None
        for batch in data:
            # print("Batch: ", batch)
            # predict phoneme duration
            phone_length = torch.from_numpy(batch["phone_length"]).long().to(self.device)
            phone_idx = torch.from_numpy(batch["phone_idx"]).long().to(self.device)
            with torch.inference_mode():
                phone_duration = self.duration_model(phone_idx, phone_length)[:, :, 0] * 1000
            phone_duration = torch.where(
                phone_idx == self.sil_idx, torch.clamp_min(phone_duration, 200), phone_duration
            )
            phone_duration = torch.where(phone_idx == 0, 0, phone_duration)

            # generate waveform
            end_time = torch.cumsum(phone_duration, dim=-1)
            start_time = end_time - phone_duration
            start_frame = start_time / 1000 * self.hps.data.sampling_rate / self.hps.data.hop_length
            end_frame = end_time / 1000 * self.hps.data.sampling_rate / self.hps.data.hop_length
            spec_length = end_frame.max(dim=-1).values
            pos = torch.arange(0, spec_length.item(), device=self.device)
            attn = torch.logical_and(
                pos[None, :, None] >= start_frame[:, None, :],
                pos[None, :, None] < end_frame[:, None, :],
            ).float()
            with torch.inference_mode():
                y_hat = self.generator.infer(
                    phone_idx, phone_length, spec_length, attn, max_len=None, noise_scale=0.0
                )[0]

            wave = y_hat[0, 0].data.detach().cpu().numpy()

            if output is None:
                output = wave
            else:
                output = np.concatenate((output, wave))
        
        return self.hps.data.sampling_rate, output

    def postprocess(self, inference_output):
        audio_numpy = inference_output[1]
        sample_rate = inference_output[0]
        path = "/tmp/{}.wav".format(uuid.uuid4().hex)
        write(path, sample_rate, audio_numpy)
        with open(path, "rb") as output:
            data = output.read()
        os.remove(path)
        return [data]
