# -*- coding: utf-8 -*-
"""Trabajo Final NOESTRUCTURADOS.ipynb
pip install --upgrade pip
! pip install streamlit -q
import streamlit as st

%%writefile app.py
import streamlit as st
st.subheader("dtfgh")
st.title("rtdfg")

st.write("fhgjbknk")
  

texto = st.text_input("¿Me puedes dar una frase?")
maxlength = 2  #que numero de catacteres poner de limite

while len(texto) < maxlength:
  st.write(texto)
else:
  texto = ""
  st.write("El texto supera el límite de caracteres")

import warnings
warnings.filterwarnings("ignore")
import torch
from transformers import BertTokenizerFast, EncoderDecoderModel
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = 'mrm8488/bert2bert_shared-spanish-finetuned-summarization'
tokenizer = BertTokenizerFast.from_pretrained(ckpt)
model = EncoderDecoderModel.from_pretrained(ckpt).to(device)

def generate_summary(text):
    inputs = tokenizer([text], padding="max_length", truncation=True, 
                       max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    output = model.generate(input_ids, attention_mask=attention_mask)
    return tokenizer.decode(output[0], skip_special_tokens=True)

resumen = generate_summary(texto)
resumen

from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import IPython.display as ipd

models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
    "facebook/tts_transformer-es-css10",
    arg_overrides={"vocoder": "hifigan", "fp16": False}
)
model = models[0]
TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
#generator = task.build_generator(model, cfg)
generator = task.build_generator([model], cfg)
#text = "Hola, esta es una prueba."
text = resumen
sample = TTSHubInterface.get_model_input(task, text)
wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)

ipd.Audio(wav, rate=rate)

streamlit run app.py --browser.gatherUsageStats False
