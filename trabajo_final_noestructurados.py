# -*- coding: utf-8 -*-
"""Trabajo Final NOESTRUCTURADOS.ipynb
! pip install streamlit -q
import streamlit as st

st.set_page_config(page_title="Page Title",layout="wide")

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
  
streamlit run trabajo_final_noestructurados.py
