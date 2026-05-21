import streamlit as st
import pathlib
import platform
import plotly.express as px
from PIL import Image

# 1. Tizim yo'llarini to'g'rilash (Importlardan oldin turishi shart!)
plt = platform.system()
if plt != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

# 2. FastAI va Torch importlari
import torch
import fastai
from fastai.vision.all import *

st.title("O'zbekiston armiyasida ishlatilayotgan qurol turlarini klassifikatsiya qiluvchi model")

# 3. Modelni yuklashning eng xavfsiz usuli
@st.cache_resource
def load_my_model():
    # Ba'zan fastai modelni yuklashda shu funksiyalarni qidiradi, ularni soxtalashtiramiz
    import __main__
    
    # Agar modelni o'qitishda maxsus funksiya ishlatgan bo'lsangiz, buni yozing:
    # __main__.sizning_funksiya_nomi = o'sha_funksiya
    
    # Model faylini oddiy tekshirish va yuklash
    model_path = pathlib.Path('./qurol_model.pkl')
    if not model_path.exists():
        st.error("Xato: 'qurol_model.pkl' fayli loyiha jildida topilmadi!")
        st.stop()
        
    return load_learner(model_path)

# Modelni ishga tushirish
try:
    model = load_my_model()
except Exception as e:
    st.error("⚠️ Modelni yuklashda ichki tizim xatosi yuz berdi!")
    st.info("Bu ko'pincha model fayli to'liq yuklanmagani yoki GitHub-ga noto'g'ri joylangani sababli bo'ladi.")
    st.exception(e)
    st.stop()

# 4. Foydalanuvchi interfeysi (Rasm yuklash)
file = st.file_uploader('Rasm yuklash', type=['jpg', 'jpeg', 'png', 'webp'])

if file:
    st.image(file, caption="Yuklangan rasm")
    
    # Rasm tayyorlash va bashorat
    img = PILImage.create(file)
    pred, pred_id, probs = model.predict(img)

    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]:.4f}")

    # Diagramma yaratish (Numpy massiviga o'tkazilgan holatda)
    probabilities = probs.numpy() * 100
    labels = model.dls.vocab

    fig = px.bar(
        x=probabilities, 
        y=labels, 
        orientation='h',
        labels={'x': 'Ehtimollik (%)', 'y': 'Qurol turi'},
        title="Klassifikatsiya natijalari"
    )
    st.plotly_chart(fig)
