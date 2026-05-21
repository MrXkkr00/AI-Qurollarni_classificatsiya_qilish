import pathlib
import platform

# Tizim moslashuvi
plt = platform.system()
if plt != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

# Streamlit va qolgan importlar endi pastdan boshlanadi
import streamlit as st
import plotly.express as px
from PIL import Image
import fastai
from fastai.vision.all import *

st.title("O'zbekiston armiyasida ishlatilayotgan qurol turlarini klassifikatsiya qiluvchi model")

# Modelni keshga olib yuklash (har safar sahifa yangilanganda qayta yuklanmaydi)
@st.cache_resource
def load_my_model():
    return load_learner('./qurol_model.pkl')

# Modelni yuklashga urinib ko'ramiz
try:
    model = load_my_model()
except Exception as e:
    st.error("Modelni yuklashda muammo yuz berdi. Iltimos, requirements.txt versiyalarini tekshiring.")
    st.exception(e)
    st.stop()

# Fayl yuklash
file = st.file_uploader('Rasm yuklash', type=['jpg','png', 'webp'])

if file:
    # Rasmni ko'rsatish
    st.image(file, caption="Yuklangan rasm")

    img = PILImage.create(file)
    
    # Bashorat qilish
    pred, pred_id, probs = model.predict(img)

    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]:.4f}")

    # Diagramma yaratish (probs obyektini numpy massiviga o'tkazish xavfsizroq)
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
