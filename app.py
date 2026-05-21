import streamlit as st
from fastai.vision.all import PILImage, load_learner
import pathlib
import plotly.express as px
import warnings
import platform
from PIL import Image

# Operatsion tizimni tekshirish va pathlib muammosini hal qilish
plt = platform.system()
if plt != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

st.title("O'zbekiston armiyasida ishlatilayotgan qurol turlarini klassifikatsiya qiluvchi model")

# Modelni yuklash
model = load_learner('./qurol_model.pkl')

# Fayl yuklash
file = st.file_uploader('Rasm yuklash', type=['jpg', 'png', 'webp'])

if file:
    st.image(file)

    img = PILImage.create(file)
    
    # Bashorat qilish
    pred, pred_id, probs = model.predict(img)

    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]:.4f}")

    # Diagramma
    fig = px.bar(x=probs*100, y=model.dls.vocab, orientation='h')
    st.plotly_chart(fig)
