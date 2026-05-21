import streamlit as st
import pathlib
import platform
import warnings
import sys
from types import ModuleType

# 1. Barcha ogohlantirishlarni o'chirish
warnings.filterwarnings("ignore")

# Fastai kutubxonasining asl klasslarini oldindan yuklab olamiz
import fastcore.transform as fastcore_trans
from fastai.vision.all import PILImage, load_learner
import plotly.express as px

# ─────────────────────────────────────────────────────────────────
# 2. FASTTRANSFORM VA PIPELINE XATOLIGINI TUZATISH
# ─────────────────────────────────────────────────────────────────
try:
    # Asosiy 'fasttransform' paketi
    fasttransform = ModuleType('fasttransform')
    fasttransform.__path__ = []  # Paket ekanligini bildiramiz
    
    # Ichidagi 'fasttransform.transform' moduli
    transform = ModuleType('fasttransform.transform')
    
    # Model qidirayotgan 'Pipeline' klassini fastai/fastcore'ning asl Pipeline'iga tenglaymiz
    transform.Pipeline = fastcore_trans.Pipeline
    
    # Ularni tizim xotirasiga (namespace) majburan joylaymiz
    fasttransform.transform = transform
    sys.modules['fasttransform'] = fasttransform
    sys.modules['fasttransform.transform'] = transform
except Exception as e:
    pass

# ─────────────────────────────────────────────────────────────────
# 3. PATHLIB MUAMMOSINI TUZATISH (Windows/Linux)
# ─────────────────────────────────────────────────────────────────
plt = platform.system()
if plt != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath
else:
    pathlib.PosixPath = pathlib.WindowsPath

st.title("O'zbekiston armiyasida ishlatilayotgan qurol turlarini klassifikatsiya qiluvchi model")

# Modelni keshlab yuklash
@st.cache_resource
def load_my_model():
    return load_learner('qurol_model.pkl')

try:
    model = load_my_model()
except Exception as e:
    st.error("Modelni yuklashda muammo bo'ldi. pkl fayli mos kelmayapti yoki buzilgan.")
    st.write(f"Xatolik tafsiloti: {e}")
    st.stop()

# Fayl yuklash oynasi
file = st.file_uploader('Rasm yuklash', type=['jpg', 'png', 'webp', 'jpeg'])

if file:
    st.image(file, caption="Yuklangan rasm", use_container_width=True)
    
    img = PILImage.create(file)
    
    with st.spinner("Model rasmga qarab qurolni aniqlamoqda..."):
        pred, pred_id, probs = model.predict(img)

    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]:.4f}")

    # Grafik chizish
    fig = px.bar(
        x=probs.numpy() * 100, 
        y=model.dls.vocab, 
        orientation='h',
        labels={'x': 'Ehtimollik (%)', 'y': 'Qurol turi'},
        title="Klassifikatsiya foizlarida"
    )
    st.plotly_chart(fig)
