import streamlit as st
import pathlib
import platform
import warnings

# 1. Barcha ogohlantirishlarni (warnings) butunlay bloklaymiz
warnings.filterwarnings("ignore")

# 2. Pathlib mosligini ta'minlash (Windows/Linux muammosi uchun)
plt = platform.system()
if plt != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath
else:
    pathlib.PosixPath = pathlib.WindowsPath

# Kutubxonalarni ogohlantirishlar o'chirilgandan keyin yuklaymiz
from fastai.vision.all import PILImage, load_learner
import plotly.express as px

st.title("O'zbekiston armiyasida ishlatilayotgan qurol turlarini klassifikatsiya qiluvchi model")

# Modelni keshga olib yuklash
@st.cache_resource
def load_my_model():
    return load_learner('qurol_model.pkl')

# Model yuklanishini tekshirish
try:
    model = load_my_model()
except Exception as e:
    st.error("Modelni yuklashda muammo bo'ldi. Versiyalar o'zgargani sababli bo'lishi mumkin.")
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
