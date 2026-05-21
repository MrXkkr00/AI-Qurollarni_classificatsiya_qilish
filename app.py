import streamlit as st
from fastai.vision.all import PILImage, load_learner
import pathlib
import plotly.express as px
import platform

# MUHIM: Streamlit Cloud (Linux) da Windowsda yaratilgan modelni yuklash uchun
# Operatsion tizimdan qat'i nazar WindowsPath-ni PosixPath-ga majburan o'giramiz
plt = platform.system()
if plt != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath
else:
    pathlib.PosixPath = pathlib.WindowsPath

st.title("O'zbekiston armiyasida ishlatilayotgan qurol turlarini klassifikatsiya qiluvchi model")

# Modelni yuklash (Xatolik kelib chiqmasligi uchun try-except ichiga olamiz)
@st.cache_resource
def load_my_model():
    return load_learner('./qurol_model.pkl')

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Modelni yuklashda xatolik yuz berdi: {e}")
    st.stop()

# Fayl yuklash
file = st.file_uploader('Rasm yuklash', type=['jpg', 'png', 'webp', 'jpeg'])

if file:
    # Rasmni ko'rsatish
    st.image(file, caption="Yuklangan rasm", use_container_width=True)

    # Rasmni fastai formatiga o'tkazish
    img = PILImage.create(file)
    
    # Bashorat qilish va yuklanish belgisini ko'rsatish
    with st.spinner("Model rasmga qarab qurolni aniqlamoqda..."):
        pred, pred_id, probs = model.predict(img)

    # Natijalarni chiqarish
    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]:.4f}")

    # Diagramma
    fig = px.bar(
        x=probs.numpy() * 100,  # Tensor-ni numpy-ga o'giramiz diagramma xato bermasligi uchun
        y=model.dls.vocab, 
        orientation='h',
        labels={'x': 'Ehtimollik (%)', 'y': 'Qurol turi'},
        title="Klassifikatsiya foizlarida"
    )
    st.plotly_chart(fig)
