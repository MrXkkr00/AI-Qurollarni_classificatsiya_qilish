import streamlit as st
import pathlib
import platform

# ─────────────────────────────────────────────────────────────────
# ENG MUHIM QISM: Streamlit Linux serverida Windows modelini o'qish
# ─────────────────────────────────────────────────────────────────
plt = platform.system()
if plt != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath
else:
    pathlib.PosixPath = pathlib.WindowsPath

# Fastai va boshqa kutubxonalarni pathlib o'zgargandan KEYIN yuklash kerak
from fastai.vision.all import PILImage, load_learner
import plotly.express as px

st.title("O'zbekiston armiyasida ishlatilayotgan qurol turlarini klassifikatsiya qiluvchi model")

# Modelni xavfsiz yuklash funksiyasi
@st.cache_resource
def load_my_model():
    # .pkl faylingiz aynan shu papkada (app.py bilan yonma-yon) joylashgan bo'lishi kerak
    return load_learner('qurol_model.pkl')

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Modelni yuklashda ichki muammo yuz berdi. "
             f"Katta ehtimol bilan .pkl fayl to'liq yuklanmagan yoki mos kelmayapti.")
    st.info(f"Xatolik tafsiloti: {e}")
    st.stop()

# Fayl yuklash
file = st.file_uploader('Rasm yuklash', type=['jpg', 'png', 'webp', 'jpeg'])

if file:
    st.image(file, caption="Yuklangan rasm", use_container_width=True)

    img = PILImage.create(file)
    
    with st.spinner("Model rasmga qarab qurolni aniqlamoqda..."):
        pred, pred_id, probs = model.predict(img)

    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]:.4f}")

    # Diagramma
    fig = px.bar(
        x=probs.numpy() * 100, 
        y=model.dls.vocab, 
        orientation='h',
        labels={'x': 'Ehtimollik (%)', 'y': 'Qurol turi'},
        title="Klassifikatsiya foizlarida"
    )
    st.plotly_chart(fig)
