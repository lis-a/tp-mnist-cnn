from streamlit_drawable_canvas import st_canvas
import streamlit as st
import numpy as np
import requests
from PIL import Image, ImageOps
import io

st.title("🖌️ Reconnaissance de chiffres manuscrits")
st.write("Dessine un chiffre (0–9) ci-dessous, et je te dirai lequel c’est.")

# Zone de dessin
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Affichage de l'image dessinée
    img = canvas_result.image_data

    # Convertir l’image en niveau de gris, inversée, et redimensionnée
    img = Image.fromarray((255 - img[:, :, 0]).astype(np.uint8))
    img = ImageOps.invert(img)
    img = img.resize((28, 28)).convert("L")

    st.image(img, caption="Image normalisée (28x28)", width=100)

    # Envoi à l’API FastAPI
    if st.button("🔍 Prédire le chiffre"):
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        files = {"file": ("image.png", buffered.getvalue(), "image/png")}

        try:
            response = requests.post("http://localhost:8000/api/v1/predict", files=files)
            if response.status_code == 200:
                st.success(f"✨ Chiffre prédit : **{response.json()['prediction']}**")
            else:
                st.error("Erreur côté API.")
        except Exception as e:
            st.error(f"Erreur de connexion à l'API : {e}")
