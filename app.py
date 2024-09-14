import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import subprocess
import tempfile

# Função para exibir imagem
def display_image(image):
    st.image(image, use_column_width=True)

# Configuração do layout
st.title("Virtual Try-On Demo")
st.write("Faça upload das imagens e clique no botão para processar.")

# Upload de imagens
uploaded_person = st.file_uploader("Escolha a imagem da pessoa", type=["jpg", "png"])
uploaded_cloth = st.file_uploader("Escolha a imagem da roupa", type=["jpg", "png"])

if uploaded_person and uploaded_cloth:
    # Salvar imagens temporariamente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_person:
        temp_person.write(uploaded_person.read())
        temp_person_path = temp_person.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_cloth:
        temp_cloth.write(uploaded_cloth.read())
        temp_cloth_path = temp_cloth.name
    
    # Renomear arquivos para os nomes esperados pelo código
    os.rename(temp_person_path, "./static/origin_web.jpg")
    os.rename(temp_cloth_path, "./static/cloth_web.jpg")

    st.write("Imagens carregadas com sucesso!")

    # Mostrar imagens carregadas
    original = cv2.cvtColor(cv2.imread("./static/origin_web.jpg"), cv2.COLOR_BGR2RGB)
    cloth = cv2.cvtColor(cv2.imread("./static/cloth_web.jpg"), cv2.COLOR_BGR2RGB)
    st.subheader("Imagem da pessoa")
    display_image(original)
    st.subheader("Imagem da roupa")
    display_image(cloth)

    # Botão para rodar o processamento
    if st.button('Rodar Processamento'):
        with st.spinner('Processando...'):
            # Execute o script principal
            subprocess.run(["python", "main.py", "--background", "True"])

            # Mostrar resultado
            final_img = Image.open("./static/finalimg.png")
            st.subheader("Resultado Final")
            display_image(final_img)

            st.write("Processamento concluído!")

else:
    st.write("Por favor, faça o upload das imagens para começar.")

