import streamlit as st
import cv2
from PIL import Image
import numpy as np

# Charger le classificateur en cascade pour la détection des visages
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Ajouter des instructions à l'interface utilisateur
st.write("""
# Application de Détection des Visages
""")

# Télécharger une image
uploaded_file = st.file_uploader("Choisissez une image...", type=['jpg', 'jpeg', 'png','jfif'])

# Sélectionner la couleur des rectangles
rectangle_color = st.color_picker('Choisissez la couleur des rectangles', '#00FF00')

# Ajuster les paramètres de détection
scale_factor = st.slider('Ajuster le scaleFactor', 1.1, 2.0, 1.1)
min_neighbors = st.slider('Ajuster le minNeighbors', 1, 10, 5)

if uploaded_file is not None:
    # Lire l'image
    image = Image.open(uploaded_file)
    image_np = np.array(image.convert('RGB'))

    # Convertir en niveaux de gris
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Détecter les visages
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

    # Dessiner des rectangles autour des visages détectés
    for (x, y, w, h) in faces:
        cv2.rectangle(image_np, (x, y), (x + w, y + h), tuple(int(rectangle_color[i:i+2], 16) for i in (1, 3, 5)), 2)

    # Afficher l'image avec les visages détectés
    st.image(image_np, caption='Image avec visages détectés.', use_column_width=True)

    # Ajouter un bouton pour enregistrer l'image
    if st.button('Enregistrer l’image avec visages détectés'):
        cv2.imwrite('output_image.png', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        st.success('Image enregistrée avec succès!')



