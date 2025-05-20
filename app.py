import streamlit as st
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Función para aplicar nivel y ventana ---
def apply_window_level(img, window_width, window_center):
    img = img.astype(np.float32)
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2
    img = np.clip(img, min_val, max_val)
    img = (img - min_val) / (max_val - min_val)
    return img

# --- Título ---
st.title("Planificación de Braquiterapia")
st.write("Sube una carpeta con archivos DICOM para visualizar los cortes.")

# --- Carga de archivos DICOM ---
uploaded_files = st.file_uploader("Sube los archivos DICOM", type=["dcm"], accept_multiple_files=True)

img = None  # Inicializa para evitar errores
if uploaded_files:
    # Leer los archivos DICOM
    slices = []
    for file in uploaded_files:
        ds = pydicom.dcmread(file)
        slices.append(ds)

    # Ordenar los slices por posición
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except:
        slices.sort(key=lambda x: x.InstanceNumber)

    # Convertir a un array 3D
    img = np.stack([s.pixel_array for s in slices])
    img = img.astype(np.int16)

if img is not None:
    n_ax, n_cor, n_sag = img.shape

    # Valores de ventana y nivel por defecto
    min_val, max_val = float(img.min()), float(img.max())
    default_ww = max_val - min_val
    default_wc = min_val + default_ww / 2
    ww = st.sidebar.slider("Window Width (WW)", 1.0, default_ww * 2, default_ww)
    wc = st.sidebar.slider("Window Center (WC)", min_val, max_val, default_wc)

    # Selección del corte
    corte = st.sidebar.radio("Selecciona el tipo de corte", ("Axial", "Coronal", "Sagital"))

    if corte == "Axial":
        corte_idx = st.sidebar.slider("Índice axial", 0, n_ax - 1, n_ax // 2)
        axial_img = img[corte_idx, :, :]
        coronal_img = img[:, n_cor // 2, :]
        sagital_img = img[:, :, n_sag // 2]
    elif corte == "Coronal":
        corte_idx = st.sidebar.slider("Índice coronal", 0, n_cor - 1, n_cor // 2)
        coronal_img = img[:, corte_idx, :]
        axial_img = img[n_ax // 2, :, :]
        sagital_img = img[:, :, n_sag // 2]
    elif corte == "Sagital":
        corte_idx = st.sidebar.slider("Índice sagital", 0, n_sag - 1, n_sag // 2)
        sagital_img = img[:, :, corte_idx]
        axial_img = img[n_ax // 2, :, :]
        coronal_img = img[:, n_cor // 2, :]

    # --- Distribución en columnas ---
    col1, col2, col3 = st.columns([1, 1, 0.5])

    # --- Axial ---
    with col1:
        st.markdown("**Axial**")
        fig1, ax1 = plt.subplots()
        ax1.axis('off')
        ax1.imshow(apply_window_level(axial_img, ww, wc), cmap='gray', origin='lower')
        st.pyplot(fig1)

    # --- Coronal y Sagital ---
    with col2:
        st.markdown("**Coronal**")
        fig2, ax2 = plt.subplots()
        ax2.axis('off')
        ax2.imshow(apply_window_level(coronal_img, ww, wc), cmap='gray', origin='lower')
        st.pyplot(fig2)

        st.markdown("**Sagital**")
        fig3, ax3 = plt.subplots()
        ax3.axis('off')
        ax3.imshow(apply_window_level(sagital_img, ww, wc), cmap='gray', origin='lower')
        st.pyplot(fig3)

    # --- Logo en col3 ---
    with col3:
        st.markdown("**Logo**")
        st.image("AUNA.jpg", width=150)  # Asegúrate de tener el archivo "logo.png" en la carpeta

else:
    st.warning("Por favor sube archivos DICOM para visualizar las imágenes.")
