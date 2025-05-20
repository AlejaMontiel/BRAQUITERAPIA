import os
import io
import zipfile
import tempfile

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import SimpleITK as sitk
from skimage.transform import resize
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Brachyanalysis")

st.markdown("""
<style>
    .giant-title { color: #28aec5; text-align: center; font-size: 72px; margin: 30px 0; font-weight: bold; text-transform: uppercase; letter-spacing: 2px; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }
    .sub-header { color: #c0d711; font-size: 24px; margin-bottom: 15px; font-weight: bold; }
    .stButton>button { background-color: #28aec5; color: white; border: none; border-radius: 4px; padding: 8px 16px; }
    .stButton>button:hover { background-color: #1c94aa; }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown('<p class="sub-header">Visualizador de imágenes DICOM</p>', unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Sube un archivo ZIP con tus archivos DICOM", type="zip")

def find_dicom_series(directory):
    series_found = []
    for root, dirs, files in os.walk(directory):
        try:
            series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(root)
            for sid in series_ids:
                file_list = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(root, sid)
                if file_list:
                    series_found.append((sid, root, file_list))
        except Exception:
            continue
    return series_found

def apply_window_level(image, window_width, window_center):
    img_float = image.astype(float)
    min_v = window_center - window_width / 2.0
    max_v = window_center + window_width / 2.0
    windowed = np.clip(img_float, min_v, max_v)
    if max_v != min_v:
        return (windowed - min_v) / (max_v - min_v)
    return np.zeros_like(img_float)

dirname = None
if uploaded_file:
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(io.BytesIO(uploaded_file.read()), 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    dirname = temp_dir
    st.sidebar.success("Archivos extraídos correctamente.")

dicom_series = None
img = None
original_image = None
if dirname:
    with st.spinner('Buscando series DICOM...'):
        dicom_series = find_dicom_series(dirname)
    if dicom_series:
        options = [f"Serie {i + 1}: {series[0][:10]}... ({len(series[2])} archivos)" for i, series in enumerate(dicom_series)]
        selection = st.sidebar.selectbox("Seleccionar serie DICOM:", options)
        selected_idx = options.index(selection)
        sid, dirpath, files = dicom_series[selected_idx]
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(files)
        data = reader.Execute()
        img = sitk.GetArrayViewFromImage(data)
        original_image = img
    else:
        st.sidebar.error("No se encontraron DICOM válidos en el ZIP cargado.")

if img is not None:
    n_ax, n_cor, n_sag = img.shape
    min_val, max_val = float(img.min()), float(img.max())
    default_ww = max_val - min_val
    default_wc = min_val + default_ww / 2
    ww, wc = default_ww, default_wc

    st.markdown('<h2 style="color:#28aec5;text-align:center;">Vistas 2D: Axial, Coronal y Sagital</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    axial_idx = col1.slider("Índice axial", 0, n_ax - 1, n_ax // 2, key="axial")
    coronal_idx = col2.slider("Índice coronal", 0, n_cor - 1, n_cor // 2, key="coronal")
    sagital_idx = col3.slider("Índice sagital", 0, n_sag - 1, n_sag // 2, key="sagital")

    axial_img = img[axial_idx, :, :]
    coronal_img = img[:, coronal_idx, :]
    sagital_img = img[:, :, sagital_idx]

    def show_slice(img_slice, title):
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.set_title(title, color='#28aec5', fontsize=12)
        ax.imshow(apply_window_level(img_slice, ww, wc), cmap='gray', origin='lower')
        return fig

    col1.pyplot(show_slice(axial_img, "Vista Axial"))
    col2.pyplot(show_slice(coronal_img, "Vista Coronal"))
    col3.pyplot(show_slice(sagital_img, "Vista Sagital"))

    st.markdown('<h2 style="color:#28aec5;text-align:center;margin-top:50px;">Reconstrucción 3D</h2>', unsafe_allow_html=True)

    # Vista 3D
    target_shape = (64, 64, 64)
    img_resized = resize(original_image, target_shape, anti_aliasing=True)
    x, y, z = np.mgrid[0:target_shape[0], 0:target_shape[1], 0:target_shape[2]]
    fig3d = go.Figure(data=go.Volume(
        x=x.flatten(), y=y.flatten(), z=z.flatten(),
        value=img_resized.flatten(),
        opacity=0.1,
        surface_count=15,
        colorscale="Gray",
    ))
    fig3d.update_layout(margin=dict(l=10, r=10, t=10, b=10))

    st.plotly_chart(fig3d, use_container_width=True)
