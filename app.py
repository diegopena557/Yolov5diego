import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys

st.set_page_config(
    page_title="🔍 Detección Inteligente de Objetos",
    page_icon="🤖",
    layout="wide"
)

# Estilos personalizados
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stApp {
        background-color: #0e1117;
    }
    h1, h2, h3, h4 {
        color: #00c0ff;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_yolov5_model(model_path='yolov5s.pt'):
    try:
        import yolov5
        model = yolov5.load(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return None

# Diccionario con explicaciones de objetos
object_descriptions = {
    "person": "👤 Se ha detectado una persona. Ideal para análisis de seguridad o conteo de visitantes.",
    "car": "🚗 Vehículo detectado. Los modelos YOLO se usan también en sistemas de tráfico inteligente.",
    "dog": "🐶 Es un perro. Detección útil para monitoreo de fauna urbana.",
    "cat": "🐱 Gato detectado. Los modelos pueden ayudar en rescates o adopciones.",
    "bottle": "🥤 Botella detectada. Aplicación común en reciclaje automatizado.",
    "cell phone": "📱 Teléfono detectado. Común en análisis de comportamiento en tiendas.",
    "chair": "🪑 Silla detectada. Usado en diseño de interiores o conteo de mobiliario."
}

st.title("🤖 Detección Inteligente de Objetos con Explicaciones")
st.markdown("Captura una imagen y descubre no solo **qué hay**, sino también **para qué sirve** lo que ve el modelo.")

with st.spinner("Cargando modelo YOLOv5..."):
    model = load_yolov5_model()

if model:
    st.sidebar.header("⚙️ Configuración")
    model.conf = st.sidebar.slider('Confianza mínima', 0.0, 1.0, 0.25, 0.01)
    model.iou = st.sidebar.slider('Umbral IoU', 0.0, 1.0, 0.45, 0.01)
    
    picture = st.camera_input("📸 Captura una imagen para analizar")

    if picture:
        bytes_data = picture.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        with st.spinner("Analizando la imagen..."):
            results = model(cv2_img)

        predictions = results.pred[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        categories = predictions[:, 5]

        label_names = model.names

        col1, col2 = st.columns(2)
        with col1:
            results.render()
            st.image(cv2_img, channels='BGR', use_container_width=True)
        
        with col2:
            st.subheader("📋 Objetos detectados y curiosidades")
            detected_objects = []
            for category in categories:
                label = label_names[int(category)]
                if label not in detected_objects:
                    detected_objects.append(label)
                    description = object_descriptions.get(label, "🔹 Objeto detectado, sin información adicional disponible.")
                    st.markdown(f"**{label.capitalize()}** — {description}")
            
            st.divider()

            # Tabla resumen
            category_count = pd.Series([label_names[int(c)] for c in categories]).value_counts()
            df = pd.DataFrame({
                "Categoría": category_count.index,
                "Cantidad": category_count.values
            })
            st.dataframe(df, use_container_width=True)
            st.bar_chart(df.set_index("Categoría"))
else:
    st.error("No se pudo cargar el modelo. Por favor verifica las dependencias e inténtalo nuevamente.")

st.markdown("---")
st.caption("Desarrollado por Diego Peña | Basado en YOLOv5 y Streamlit 🤖")

