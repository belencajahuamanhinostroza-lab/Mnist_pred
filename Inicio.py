import os

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Detector de Dígitos MNIST",
    page_icon="✍️",
    layout="wide"
)


# ============================================================
# ESTILO
# ============================================================

st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
    }

    .digit-result {
        font-size: 80px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Información")

    st.write(f"**TensorFlow:** {tf.__version__}")

    st.markdown("---")

    st.write("### 📁 Modelo")

    if os.path.exists("mnist_model.h5"):
        st.success("✅ mnist_model.h5 encontrado")
    else:
        st.error("❌ mnist_model.h5 no encontrado")

    st.markdown("---")

    st.write("### 📝 Cómo usar")

    st.write("""
    1. Dibuja un número en el recuadro.
    2. Presiona **Predecir**.
    3. El modelo analizará tu dibujo.
    4. Observa el resultado y las probabilidades.
    """)


# ============================================================
# TÍTULO
# ============================================================

st.markdown(
    '<div class="title">✍️ Detector de Dígitos MNIST</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Modelo CNN entrenado con el conjunto de datos MNIST</div>',
    unsafe_allow_html=True
)


# ============================================================
# CARGAR MODELO
# ============================================================

@st.cache_resource
def load_mnist_model():

    model_path = "mnist_model.h5"

    if not os.path.exists(model_path):
        st.error("❌ No se encontró el archivo `mnist_model.h5`.")

        st.info("""
        **Verifica tu repositorio de GitHub.**

        La estructura debe ser:

        ```
        Detector-MNIST/
        ├── app.py
        ├── requirements.txt
        └── mnist_model.h5
        ```
        """)

        st.stop()

    try:
        model = keras.models.load_model(model_path)

        return model

    except Exception as e:

        st.error("❌ No se pudo cargar el modelo `mnist_model.h5`.")

        st.code(str(e))

        st.info("""
        Es posible que el modelo haya sido creado con una versión
        diferente de TensorFlow/Keras.
        """)

        st.stop()


# Cargar modelo
model = load_mnist_model()


# ============================================================
# MENSAJE DE MODELO CARGADO
# ============================================================

st.success("✅ Modelo MNIST cargado correctamente")


# ============================================================
# ÁREA PRINCIPAL
# ============================================================

st.write("## 🎨 Dibuja un dígito")

st.write(
    "Escribe un número del **0 al 9** usando el mouse o el dedo."
)


# ============================================================
# COLUMNAS
# ============================================================

col1, col2 = st.columns([2, 1])


# ============================================================
# CANVAS
# ============================================================

with col1:

    st.write("### 🖊️ Área de dibujo")

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        width=400,
        height=400,
        drawing_mode="freedraw",
        key="canvas"
    )


# ============================================================
# CONTROLES
# ============================================================

with col2:

    st.write("### 🎮 Controles")

    predict_btn = st.button(
        "🔍 Predecir",
        type="primary",
        use_container_width=True
    )

    clear_btn = st.button(
        "🗑️ Limpiar",
        use_container_width=True
    )

    if clear_btn:
        st.rerun()


# ============================================================
# PREDICCIÓN
# ============================================================

if predict_btn:

    # --------------------------------------------------------
    # Verificar que exista imagen
    # --------------------------------------------------------

    if canvas_result.image_data is None:

        st.warning(
            "⚠️ Primero debes dibujar un dígito."
        )

    else:

        # ----------------------------------------------------
        # Obtener imagen del canvas
        # ----------------------------------------------------

        canvas_image = canvas_result.image_data

        # Comprobar si está vacío
        if np.max(canvas_image) == 0:

            st.warning(
                "⚠️ El canvas está vacío. Dibuja un dígito primero."
            )

        else:

            with st.spinner("🤖 Analizando tu dibujo..."):

                try:

                    # ----------------------------------------
                    # Convertir a imagen PIL
                    # ----------------------------------------

                    image = Image.fromarray(
                        canvas_image.astype("uint8")
                    )

                    # Convertir a escala de grises
                    image = image.convert("L")

                    # ----------------------------------------
                    # Redimensionar a 28x28
                    # ----------------------------------------

                    image_resized = image.resize(
                        (28, 28),
                        Image.Resampling.LANCZOS
                    )

                    # ----------------------------------------
                    # Convertir a numpy
                    # ----------------------------------------

                    img_array = np.array(
                        image_resized,
                        dtype=np.float32
                    )

                    # Normalizar
                    img_array = img_array / 255.0

                    # ----------------------------------------
                    # Adaptar al formato del modelo
                    # ----------------------------------------

                    img_array = img_array.reshape(
                        1,
                        28,
                        28,
                        1
                    )

                    # ----------------------------------------
                    # Predicción
                    # ----------------------------------------

                    prediction = model.predict(
                        img_array,
                        verbose=0
                    )

                    # ----------------------------------------
                    # Obtener resultado
                    # ----------------------------------------

                    digit = int(
                        np.argmax(prediction[0])
                    )

                    confidence = float(
                        np.max(prediction[0]) * 100
                    )

                    # Segunda opción
                    sorted_indices = np.argsort(
                        prediction[0]
                    )

                    alternative = int(
                        sorted_indices[-2]
                    )

                    alternative_confidence = float(
                        prediction[0][alternative] * 100
                    )

                except Exception as e:

                    st.error(
                        "❌ Ocurrió un error durante la predicción."
                    )

                    st.code(str(e))

                    st.stop()


            # =================================================
            # RESULTADO
            # =================================================

            st.markdown("---")

            st.write("## 🎯 Resultado")


            st.success(
                f"### El dígito detectado es: **{digit}**"
            )


            # Mostrar dígito grande

            st.markdown(
                f"""
                <div class="result-box">
                    <div class="digit-result">{digit}</div>
                    <div>🎯 Dígito detectado</div>
                </div>
                """,
                unsafe_allow_html=True
            )


            # =================================================
            # MÉTRICAS
            # =================================================

            st.write("### 📊 Información de la predicción")

            col_m1, col_m2, col_m3 = st.columns(3)

            with col_m1:

                st.metric(
                    "Predicción",
                    digit
                )

            with col_m2:

                st.metric(
                    "Confianza",
                    f"{confidence:.1f}%"
                )

            with col_m3:

                st.metric(
                    "2ª opción",
                    f"{alternative} ({alternative_confidence:.1f}%)"
                )


            # =================================================
            # IMÁGENES
            # =================================================

            st.write("### 🖼️ Procesamiento de la imagen")

            col_img1, col_img2 = st.columns(2)

            with col_img1:

                st.write("**Dibujo original:**")

                st.image(
                    canvas_image,
                    width=250
                )

            with col_img2:

                st.write("**Imagen procesada (28 × 28):**")

                st.image(
                    image_resized,
                    width=250
                )


            # =================================================
            # PROBABILIDADES
            # =================================================

            st.write("### 📊 Probabilidades por dígito")

            probabilities = prediction[0] * 100

            prob_df = pd.DataFrame({
                "Dígito": [
                    str(i)
                    for i in range(10)
                ],
                "Probabilidad (%)": probabilities
            })

            st.bar_chart(
                prob_df.set_index("Dígito")
            )


            # =================================================
            # DETALLES
            # =================================================

            with st.expander("🔍 Ver detalles de todas las probabilidades"):

                for i, prob in enumerate(prediction[0]):

                    percentage = prob * 100

                    if i == digit:

                        st.write(
                            f"🎯 **Dígito {i}: {percentage:.2f}%**"
                        )

                    else:

                        st.write(
                            f"Dígito {i}: {percentage:.2f}%"
                        )


# ============================================================
# PIE DE PÁGINA
# ============================================================

st.markdown("---")

st.caption(
    "🤖 Detector de Dígitos MNIST • TensorFlow + Keras + Streamlit"
)
