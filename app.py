import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import openai

# Inicializar `spacy` y añadir SpacyTextBlob directamente
nlp = spacy.blank("en")  # Crear un pipeline vacío sin necesidad de `en_core_web_sm`
nlp.add_pipe("spacytextblob")

# Configuración de la barra lateral
st.sidebar.header("Configuración")
openai_api_key = st.sidebar.text_input("Introduce tu API de OpenAI", type="password")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])

if openai_api_key and uploaded_file:
    # Configurar la API de OpenAI
    openai.api_key = openai_api_key

    # Cargar los datos
    data = pd.read_csv(uploaded_file)
    
    # Verificar si el archivo tiene columnas antes de intentar seleccionar una
    if not data.empty:
        # Selección de la columna de texto
        text_column = st.sidebar.selectbox("Selecciona la columna de texto para el análisis de sentimiento", ["Seleccione una columna"] + list(data.columns))
        category_column = st.sidebar.selectbox("Selecciona la columna de categoría o producto (opcional)", ["Ninguno"] + list(data.columns))
        date_column = st.sidebar.selectbox("Selecciona la columna de fecha (opcional)", ["Ninguno"] + list(data.columns))
        
        # Añadir botón para iniciar el análisis
        start_analysis = st.sidebar.button("Iniciar análisis de sentimientos")
        
        # Ejecutar el análisis de sentimiento solo si el usuario ha seleccionado una columna de texto y ha hecho clic en el botón
        if start_analysis and text_column != "Seleccione una columna":
            # Clasificar Sentimiento usando SpacyTextBlob, con manejo de valores nulos o no textuales
            def classify_sentiment(text):
                try:
                    if pd.isna(text) or not isinstance(text, str):
                        return "Neutral"  # Clasificar como neutral si no es texto o si es nulo
                    doc = nlp(text)
                    polarity = doc._.blob.polarity
                    if polarity > 0.1:
                        return "Positivo"
                    elif polarity < -0.1:
                        return "Negativo"
                    else:
                        return "Neutral"
                except Exception as e:
                    st.write(f"Error al analizar la fila: {e}")
                    return "Neutral"

            st.info("Clasificando el sentimiento de cada fila...")
            try:
                data["Sentimiento"] = data[text_column].apply(classify_sentiment)
            except Exception as e:
                st.error(f"Error al aplicar el análisis de sentimientos: {e}")
            
            # Verificar si la columna "Sentimiento" fue creada correctamente
            if "Sentimiento" in data.columns:
                # Mostrar gráficos (similar al código anterior)

                # Explicación con OpenAI (una sola llamada)
                if st.button("Generar Explicación de los Gráficos"):
                    explanation_prompt = """Explica brevemente los siguientes gráficos de análisis de sentimiento:
                    1. Distribución de Sentimientos - Muestra los porcentajes de cada tipo de sentimiento (Positivo, Negativo, Neutral) en el dataset.
                    2. Análisis de Productos por Sentimiento - Muestra los sentimientos específicos de cada producto o categoría.
                    3. Sentimiento Promedio por Categoría o Producto - Asigna puntajes a cada sentimiento y muestra el promedio por categoría o producto.
                    4. Tendencias de Sentimiento por Fecha - Visualiza cómo cambian los sentimientos a lo largo del tiempo en el dataset.
                    """
                    try:
                        response = openai.Completion.create(
                            engine="text-davinci-003",
                            prompt=explanation_prompt,
                            max_tokens=150
                        )
                        explanation = response.choices[0].text.strip()
                        st.write("### Explicación de los Gráficos:")
                        st.write(explanation)
                    except openai.error.OpenAIError as e:
                        st.error(f"Error en la llamada a la API de OpenAI: {e}")
                    except Exception as e:
                        st.error(f"Ocurrió un error inesperado: {e}")
else:
    st.info("Introduce la API y sube un archivo para comenzar.")
