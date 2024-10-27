import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import openai

# Inicializar `spacy` y añadir SpacyTextBlob directamente
nlp = spacy.blank("en")  # Crear un pipeline vacío sin necesidad de `en_core_web_sm`
nlp.add_pipe("spacytextblob")

# Ingreso de la API Key de OpenAI
st.sidebar.header("Configuración de la API de OpenAI")
openai_api_key = st.sidebar.text_input("Introduce tu API Key de OpenAI", type="password")
openai.api_key = openai_api_key

# Cargar archivo CSV
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])
if openai_api_key and uploaded_file:
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
                # Gráfico 1: Distribución de Sentimientos
                st.header("Distribución de Sentimientos")
                sentiment_counts = data["Sentimiento"].value_counts()
                fig, ax = plt.subplots()
                sentiment_counts.plot(kind="bar", ax=ax)
                ax.set_xlabel("Sentimiento")
                ax.set_ylabel("Conteo")
                st.pyplot(fig)
                
                # Gráfico de pastel
                fig, ax = plt.subplots()
                sentiment_counts.plot(kind="pie", autopct='%1.1f%%', ax=ax)
                st.pyplot(fig)

                # Gráfico 2: Análisis de Productos por Sentimiento
                if category_column != "Ninguno":
                    st.header("Análisis de Productos por Sentimiento")
                    product_sentiment_counts = data.groupby([category_column, "Sentimiento"]).size().unstack(fill_value=0)
                    fig, ax = plt.subplots()
                    product_sentiment_counts.plot(kind="bar", stacked=False, ax=ax)
                    ax.set_xlabel(category_column)
                    ax.set_ylabel("Conteo de Sentimiento")
                    st.pyplot(fig)

                # Gráfico 3: Sentimiento Promedio por Categoría o Producto
                st.header("Sentimiento Promedio por Categoría o Producto")
                sentiment_mapping = {"Positivo": 1, "Neutral": 0, "Negativo": -1}
                data["sentiment_score"] = data["Sentimiento"].map(sentiment_mapping)
                if category_column != "Ninguno":
                    avg_sentiment = data.groupby(category_column)["sentiment_score"].mean()
                    fig, ax = plt.subplots()
                    avg_sentiment.plot(kind="line", ax=ax)
                    ax.set_xlabel(category_column)
                    ax.set_ylabel("Puntaje Promedio de Sentimiento")
                    st.pyplot(fig)

                # Gráfico 4: Tendencias de Sentimiento por Fecha
                if date_column != "Ninguno":
                    st.header("Tendencias de Sentimiento por Fecha")
                    data[date_column] = pd.to_datetime(data[date_column])
                    sentiment_trend = data.groupby([date_column, "Sentimiento"]).size().unstack(fill_value=0)
                    fig, ax = plt.subplots()
                    sentiment_trend.plot(kind="line", ax=ax)
                    ax.set_xlabel("Fecha")
                    ax.set_ylabel("Conteo de Sentimiento")
                    st.pyplot(fig)

                # Generar explicación automáticamente después de los gráficos
                explanation_prompt = """Explica con detalle los resultados obtenidos los siguientes gráficos de análisis de sentimiento y posibles recomendaciones al respecto:
                1. Distribución de Sentimientos - Muestra los porcentajes de cada tipo de sentimiento (Positivo, Negativo, Neutral) en el dataset.
                2. Análisis de Productos por Sentimiento - Muestra los sentimientos específicos de cada producto o categoría.
                3. Sentimiento Promedio por Categoría o Producto - Asigna puntajes a cada sentimiento y muestra el promedio por categoría o producto.
                4. Tendencias de Sentimiento por Fecha - Visualiza cómo cambian los sentimientos a lo largo del tiempo en el dataset.
                """
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "Eres un experto en análisis de datos y estrategias de marketing."},
                            {"role": "user", "content": explanation_prompt}
                        ],
                        max_tokens=1250
                    )
                    explanation = response.choices[0].message['content'].strip()
                    st.write("### Explicación de los Gráficos:")
                    st.write(explanation)
                except Exception as e:
                    st.error(f"Error en la solicitud a la API de OpenAI: {e}")
else:
    st.info("Introduce tu API Key de OpenAI y sube un archivo para continuar.")
