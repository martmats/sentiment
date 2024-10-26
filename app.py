import streamlit as st
import pandas as pd
import plotly.express as px
import openai
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Análisis de Sentimiento con API de OpenAI", layout="wide")

# Título de la aplicación
st.title("Análisis de Sentimiento de Clientes con OpenAI")

# Ingreso de la API Key de OpenAI
st.sidebar.header("Configuración de la API de OpenAI")
openai_api_key = st.sidebar.text_input("Introduce tu API Key de OpenAI", type="password")

# Verificar si la API Key está ingresada
if openai_api_key:
    st.sidebar.success("API Key ingresada correctamente.")
    openai.api_key = openai_api_key

    # Cargar archivo CSV
    st.sidebar.header("Carga tu archivo de datos")
    uploaded_file = st.sidebar.file_uploader("Sube un archivo CSV", type=["csv"])

    # Agregar opción para seleccionar delimitador
    delimiter = st.sidebar.selectbox("Selecciona el delimitador utilizado en tu archivo CSV", [",", ";", "\t", "|"])

    if uploaded_file:
        # Intentar cargar el archivo CSV con control de errores y delimitador seleccionado
        try:
            data = pd.read_csv(uploaded_file, delimiter=delimiter, on_bad_lines='skip', encoding='utf-8')
        except Exception as e:
            st.error(f"Error al cargar el CSV: {e}")
            data = None

        if data is not None:
            st.subheader("Datos cargados (mostrando las primeras filas)")
            st.write(data.head())

            # Seleccionar la columna de texto para el análisis de sentimiento
            st.sidebar.header("Selecciona la columna de texto")
            text_column = st.sidebar.selectbox("Columna de texto para analizar", data.columns)

            if text_column:
                st.subheader(f"Análisis de sentimiento basado en la columna: {text_column}")

                # Procesar cada texto y hacer una solicitud de análisis de sentimiento con OpenAI
                sentiment_results = []
                for text in data[text_column].dropna():
                    try:
                        response = openai.Completion.create(
                            model="gpt-3.5-turbo",
                            prompt=f"Analiza el sentimiento de este texto y clasifícalo en positivo, neutral o negativo: '{text}'",
                            max_tokens=10
                        )
                        sentiment = response.choices[0].text.strip()
                        sentiment_results.append(sentiment)
                    except Exception as e:
                        sentiment_results.append("Error")
                        st.error(f"Error en la solicitud a la API de OpenAI: {e}")

                # Agregar los resultados al DataFrame
                data['Sentimiento'] = sentiment_results

                # Mostrar los resultados
                st.subheader("Resultados del Análisis de Sentimiento")
                st.write(data[[text_column, 'Sentimiento']].head())

                # Visualización de los resultados
                st.subheader("Distribución de Sentimientos")
                sentiment_count = data['Sentimiento'].value_counts()
                fig = px.pie(sentiment_count, names=sentiment_count.index, values=sentiment_count.values, 
                             title="Distribución de Sentimientos")
                st.plotly_chart(fig)

                st.subheader("Sentimiento por Filtros Adicionales")
                # Seleccionar columnas adicionales para análisis de segmentación
                filter_columns = st.sidebar.multiselect("Selecciona columnas adicionales para segmentar el sentimiento", 
                                                        [col for col in data.columns if col != text_column and col != 'Sentimiento'])

                if filter_columns:
                    for col in filter_columns:
                        fig = px.histogram(data, x=col, color='Sentimiento', barmode='group', 
                                           title=f"Distribución de Sentimiento por {col}")
                        st.plotly_chart(fig)

else:
    st.warning("Por favor, ingresa tu API Key de OpenAI para comenzar.")
S
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})

