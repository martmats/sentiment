import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import openai
from io import StringIO

# Configuración de la barra lateral
st.sidebar.header("Configuración")
openai_api_key = st.sidebar.text_input("Introduce tu API de OpenAI", type="password")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])
sentiment_column = st.sidebar.text_input("Columna de Sentimiento (Ej: 'sentiment')", value="sentiment")
category_column = st.sidebar.text_input("Columna de Categoría o Producto (Opcional)")
date_column = st.sidebar.text_input("Columna de Fecha (Opcional)")

if openai_api_key and uploaded_file and sentiment_column:
    # Cargar los datos
    data = pd.read_csv(uploaded_file)
    
    # Gráfico 1: Distribución de Sentimientos
    st.header("Distribución de Sentimientos")
    sentiment_counts = data[sentiment_column].value_counts()
    fig, ax = plt.subplots()
    sentiment_counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Sentimiento")
    ax.set_ylabel("Conteo")
    st.pyplot(fig)
    
    # Pie Chart
    fig, ax = plt.subplots()
    sentiment_counts.plot(kind="pie", autopct='%1.1f%%', ax=ax)
    st.pyplot(fig)

    # Gráfico 2: Análisis de Productos por Sentimiento
    if category_column in data.columns:
        st.header("Análisis de Productos por Sentimiento")
        product_sentiment_counts = data.groupby([category_column, sentiment_column]).size().unstack(fill_value=0)
        fig, ax = plt.subplots()
        product_sentiment_counts.plot(kind="bar", stacked=False, ax=ax)
        ax.set_xlabel(category_column)
        ax.set_ylabel("Conteo de Sentimiento")
        st.pyplot(fig)

    # Gráfico 3: Sentimiento Promedio por Categoría o Producto
    st.header("Sentimiento Promedio por Categoría o Producto")
    sentiment_mapping = {"Positivo": 1, "Neutral": 0, "Negativo": -1}
    data["sentiment_score"] = data[sentiment_column].map(sentiment_mapping)
    if category_column in data.columns:
        avg_sentiment = data.groupby(category_column)["sentiment_score"].mean()
        fig, ax = plt.subplots()
        avg_sentiment.plot(kind="line", ax=ax)
        ax.set_xlabel(category_column)
        ax.set_ylabel("Puntaje Promedio de Sentimiento")
        st.pyplot(fig)

    # Gráfico 4: Tendencias de Sentimiento por Fecha
    if date_column in data.columns:
        st.header("Tendencias de Sentimiento por Fecha")
        data[date_column] = pd.to_datetime(data[date_column])
        sentiment_trend = data.groupby([date_column, sentiment_column]).size().unstack(fill_value=0)
        fig, ax = plt.subplots()
        sentiment_trend.plot(kind="line", ax=ax)
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Conteo de Sentimiento")
        st.pyplot(fig)

    # Explicación con OpenAI
    if st.button("Generar Explicación de los Gráficos"):
        openai.api_key = openai_api_key
        explanation_prompt = "Explica los siguientes gráficos de análisis de sentimiento: \n1. Distribución de Sentimientos \n2. Análisis de Productos por Sentimiento \n3. Sentimiento Promedio por Categoría o Producto \n4. Tendencias de Sentimiento por Fecha."
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=explanation_prompt,
                max_tokens=150
            )
            explanation = response.choices[0].text.strip()
            st.write("### Explicación de los Gráficos:")
            st.write(explanation)
        except Exception as e:
            st.error("Error al generar la explicación: " + str(e))
else:
    st.info("Introduce la API y sube un archivo para comenzar.")

