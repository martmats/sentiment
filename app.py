import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import download
from io import StringIO

# Descargar datos de VADER (necesario la primera vez)
download('vader_lexicon')

# Configuración de la barra lateral
st.sidebar.header("Configuración")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])
text_column = st.sidebar.text_input("Columna de Texto para Análisis de Sentimiento", value="texto")
category_column = st.sidebar.text_input("Columna de Categoría o Producto (Opcional)")
date_column = st.sidebar.text_input("Columna de Fecha (Opcional)")

if uploaded_file and text_column:
    # Cargar los datos
    data = pd.read_csv(uploaded_file)
    
    # Inicializar el analizador de sentimientos
    analyzer = SentimentIntensityAnalyzer()

    # Clasificar Sentimiento usando VADER
    def classify_sentiment(text):
        scores = analyzer.polarity_scores(text)
        if scores['compound'] >= 0.05:
            return "Positivo"
        elif scores['compound'] <= -0.05:
            return "Negativo"
        else:
            return "Neutral"

    st.info("Clasificando el sentimiento de cada fila...")
    data["Sentimiento"] = data[text_column].apply(classify_sentiment)
    
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
    if category_column in data.columns:
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
        sentiment_trend = data.groupby([date_column, "Sentimiento"]).size().unstack(fill_value=0)
        fig, ax = plt.subplots()
        sentiment_trend.plot(kind="line", ax=ax)
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Conteo de Sentimiento")
        st.pyplot(fig)
else:
    st.info("Sube un archivo y selecciona la columna de texto para empezar.")
