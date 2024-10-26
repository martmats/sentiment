import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI

# Título y descripción
st.title("Análisis de Sentimiento")
st.write(
    "Esta herramienta permite analizar el sentimiento de los datos de un archivo CSV. "
    "Para usar esta app, necesitas proporcionar una clave API de OpenAI"
)

# Entrada para la clave API de OpenAI
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Por favor añade tu clave API de OpenAI para continuar.", icon="🗝️")
else:
    # Crear cliente de OpenAI
    client = OpenAI(api_key=openai_api_key)

    # Cargar archivo CSV
    uploaded_file = st.file_uploader("Sube un archivo CSV para análisis de sentimiento", type="csv")
    if uploaded_file:
        # Leer archivo CSV en un DataFrame
        data = pd.read_csv(uploaded_file)
        
        # Mostrar datos cargados
        st.write("Datos cargados:")
        st.dataframe(data)

        # Seleccionar columna de texto para análisis
        text_column = st.selectbox("Selecciona la columna de texto para analizar el sentimiento", data.columns)

        # Botón para realizar análisis de sentimiento
        if st.button("Analizar Sentimiento"):
            sentiment_results = []
            for text in data[text_column].dropna():
                try:
                    # Generar respuesta usando la API de OpenAI
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": f"Analiza el sentimiento de este texto: {text}"}]
                    )
                    
                    # Comprobar la respuesta y extraer el contenido
                    if response.choices and 'content' in response.choices[0].message:
                        sentiment = response.choices[0].message['content']
                    else:
                        sentiment = "Indefinido"
                    
                except Exception as e:
                    sentiment = f"Error: {str(e)}"
                
                sentiment_results.append(sentiment)

            # Añadir resultados al DataFrame
            data["Sentimiento"] = sentiment_results
            
            # Gráfico de barras para distribución de sentimientos
            st.subheader("Distribución de Sentimientos")
            sentiment_counts = data["Sentimiento"].value_counts()
            fig, ax = plt.subplots()
            sentiment_counts.plot(kind="bar", ax=ax, color=['green', 'blue', 'red'])
            ax.set_ylabel("Cantidad")
            ax.set_xlabel("Tipo de Sentimiento")
            st.pyplot(fig)

            # Gráfico de barras agrupadas por producto o categoría, si existe columna "Producto"
            if "Producto" in data.columns:
                st.subheader("Sentimiento por Producto")
                sentiment_by_product = data.groupby(["Producto", "Sentimiento"]).size().unstack()
                fig, ax = plt.subplots()
                sentiment_by_product.plot(kind="bar", stacked=True, ax=ax)
                ax.set_ylabel("Cantidad")
                st.pyplot(fig)

            # Gráfico de promedio de sentimiento por categoría
            # Mapea sentimientos a valores numéricos
            data["Sentimiento_Num"] = data["Sentimiento"].map({"Positivo": 1, "Neutral": 0, "Negativo": -1})
            if "Producto" in data.columns:
                avg_sentiment_by_product = data.groupby("Producto")["Sentimiento_Num"].mean()
                st.subheader("Calificación Promedio por Producto")
                fig, ax = plt.subplots()
                avg_sentiment_by_product.plot(kind="bar", ax=ax)
                ax.set_ylabel("Sentimiento Promedio")
                st.pyplot(fig)

    else:
        st.info("Por favor sube un archivo CSV para realizar el análisis.")
