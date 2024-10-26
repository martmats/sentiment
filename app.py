import streamlit as st
import pandas as pd
from openai import OpenAI

# Show title and description
st.title("Análisis de Sentimiento")
st.write(
    "Esta herramienta permite analizar el sentimiento de los datos de un archivo CSV. "
    "Para usar esta app, necesitas proporcionar una clave API de OpenAI"
)

# Ask user for their OpenAI API key via `st.text_input`
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Por favor añade tu clave API de OpenAI para continuar.", icon="🗝️")
else:
    # Create an OpenAI client
    client = OpenAI(api_key=openai_api_key)

    # File uploader for CSV
    uploaded_file = st.file_uploader("Sube un archivo CSV para análisis de sentimiento", type="csv")
    if uploaded_file:
        # Read the CSV file into a DataFrame
        data = pd.read_csv(uploaded_file)
        
        # Display the data
        st.write("Datos cargados:")
        st.dataframe(data)

        # Choose a column for sentiment analysis
        text_column = st.selectbox("Selecciona la columna de texto para analizar el sentimiento", data.columns)

        # Button to perform sentiment analysis
        if st.button("Analizar Sentimiento"):
            sentiment_results = []
            for text in data[text_column].dropna():
                # Generate a response using the OpenAI API
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": f"Analiza el sentimiento de este texto: {text}"}]
                )
                sentiment = response.choices[0].message['content']
                sentiment_results.append(sentiment)
            
            # Add the results to the DataFrame and display
            data["Sentimiento"] = sentiment_results
            st.write("Resultados de Análisis de Sentimiento:")
            st.dataframe(data)
    else:
        st.info("Por favor sube un archivo CSV para realizar el análisis.")
