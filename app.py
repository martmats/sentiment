import streamlit as st
import pandas as pd
from openai import OpenAI

# Show title and description
st.title("Chatbot y Análisis de Sentimiento")
st.write(
    "Este chatbot utiliza el modelo GPT-3.5 de OpenAI para generar respuestas. "
    "También permite analizar el sentimiento de los datos de un archivo CSV. "
    "Para usar esta app, necesitas proporcionar una clave API de OpenAI, que puedes obtener [aquí](https://platform.openai.com/account/api-keys)."
)

# Ask user for their OpenAI API key via `st.text_input`
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Por favor añade tu clave API de OpenAI para continuar.", icon="🔑")
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
                # Generate a response with the OpenAI API
                response = client.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": f"Analiza el sentimiento de este texto: {text}"}]
                )
                # Check if content exists in the response
                if response.choices and response.choices[0].message.content:
                    sentiment = response.choices[0].message.content
                else:
                    sentiment = "Error: No se obtuvo respuesta de la API"
                
                # Add the result to the list of sentiments
                sentiment_results.append(sentiment)
            
            # Add the results to the DataFrame and display
            data["Sentimiento"] = sentiment_results
            st.write("Resultados de Análisis de Sentimiento:")
            st.dataframe(data)
    else:
        st.info("Por favor sube un archivo CSV para realizar el análisis.")
    
    # Chatbot session state for message storage
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input field
    if prompt := st.chat_input("¿Cómo te puedo ayudar?"):
        # Store and display the current prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate a response using the OpenAI API
        response = client.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]
        )
        
        # Extract and display the assistant's response
        assistant_response = response.choices[0].message["content"]
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
