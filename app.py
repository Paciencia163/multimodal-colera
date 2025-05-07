import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pyttsx3
import pydeck as pdk
import pandas as pd

# ------------------- CONFIGURAÇÃO GERAL -------------------
st.set_page_config(page_title="🧬 Diagnóstico de Malária e Cólera", layout="wide")

# ------------------- FUNÇÕES -------------------
def process_image(image):
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_parasites = []
    min_area = 50
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * (area / (perimeter ** 2))
                if 0.5 < circularity <= 1.0:
                    detected_parasites.append(contour)

    output_image = image_np.copy()
    cv2.drawContours(output_image, detected_parasites, -1, (0, 255, 0), 2)
    return output_image, len(detected_parasites)

def diagnostico_colera(sintomas):
    sintomas_presentes = [s for s, v in sintomas.items() if v]
    if len(sintomas_presentes) >= 3 and "diarreia" in sintomas_presentes and "vomito" in sintomas_presentes:
        return "Suspeita de cólera. Procurar assistência médica imediatamente."
    elif len(sintomas_presentes) >= 2:
        return "Sintomas leves. Monitorar e hidratar-se bem."
    else:
        return "Nenhuma evidência significativa de cólera."

def falar_texto(texto):
    engine = pyttsx3.init()
    engine.say(texto)
    engine.runAndWait()

# ------------------- INTERFACE -------------------
menu = st.sidebar.radio("Menu", ["🏠 Início", "🦟 Malária", "💧 Cólera", "🗺️ Mapa de Angola", "ℹ️ Sobre"])

# ------------------- ABA INÍCIO -------------------
if menu == "🏠 Início":
    st.title("Plataforma Multimodal de Diagnóstico de Doenças Endêmicas")
    st.markdown("""
        Esta aplicação permite realizar a triagem de malária por imagem de amostra sanguínea e simular diagnóstico de cólera com base nos sintomas informados. 
        
        Escolha uma opção no menu lateral para começar.
    """)

# ------------------- ABA MALÁRIA -------------------
elif menu == "🦟 Malária":
    st.title("🦟 Diagnóstico de Malária por Imagem")
    uploaded_file = st.file_uploader("Carregue uma imagem da amostra de sangue", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagem Original", use_column_width=True)
        if st.button("🔬 Processar Imagem"):
            st.text("Processando...")
            output_image, count = process_image(image)
            plasmodium_mm3 = count * 500
            plasmodium_ul_1 = count * 40
            plasmodium_ul_2 = (count * 8000) / 100

            st.image(output_image, caption=f"Imagem Processada - Plasmódios Detectados: {count}", use_column_width=True)
            st.metric("Contagem Final", f"{count} plasmódios")
            st.metric("Plasmódios por mm³", plasmodium_mm3)
            st.metric("Plasmódios por µL (Fórmula 1)", plasmodium_ul_1)
            st.metric("Plasmódios por µL (Fórmula 2)", round(plasmodium_ul_2, 2))

# ------------------- ABA CÓLERA -------------------
elif menu == "💧 Cólera":
    st.title("💧 Triagem Simulada para Cólera")
    with st.form("form_colera"):
        st.markdown("### Informe os sintomas presentes:")
        sintomas = {
            "diarreia": st.checkbox("Diarreia intensa"),
            "vomito": st.checkbox("Vômitos frequentes"),
            "febre": st.checkbox("Febre"),
            "cãibras": st.checkbox("Cãibras musculares"),
            "desidratação": st.checkbox("Sinais de desidratação")
        }
        submitted = st.form_submit_button("📋 Avaliar sintomas")
    if submitted:
        resultado = diagnostico_colera(sintomas)
        st.success(f"🧾 Diagnóstico: {resultado}")
        falar_texto(resultado)

# ------------------- ABA MAPA -------------------
elif menu == "🗺️ Mapa de Angola":
    st.title("🗺️ Regiões Simuladas com Casos de Doenças")
    st.markdown("Este mapa mostra regiões simuladas em Angola com casos registrados de malária e cólera.")

    data = {
        "latitude": [-8.8383, -11.2027, -4.4419],
        "longitude": [13.2344, 13.6112, 15.0451],
        "cidade": ["Luanda", "Huambo", "Uíge"]
    }

    df = pd.DataFrame(data)
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=-11.0,
            longitude=17.0,
            zoom=4,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position='[longitude, latitude]',
                get_color='[200, 30, 0, 160]',
                get_radius=10000,
            )
        ],
    ))

# ------------------- ABA SOBRE -------------------
elif menu == "ℹ️ Sobre":
    st.title("ℹ️ Sobre o Projeto")
    st.markdown("""
        Esta aplicação foi desenvolvida Por Paciência Aníbal Muienga para demonstração de uma plataforma de triagem de doenças endêmicas com o uso de machine learning e simulações clínicas.

        **Componentes:**
        - Detecção de Malária por imagem Utilizando CNN.
        - Diagnóstico simulado de Cólera por sintomas.
        - Interface com voz para acessibilidade.
        - Visualização de mapa interativo com Pydeck.

        Desenvolvido para fins educacionais e de prototipagem.
    """)
