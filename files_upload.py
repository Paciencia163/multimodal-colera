import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
import tempfile
import os

class FilesUpload:

    def __init__(self):
        self.file_types = ["png", "jpg", "jpeg"]
        self.image_shape = (130, 130, 3)

    def run(self, max_files=20):
        st.image("osapi2.jpeg", caption="Akin - Osapi Care", use_column_width=True)
        
        # Carregar múltiplos arquivos
        img_files = st.file_uploader(
            "Carregue até 20 arquivos", 
            type=self.file_types, 
            accept_multiple_files=True
        )
        
        if not img_files:
            st.info(f"Carregue arquivos nos formatos: {', '.join(self.file_types)}")
            return None
        
        if len(img_files) > max_files:
            st.error(f"Por favor, carregue no máximo {max_files} arquivos.")
            return None

        # Processar cada arquivo carregado
        images = []
        for img_file in img_files:
            try:
                # Exibir detalhes do arquivo
                st.image(img_file, caption=f"Carregado: {img_file.name}", use_column_width=True)
                st.write({"Nome": img_file.name, "Tipo": img_file.type, "Tamanho": img_file.size})

                # Salvar temporariamente o arquivo e processar a imagem
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(img_file.name)[-1]) as temp:
                    temp.write(img_file.getbuffer())
                    temp_path = temp.name

                # Pré-processar a imagem
                img = image.load_img(
                    temp_path,
                    target_size=self.image_shape,
                    color_mode='rgb'
                )
                img_array = image.img_to_array(img) / 255.0
                images.append(img_array)
            finally:
                # Certificar-se de excluir o arquivo temporário
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        return images
