import os
from datasets import load_dataset
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()
token = os.getenv("HF_TOKEN")

def download_spain_data(limit_gb=10):
    if not token:
        print("Error: No se encontró HF_TOKEN en el archivo .env")
        return

    print("Conectando con HuggingFace Hub para descargar el subset de España...")
    
    try:
        # Intentamos cargar el dataset. 
        # Nota: El dataset de Nvidia suele estar organizado por 'scene' o 'location'.
        # Usamos streaming=True para no descargar todo de golpe.
        ds = load_dataset(
            "nvidia/PhysicalAI-Autonomous-Vehicles", 
            token=token,
            streaming=True
        )
        
        print("Dataset conectado con éxito. Buscando clips en España...")
        
        count = 0
        for sample in ds['train']:
            # Verificamos si existe metadato de localización
            # El esquema exacto puede variar, intentamos buscar 'location' o 'country'
            location = sample.get('meta', {}).get('location', '').lower()
            
            if 'spain' in location or 'españa' in location:
                print(f"Encontrado clip en España: {sample.get('clip_id', 'N/A')}")
                # Aquí lógica para guardar el chunk localmente si se desea
                count += 1
            
            if count >= 5: # Limitamos la búsqueda inicial para esta prueba
                break
        
        if count == 0:
            print("No se encontraron clips etiquetados como 'Spain' en los primeros registros del stream.")
            print("Es posible que necesitemos filtrar por coordenadas o metadatos específicos.")
            
    except Exception as e:
        print(f"Ocurrió un error durante la descarga/conexión: {e}")

if __name__ == "__main__":
    # Instalamos python-dotenv si no está para manejar el .env
    download_spain_data()
