import os
import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN")

def download_spain_subset():
    if not token:
        print("Error: HF_TOKEN no configurado.")
        return

    repo_id = "nvidia/PhysicalAI-Autonomous-Vehicles"
    
    try:
        print("1. Descargando metadatos de recolección...")
        meta_path = hf_hub_download(repo_id=repo_id, filename="metadata/data_collection.parquet", repo_type="dataset", token=token)
        df_meta = pd.read_parquet(meta_path)
        
        # Filtrar clips de España
        spain_meta = df_meta[df_meta['country'].str.contains('Spain|España', case=False, na=False)]
        
        if spain_meta.empty:
            print("No se encontraron clips en España en los metadatos.")
            return

        print(f"Encontrados {len(spain_meta)} clips en España.")
        
        # Obtener los clip_ids (el índice del dataframe suele ser el clip_id o tiene una columna)
        # Si el índice es el clip_id, lo usamos. Si no, buscamos la columna.
        spain_clip_ids = spain_meta.index.tolist()
        target_clip = spain_clip_ids[0]
        
        print(f"Descargando datos para el clip: {target_clip}...")
        
        # Descargar el clip específico
        # La estructura suele ser clips/{clip_id}
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"clips/{target_clip}/*"],
            repo_type="dataset",
            token=token,
            max_workers=4
        )
        print(f"Descarga de clip {target_clip} completada.")
        
        # Guardar lista de clips de España para referencia futura
        with open("spain_clips.txt", "w") as f:
            for cid in spain_clip_ids:
                f.write(f"{cid}\n")
        print("Lista de clips de España guardada en spain_clips.txt")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    download_spain_subset()
