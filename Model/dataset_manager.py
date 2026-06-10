import os
import pandas as pd
from huggingface_hub import hf_hub_download
from .config import Config

class DatasetManager:
    def __init__(self):
        self.repo_id = Config.REPO_ID
        self.token = Config.HF_TOKEN
        self.snapshot_path = Config.SNAPSHOT_PATH

    def download_metadata(self):
        print("Descargando metadatos del dataset...")
        hf_hub_download(
            repo_id=self.repo_id,
            filename="metadata/data_collection.parquet",
            repo_type="dataset",
            token=self.token,
            local_dir=self.snapshot_path
        )
        hf_hub_download(
            repo_id=self.repo_id,
            filename="clip_index.parquet",
            repo_type="dataset",
            token=self.token,
            local_dir=self.snapshot_path
        )

    def get_clips_by_country(self, country):
        meta_path = os.path.join(self.snapshot_path, "metadata/data_collection.parquet")
        if not os.path.exists(meta_path):
            self.download_metadata()
            
        df = pd.read_parquet(meta_path)
        # Búsqueda flexible (Spain | España)
        country_clips = df[df['country'].str.contains(country, case=False, na=False)]
        return country_clips.index.tolist()

    def download_country_chunks(self, country, limit_chunks=1):
        clip_ids = self.get_clips_by_country(country)
        index_path = os.path.join(self.snapshot_path, "clip_index.parquet")
        df_index = pd.read_parquet(index_path)
        
        # Encontrar los chunks que contienen esos clips
        relevant_chunks = df_index[df_index.index.isin(clip_ids)]['chunk'].unique()
        
        print(f"Encontrados {len(clip_ids)} clips en {country} repartidos en {len(relevant_chunks)} chunks.")
        
        for i, chunk in enumerate(relevant_chunks[:limit_chunks]):
            chunk_str = f"{chunk:04d}"
            print(f"[{i+1}/{limit_chunks}] Descargando chunk {chunk_str} para {country}...")
            
            # Descargar Cámara Frontal Wide y Egomotion
            files = [
                f"camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_{chunk_str}.zip",
                f"labels/egomotion/egomotion.chunk_{chunk_str}.zip"
            ]
            
            for f in files:
                hf_hub_download(
                    repo_id=self.repo_id,
                    filename=f,
                    repo_type="dataset",
                    token=self.token,
                    local_dir=self.snapshot_path
                )
        
        # Guardar lista de clips localmente
        save_path = f"{country.lower()}_clips.txt"
        with open(save_path, "w") as f:
            for cid in clip_ids:
                f.write(cid + "\n")
        print(f"Lista de clips guardada en {save_path}")

if __name__ == "__main__":
    manager = DatasetManager()
    # Ejemplo: Descargar 1 chunk de España (configurado en .env o por defecto)
    manager.download_country_chunks(Config.TARGET_COUNTRY, limit_chunks=1)
