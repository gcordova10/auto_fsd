# Guía de Instalación: AutoFSD Smooth-GRPO

Este documento detalla los pasos para configurar el entorno de desarrollo y entrenamiento.

## 1. Requisitos del Sistema
- **SO:** Linux (Ubuntu 22.04+ recomendado).
- **GPU:** NVIDIA (Arquitectura Ampere+ recomendada, ej: RTX 3060+).
- **Drivers:** NVIDIA Driver 535+.
- **Python:** 3.10.

## 2. Configuración del Entorno Virtual

```bash
# Crear entorno
python3 -m venv venv

# Activar entorno
source venv/bin/activate

# Instalar dependencias
pip install -r auto_fsd/Model/requirements.txt
```

## 3. Variables de Entorno (.env)

Crea un archivo `.env` en la raíz del proyecto con el siguiente contenido:

```env
# HuggingFace Token (con acceso al dataset de Nvidia)
HF_TOKEN=tu_token_aqui

# Configuración de GPU (Ajustar según VRAM)
GROUP_SIZE=1                   # 1 para 6GB, 4 para 12GB, 8 para 24GB
ACTION_CONDITION_CHANNELS=128  # 128 para 6GB, 1440 para 12GB+

# Configuración de Datos
TARGET_COUNTRY=Spain
LIMIT_CLIPS=100
```

## 4. Descarga del Dataset

Para descargar los clips del país configurado:

```bash
python3 -m auto_fsd.Model.dataset_manager
```

## 5. Verificación

Ejecuta la prueba integral para asegurar que todo funciona:

```bash
python3 auto_fsd/Model/final_prototype_test.py
```
