# Simulador Ligero Latente (AutoFSD-LatentSim)

Este módulo implementa un motor de simulación de "Mundo Latente" diseñado para el entrenamiento por refuerzo (GRPO) del modelo AutoFSD, evitando el cuello de botella de los simuladores 3D tradicionales (como AlpaSim).

## 🚀 Concepto: Simulación en Espacio de Características

A diferencia de los simuladores que renderizan píxeles, **AutoFSD-LatentSim** opera directamente sobre los embeddings de 1440 canales generados por el Feature Fusion. 

### Inspiración en AutoSplat
El simulador adopta dos ideas clave de **AutoSplat**:
1.  **Densidad de Ocupación Latente:** Las activaciones neuronales en el mapa de características se interpretan como "probabilidad de ocupación física". 
2.  **Bypass de Renderizado:** El cálculo de colisiones se realiza mediante la intersección de tensores entre la trayectoria proyectada y los estados futuros predichos por el World Model (**JEPA**).

## 🛠️ Funcionamiento Técnico (`latent_simulator.py`)

1.  **Entrada:**
    *   `predicted_future_states`: 4 estados futuros (Batch, 1440, 7, 7) que cubren un horizonte de 6.4s.
    *   `proposed_trajectory`: Trayectoria de control (Batch, 64 steps, 2 features).
2.  **Proyección de Trayectoria:** Una red MLP mapea los 128 parámetros de la trayectoria a una máscara de ocupación en un grid de 7x7.
3.  **Cálculo de Recompensa $R_{safety}$:**
    *   Se extrae la "Densidad de Obstáculos" colapsando el espacio latente.
    *   Se penaliza la superposición entre la máscara de la trayectoria y los obstáculos predichos.

## 📊 Beneficios
*   **Velocidad:** >200 FPS (In-memory, sin IO de disco ni renderizado GPU costoso).
*   **Integración GRPO:** Proporciona una señal de recompensa inmediata y diferenciable para estabilizar la toma de decisiones.
*   **Independencia:** Elimina la necesidad de convertir archivos a formato `.usd` de NVIDIA NuRec.
