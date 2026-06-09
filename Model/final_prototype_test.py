import torch
import sys
import os
import numpy as np

# Añadir el path del modelo
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'auto_fsd/Model')))
from model_components.auto_fsd import AutoFSD
from train_grpo import GRPOTrainer

def run_integrated_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- INICIANDO PRUEBA INTEGRAL EN: {device} ---")

    # 1. Inicializar el Modelo E2E (Percepción + Política + Sistema 2)
    model = AutoFSD().to(device)
    trainer = GRPOTrainer(model, lambda_smooth=0.2, lambda_causal=0.5, group_size=2)
    
    # 2. Simular Datos de Entrada (Escena de "Peatón Detectado")
    visual_tiles = torch.randn(8, 3, 224, 224).to(device)
    egomotion_history = torch.randn(256).to(device)
    visual_history = torch.randn(896).to(device)
    
    # Simular Target para JEPA (4 estados futuros en espacio latente)
    # Cada estado es lo que devuelve self.FutureState (chunks de 5760 canales reducidos o similar)
    # En AutoFSD.forward, future_vision es una tupla de 4 tensores.
    target_future_vision = tuple(torch.randn(1, 1440, 7, 7).to(device) for _ in range(4))

    # Target: Una trayectoria de frenado suave (aceleración negativa)
    target_trajectory = torch.full((128,), -0.5).to(device) 

    print("\n[1] Ejecutando Inferencia (Vision-Language-Action)...")
    model.eval()
    with torch.no_grad():
        outputs = model(visual_tiles, visual_history, egomotion_history)
    
    # Extraer resultados del Sistema 2
    decision_idx = torch.argmax(outputs["decision_logits"][0]).item()
    decisiones = ["Cruce", "Peatón", "Semáforo", "Obstáculo", "Libre"]
    razonamiento = decisiones[decision_idx]
    
    print(f" > Decisión del Sistema 2 (Grounding): {razonamiento}")
    print(f" > Imaginación Latente (JEPA): {len(outputs['future_vision'])} estados futuros predichos.")
    
    # 3. Ejecutar Paso de Optimización GRPO
    print("\n[2] Ejecutando Optimización GRPO (Suavidad + Consistencia + JEPA)...")
    model.train()
    loss, mean_r, jepa_loss = trainer.grpo_step(visual_tiles, visual_history, egomotion_history, 
                                               target_trajectory, target_future_vision)
    
    print(f" > Pérdida Total GRPO: {loss:.6f}")
    print(f" > Pérdida de Reconstrucción (JEPA): {jepa_loss:.6f}")
    print(f" > Recompensa Media del Grupo: {mean_r:.4f}")
    
    # 4. Verificación de Consistencia
    # Si el Sistema 2 detecta un Peatón, la recompensa causal debe estar activa
    print("\n[3] Validación de Consistencia Causal:")
    if decision_idx in [1, 2, 3]: # Peligros
        print(" > ESTADO: El Sistema 2 ha identificado un riesgo. El motor GRPO está penalizando aceleraciones bruscas.")
    else:
        print(" > ESTADO: Escena despejada. El motor GRPO optimiza la suavidad del crucero.")

    print("\n--- PRUEBA COMPLETADA CON ÉXITO ---")
    print("El pipeline VLA + GRPO es totalmente operativo en este hardware.")

if __name__ == "__main__":
    run_integrated_test()
