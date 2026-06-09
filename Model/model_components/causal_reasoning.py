import torch
import torch.nn as nn

class CausalReasoningModule(nn.Module):
    """
    Sistema 2: Módulo de razonamiento causal para el Robotaxi.
    Genera una representación latente de la justificación de la maniobra
    y (opcionalmente) una secuencia de tokens de texto.
    """
    def __init__(self, input_dim=1299, hidden_dim=512, vocab_size=1000):
        super(CausalReasoningModule, self).__init__()
        
        # Encoder de razonamiento
        self.reasoning_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Head de clasificación de decisiones (Decision Grounding)
        # 0: Cruce, 1: Peatón, 2: Semáforo, 3: Obstáculo, 4: Libre
        self.decision_grounding = nn.Linear(hidden_dim, 5)
        
        # Head de texto simplificado (proyecta a espacio de vocabulario)
        # En una versión real, esto alimentaría un decodificador Transformer.
        self.text_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, feature_vector):
        # Generar embedding de razonamiento
        reasoning_latent = self.reasoning_encoder(feature_vector)
        
        # Clasificar la causa principal (Grounding)
        decision_logits = self.decision_grounding(reasoning_latent)
        
        # Generar "pensamiento" latente
        text_logits = self.text_head(reasoning_latent)
        
        return reasoning_latent, decision_logits, text_logits

def calculate_causal_consistency_reward(decision_logits, predicted_trajectory):
    """
    R_consistencia: Penaliza si el razonamiento no coincide con la acción física.
    Ejemplo: Si la causa es 'Peatón' (Grounding), la aceleración debe ser baja/negativa.
    """
    decision = torch.argmax(decision_logits, dim=-1)
    accel = predicted_trajectory.view(-1, 64, 2)[:, :, 0] # Tomar aceleración
    mean_accel = torch.mean(accel, dim=1)
    
    reward = torch.zeros_like(mean_accel)
    
    # Regla: Si hay obstáculo/peatón (decisiones 1, 2, 3), penalizar aceleración positiva alta
    mask_hazard = (decision == 1) | (decision == 2) | (decision == 3)
    reward[mask_hazard] = -torch.clamp(mean_accel[mask_hazard], min=0.0)
    
    return reward
