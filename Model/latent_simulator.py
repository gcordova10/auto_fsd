import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentWorldSimulator(nn.Module):
    """
    Simulador ligero que opera en el espacio latente de AutoFSD.
    Utiliza la lógica de AutoSplat para evaluar la seguridad de una trayectoria
    sin necesidad de renderizado 3D completo.
    """
    def __init__(self, latent_channels=1440, grid_size=7):
        super(LatentWorldSimulator, self).__init__()
        self.latent_channels = latent_channels
        self.grid_size = grid_size
        
        # Proyector de trayectoria a espacio latente
        # Mapea (Batch, 64, 2) -> (Batch, 4, latent_channels) 
        # para comparar con los 4 estados de FutureState
        self.traj_to_latent = nn.Sequential(
            nn.Linear(128, 512),
            nn.GELU(),
            nn.Linear(512, 4 * grid_size * grid_size)
        )

    def evaluate_trajectory_safety(self, predicted_future_states, proposed_trajectory):
        """
        Calcula una recompensa de seguridad comparando la trayectoria con la ocupación latente.
        
        Args:
            predicted_future_states: Tupla de 4 tensores (Batch, 1440, 7, 7) de FutureState.
            proposed_trajectory: Tensor (Batch, 128) con 64 pasos de (accel, curv).
            
        Returns:
            safety_reward: Recompensa negativa si hay colisión latente.
        """
        batch_size = proposed_trajectory.shape[0]
        
        # 1. Mapear trayectoria a una "máscara de ocupación planeada" en el grid 7x7
        # Esto simula dónde estará el coche en los 4 intervalos de tiempo (1.6s cada uno)
        traj_mask = self.traj_to_latent(proposed_trajectory)
        traj_mask = traj_mask.view(batch_size, 4, self.grid_size, self.grid_size)
        traj_mask = torch.sigmoid(traj_mask) # Probabilidad de ocupación del ego-coche
        
        collision_penalty = 0
        
        for i, future_state in enumerate(predicted_future_states):
            # 2. Estimar ocupación del entorno desde el embedding latente (Lógica AutoSplat simplificada)
            # Colapsamos canales para obtener un "mapa de densidad de obstáculos"
            # Asumimos que altas activaciones en ciertos canales indican presencia de objetos
            obstacle_density = torch.mean(torch.abs(future_state), dim=1) # (Batch, 7, 7)
            obstacle_density = F.normalize(obstacle_density, dim=(1,2))
            
            # 3. Intersección entre trayectoria planeada y densidad de obstáculos
            # Penalizamos si el ego-coche planea estar donde el World Model predice obstáculos
            intersection = traj_mask[:, i, :, :] * obstacle_density
            collision_penalty += torch.sum(intersection, dim=(1, 2))
            
        return -collision_penalty # Recompensa negativa

if __name__ == "__main__":
    # Test rápido del simulador
    sim = LatentWorldSimulator()
    
    # Mock de 4 estados futuros de JEPA (Batch=1)
    future_states = tuple(torch.randn(1, 1440, 7, 7) for _ in range(4))
    # Trayectoria de prueba
    trajectory = torch.randn(1, 128)
    
    reward = sim.evaluate_trajectory_safety(future_states, trajectory)
    print(f"Safety Reward (Latent): {reward.item():.4f}")
