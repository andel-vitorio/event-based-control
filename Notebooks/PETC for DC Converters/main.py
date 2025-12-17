import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import convolve2d


def initial_state(N, M):
  """Gera uma condição inicial aleatória (ruído branco)."""
  return np.random.choice([0, 1], size=(N, M), p=[0.8, 0.2])


def update(grid):
  """
  Calcula o próximo estado do autômato usando convolução 2D.

  Lógica:
  A convolução com um kernel de uns (excluindo o centro) conta
  os vizinhos ativos para cada célula simultaneamente.
  """

  # Kernel da Vizinhança de Moore
  # [[1, 1, 1],
  #  [1, 0, 1],
  #  [1, 1, 1]]
  kernel = np.ones((3, 3), dtype=int)
  kernel[1, 1] = 0

  # boundary='wrap' cria uma superfície toroidal (condições de contorno periódicas)
  # mode='same' mantém as dimensões da matriz original
  neighbor_count = convolve2d(grid, kernel, mode='same', boundary='wrap')

  # Aplicação das regras lógicas (B3/S23)
  # Regra 1: Célula viva sobrevive se tiver 2 ou 3 vizinhos
  survival_mask = (grid == 1) & (
      (neighbor_count == 2) | (neighbor_count == 3))

  # Regra 2: Célula morta nasce se tiver exatamente 3 vizinhos
  birth_mask = (grid == 0) & (neighbor_count == 3)

  # O novo estado é a união lógica das máscaras (resultado booleano convertido para int)
  return (survival_mask | birth_mask).astype(int)


# --- Configuração da Simulação ---
N, M = 100, 100
grid = initial_state(N, M)

fig, ax = plt.subplots()
mat = ax.matshow(grid, cmap='binary')
plt.axis('off')


def animate(i):
  global grid
  grid = update(grid)
  mat.set_data(grid)
  return [mat]


# Intervalo ajustado para visualização suave
ani = animation.FuncAnimation(fig, animate, interval=50, blit=True)
plt.show()
