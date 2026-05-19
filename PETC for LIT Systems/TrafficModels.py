from typing import List
from typing import Tuple, List, Set, Dict
import scipy.linalg as linalg
from collections import deque
import networkx as ntx
import scipy.sparse as sp
from typing import Tuple, List, Set
from z3 import *
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
from typing import List, Tuple, Set, Dict
import z3
import cvxpy as cp
import numpy as np
import networkx as nx
from scipy.signal import cont2discrete
from z3 import Solver, Real, sat, Q
from fractions import Fraction
from typing import List, Tuple, Set
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed


def _worker_expand_prefix(prefix: Tuple[int, ...], ell: int, K_set: List[int],
                          A_cal_arrays: List[np.ndarray], Q_v_arrays: List[np.ndarray],
                          valid_base_transitions: Set[Tuple[int, int]],
                          nx_dim: int, nu_bar: int):
  r"""
  Expande um prefixo inicial (ex: tamanho 2) até a profundidade \ell + 1
  usando a Poda Topológica e o Z3 isolado na memória deste processo.
  """
  # Helper local para conversão racional segura
  def to_rational(matrix):
    return [[Q(Fraction(val).limit_denominator(10**10).numerator,
               Fraction(val).limit_denominator(10**10).denominator)
             for val in row] for row in matrix]

  # Helper de verificação usando a restrição rápida original
  def check_sequence(sequence: Tuple[int, ...]) -> bool:
    solver = Solver()
    x0 = [Real(f'x0_{i}') for i in range(nx_dim)]
    solver.add(sum([x * x for x in x0]) > Q(1, 1000))
    G_acc = np.eye(nx_dim)

    for m_target in sequence:
      if m_target > nu_bar:
        return False

      for k in range(1, m_target):
        M_k = Q_v_arrays[k]
        M_proj = G_acc.T @ M_k @ G_acc
        M_z3 = to_rational(M_proj)
        quad_form = sum(x0[r] * M_z3[r][c] * x0[c]
                        for r in range(nx_dim) for c in range(nx_dim))
        solver.add(quad_form >= 0)

      if m_target < nu_bar:
        M_target_mat = Q_v_arrays[m_target]
        M_proj_trig = G_acc.T @ M_target_mat @ G_acc
        M_z3_trig = to_rational(M_proj_trig)
        quad_form_trig = sum(x0[r] * M_z3_trig[r][c] * x0[c]
                             for r in range(nx_dim) for c in range(nx_dim))
        solver.add(quad_form_trig < 0)

      G_acc = A_cal_arrays[m_target] @ G_acc

    return solver.check() == sat

  # Motor de Busca em Largura (BFS) local
  valid_sequences = [prefix]
  feasible_nodes = set()
  edges = []

  if len(prefix) > ell + 1:
    return set(), []

  for length in range(len(prefix) + 1, ell + 2):
    next_valid = []
    for seq in valid_sequences:
      for k_next in K_set:
        # PODA TOPOLÓGICA: Se não existe na base, descarta!
        if (seq[-1], k_next) not in valid_base_transitions:
          continue

        candidate = seq + (k_next,)
        if check_sequence(candidate):
          next_valid.append(candidate)

          # Extração de Nós e Arestas ao atingir o tamanho alvo
          if length == ell + 1:
            node_a = seq
            node_b = seq[1:] + (k_next,)
            edges.append((node_a, node_b, seq[0]))
            feasible_nodes.add(node_a)
            feasible_nodes.add(node_b)

    valid_sequences = next_valid
    if not valid_sequences:
      break

  return feasible_nodes, edges

# ============================================================================
# 2. CLASSE PRINCIPAL
# ============================================================================


class TrafficModelBuilder:
  r"""
  Construtor que orquestra a divisão do trabalho para gerar o modelo \ell-completo.
  """

  def __init__(self, A: np.ndarray, B: np.ndarray, K: np.ndarray,
               Xi: np.ndarray, Psi: np.ndarray, h: float, nu_bar_time: float):
    self.nx = A.shape[0]
    self.h = h
    self.nu_bar = int(round(nu_bar_time / h))

    sys_d = cont2discrete(
        (A, B, np.zeros((1, self.nx)), np.zeros((1, B.shape[1]))), h, method='zoh')
    self.Ad, self.Bd = sys_d[0], sys_d[1]

    # Matrizes NumPy puras (Serializáveis via Pickle para o Windows)
    self.A_cal = [None] * (self.nu_bar + 1)
    self.Q_v = [None] * (self.nu_bar + 1)

    I = np.eye(self.nx)
    self.A_cal[0] = I

    for v in range(1, self.nu_bar + 1):
      Ad_pow_v = np.linalg.matrix_power(self.Ad, v)
      S_v = np.zeros_like(self.Ad)
      for p in range(v):
        S_v += np.linalg.matrix_power(self.Ad, p)

      A_cal_v = Ad_pow_v + S_v @ self.Bd @ K
      self.A_cal[v] = A_cal_v

      diff = I - A_cal_v
      self.Q_v[v] = (A_cal_v.T @ Psi @ A_cal_v) - (diff.T @ Xi @ diff)

  def _check_sequence_sequential(self, sequence: Tuple[int, ...]) -> bool:
    """Verificador sequencial rápido para construir o filtro da base na thread principal."""
    def to_rational(matrix):
      return [[Q(Fraction(val).limit_denominator(10**10).numerator,
                 Fraction(val).limit_denominator(10**10).denominator) for val in row] for row in matrix]

    solver = Solver()
    x0 = [Real(f'x0_{i}') for i in range(self.nx)]
    solver.add(sum([x * x for x in x0]) > Q(1, 1000))
    G_acc = np.eye(self.nx)

    for m_target in sequence:
      if m_target > self.nu_bar:
        return False
      for k in range(1, m_target):
        M_proj = G_acc.T @ self.Q_v[k] @ G_acc
        M_z3 = to_rational(M_proj)
        quad_form = sum(x0[r] * M_z3[r][c] * x0[c]
                        for r in range(self.nx) for c in range(self.nx))
        solver.add(quad_form >= 0)

      if m_target < self.nu_bar:
        M_proj_trig = G_acc.T @ self.Q_v[m_target] @ G_acc
        M_z3_trig = to_rational(M_proj_trig)
        quad_form_trig = sum(x0[r] * M_z3_trig[r][c] * x0[c]
                             for r in range(self.nx) for c in range(self.nx))
        solver.add(quad_form_trig < 0)

      G_acc = self.A_cal[m_target] @ G_acc

    return solver.check() == sat

  def build_exact_l_complete_graph(self, ell: int, K_set: List[int], max_workers: int = None, verbose: bool = True) -> nx.DiGraph:
    if verbose:
      print(r"=== [Init] Calculando Filtro Topológico (Main Process) ===")

    layer_1 = []
    for k in K_set:
      if self._check_sequence_sequential((k,)):
        layer_1.append((k,))

    layer_2_prefixes = []
    valid_base_transitions = set()

    for seq in layer_1:
      for k_next in K_set:
        candidate = seq + (k_next,)
        if self._check_sequence_sequential(candidate):
          layer_2_prefixes.append(candidate)
          valid_base_transitions.add((seq[-1], k_next))

    if verbose:
      print(
          f"   Transições base mapeadas: {len(valid_base_transitions)}")

    G = nx.DiGraph()

    # Caso \ell = 1, resolvemos direto na principal
    if ell == 1:
      G.add_nodes_from(layer_1)
      for node_a in layer_1:
        for k_next in K_set:
          if (node_a[-1], k_next) in valid_base_transitions:
            G.add_edge(node_a, (k_next,),
                       weight=node_a[0], label=str(node_a[0]))
      return G

    # Caso \ell >= 2, disparamos o Multiprocessing nativo (Verdadeiro paralelismo)
    if verbose:
      print(
          rf"=== [Multiprocessing] Distribuindo galhos para \ell={ell} ===")

    all_valid_nodes = set()
    all_edges = []

    # Deixe uma CPU livre para o Windows não travar o SO
    if max_workers is None:
      max_workers = max(1, multiprocessing.cpu_count() - 1)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
      futures = []
      # Cada prefixo factível de tamanho 2 é entregue a um processo diferente
      for prefix in layer_2_prefixes:
        futures.append(
            executor.submit(
                _worker_expand_prefix,
                prefix, ell, K_set,
                self.A_cal, self.Q_v, valid_base_transitions,
                self.nx, self.nu_bar
            )
        )

      for future in as_completed(futures):
        try:
          nodes, edges = future.result()
          all_valid_nodes.update(nodes)
          all_edges.extend(edges)
        except Exception as e:
          if verbose:
            print(f" [Worker Error]: {e}")

    G.add_nodes_from(all_valid_nodes)
    for u, v, weight in all_edges:
      G.add_edge(u, v, weight=u[0], label=str(u[0]))

    if verbose:
      print(r"=== Final Report ===")
      print(f"Final Nodes: {G.number_of_nodes()}")
      print(f"Edges:       {G.number_of_edges()}")

    return G


def _worker_hybrid_exact_expand(prefix, ell, K_set, A_cal, Q_v, valid_base, nx_dim, nu_bar):

  def to_rational(matrix, ctx):
    return [[z3.Q(Fraction(val).limit_denominator(10**10).numerator,
                  Fraction(val).limit_denominator(10**10).denominator, ctx=ctx)
             for val in row] for row in matrix]

  def check_sequence_hybrid_exact(sequence):
    # --- ETAPA 1: Filtro SDR (MOSEK) ---
    # Prova rápida de NÃO-existência.
    X = cp.Variable((nx_dim, nx_dim), PSD=True)
    constraints = [cp.trace(X) == 1]
    A_acc_np = np.eye(nx_dim)

    for m in sequence:
      for j in range(1, m):
        constraints.append(
            cp.trace((A_acc_np.T @ Q_v[j] @ A_acc_np) @ X) >= 0)
      if m < nu_bar:
        constraints.append(
            cp.trace((A_acc_np.T @ Q_v[m] @ A_acc_np) @ X) <= -1e-10)
      A_acc_np = A_cal[m] @ A_acc_np

    prob = cp.Problem(cp.Minimize(0), constraints)
    try:
      prob.solve(solver=cp.MOSEK)
    except:
      return False

    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
      return False  # Se a SDR é inviável, a sequência é impossível.

    # --- ETAPA 2: Prova Real Obrigatória (Z3) ---
    # Garante que o modelo seja EXATO como no seu código original.
    ctx = z3.Context()
    solver = z3.Solver(ctx=ctx)
    x0 = [z3.Real(f'x0_{i}', ctx=ctx) for i in range(nx_dim)]
    solver.add(z3.Sum([x*x for x in x0]) > z3.Q(1, 1000, ctx=ctx))

    G_acc = np.eye(nx_dim)
    for m in sequence:
      for j in range(1, m):
        M_z3 = to_rational(G_acc.T @ Q_v[j] @ G_acc, ctx)
        solver.add(z3.Sum([x0[r]*M_z3[r][c]*x0[c]
                   for r in range(nx_dim) for c in range(nx_dim)]) >= 0)
      if m < nu_bar:
        M_trig_z3 = to_rational(G_acc.T @ Q_v[m] @ G_acc, ctx)
        solver.add(z3.Sum([x0[r]*M_trig_z3[r][c]*x0[c]
                   for r in range(nx_dim) for c in range(nx_dim)]) < 0)
      G_acc = A_cal[m] @ G_acc

    return solver.check() == z3.sat

  # Expansão BFS local para encontrar nós de comprimento ell
  current_layer = [prefix]
  for length in range(len(prefix) + 1, ell + 1):
    next_layer = []
    for seq in current_layer:
      for k_next in K_set:
        if (seq[-1], k_next) not in valid_base:
          continue
        candidate = seq + (k_next,)
        if check_sequence_hybrid_exact(candidate):
          next_layer.append(candidate)
    current_layer = next_layer
    if not current_layer:
      break

  feasible_nodes = set(current_layer)
  edges = []

  # Após encontrar os nós, gera as arestas (comprimento ell + 1)
  for node_a in feasible_nodes:
    for k_next in K_set:
      if (node_a[-1], k_next) not in valid_base:
        continue
      candidate_trans = node_a + (k_next,)
      if check_sequence_hybrid_exact(candidate_trans):
        node_b = node_a[1:] + (k_next,)
        # Note: node_b estará obrigatoriamente no set de outros workers ou deste
        edges.append((node_a, node_b, node_a[0]))

  return feasible_nodes, edges

# ============================================================================
# 2. CLASSE PRINCIPAL
# ============================================================================


class TrafficModelBuilderHybrid:
  def __init__(self, A, B, K, Xi, Psi, h, nu_bar_time, sigma=1.0):
    self.nx = A.shape[0]
    self.h = h
    self.nu_bar = int(round(nu_bar_time / h))
    sys_d = cont2discrete(
        (A, B, np.zeros((1, self.nx)), np.zeros((1, B.shape[1]))), h, method='zoh')
    self.Ad, self.Bd = sys_d[0], sys_d[1]
    self.A_cal = [None] * (self.nu_bar + 1)
    self.Q_v = [None] * (self.nu_bar + 1)
    I = np.eye(self.nx)
    self.A_cal[0] = I
    for v in range(1, self.nu_bar + 1):
      Ad_pow = np.linalg.matrix_power(self.Ad, v)
      S_v = sum(np.linalg.matrix_power(self.Ad, p) for p in range(v))
      self.A_cal[v] = Ad_pow + S_v @ self.Bd @ K
      diff = I - self.A_cal[v]
      self.Q_v[v] = (sigma * self.A_cal[v].T @ Psi @
                     self.A_cal[v]) - (diff.T @ Xi @ diff)

  def _check_z3_only(self, sequence):
    # Usado apenas para o filtro base inicial (Layer 1 e 2)
    solver = z3.Solver()
    x0 = [z3.Real(f'x0_{i}') for i in range(self.nx)]
    solver.add(z3.Sum([x*x for x in x0]) > z3.Q(1, 1000))
    G_acc = np.eye(self.nx)
    for m in sequence:
      for j in range(1, m):
        M_z3 = [[z3.Q(Fraction(val).limit_denominator(10**10).numerator,
                      Fraction(val).limit_denominator(10**10).denominator)
                 for val in row] for row in (G_acc.T @ self.Q_v[j] @ G_acc)]
        solver.add(z3.Sum([x0[r]*M_z3[r][c]*x0[c]
                   for r in range(self.nx) for c in range(self.nx)]) >= 0)
      if m < self.nu_bar:
        M_trig_z3 = [[z3.Q(Fraction(val).limit_denominator(10**10).numerator,
                           Fraction(val).limit_denominator(10**10).denominator)
                     for val in row] for row in (G_acc.T @ self.Q_v[m] @ G_acc)]
        solver.add(z3.Sum([x0[r]*M_trig_z3[r][c]*x0[c]
                   for r in range(self.nx) for c in range(self.nx)]) < 0)
      G_acc = self.A_cal[m] @ G_acc
    return solver.check() == z3.sat

  def build_hybrid_graph(self, ell, K_set, max_workers=None, verbose=True):
    if verbose:
      print(f"=== [Init] Construindo Modelo HÍBRIDO-EXATO (l={ell}) ===")

    # Filtro base (Z3 Puro)
    layer_1 = [(k,) for k in K_set if self._check_z3_only((k,))]
    layer_2_prefixes = []
    valid_base = set()
    for seq in layer_1:
      for k_next in K_set:
        candidate = seq + (k_next,)
        if self._check_z3_only(candidate):
          layer_2_prefixes.append(candidate)
          valid_base.add((seq[-1], k_next))

    G = nx.DiGraph()
    all_nodes, all_edges = set(), []
    if max_workers is None:
      max_workers = max(1, multiprocessing.cpu_count() - 1)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
      futures = [executor.submit(_worker_hybrid_exact_expand, p, ell, K_set, self.A_cal,
                                 self.Q_v, valid_base, self.nx, self.nu_bar) for p in layer_2_prefixes]
      for future in as_completed(futures):
        nodes, edges = future.result()
        all_nodes.update(nodes)
        all_edges.extend(edges)

    G.add_nodes_from(all_nodes)
    for u, v, w in all_edges:
      # Garante que o nó de destino (node_b) esteja no grafo
      # (se a transição l+1 é factível, o sufixo l também é por herança)
      G.add_node(v)
      G.add_edge(u, v, weight=u[0], label=str(u[0]))

    if verbose:
      print(
          f"Relatório Final: {G.number_of_nodes()} nós | {G.number_of_edges()} arestas.")
    return G


# Importação do Simulador do utilizador
try:
  from Simulator import recurrence_model_setm
except ImportError:
  pass

# ============================================================================
# 1. FUNÇÃO WORKER PARA SIMULAÇÃO (Paralelismo na Descoberta)
# ============================================================================


def _worker_simulation(n_samples_worker, nx_dim, ell, config_path, prob_results, h):
  discovered = set()
  for _ in range(n_samples_worker):
      # Amostragem na esfera unitária baseada na homogeneidade
    vec = np.random.randn(nx_dim)
    x0 = vec / np.linalg.norm(vec)

    event_times = recurrence_model_setm(x0, config_path, prob_results)
    if event_times is not None and len(event_times) > ell + 1:
      intervals = np.diff(event_times)
      k_seq = np.round(intervals / h).astype(int).tolist()

      for i in range(len(k_seq) - ell + 1):
        discovered.add(tuple(k_seq[i:i+ell]))
        if i + ell + 1 <= len(k_seq):
          discovered.add(tuple(k_seq[i:i+ell+1]))
  return discovered

# ============================================================================
# 2. FUNÇÃO WORKER PARA CONSTRUÇÃO DO GRAFO (Dual-Tier)
# ============================================================================


def _worker_dual_expand(prefix, ell, K_set, A_cal, Q_v, valid_base, nx_dim, nu_bar, known_valid_sequences):

  def to_rational(matrix, ctx):
    return [[z3.Q(Fraction(val).limit_denominator(10**10).numerator,
                  Fraction(val).limit_denominator(10**10).denominator, ctx=ctx)
             for val in row] for row in matrix]

  # Estado interno para contagem de "hits"
  stats = {"skip_count": 0}

  def check_sequence_dual_tier(sequence):
    # --- TIER 1: DESCOBERTA (Simulação) ---
    if sequence in known_valid_sequences:
      stats["skip_count"] += 1
      return True

    # --- TIER 2: PROVA REAL DIRETA (Z3) ---
    ctx = z3.Context()
    solver = z3.Solver(ctx=ctx)
    x0 = [z3.Real(f'x0_{i}', ctx=ctx) for i in range(nx_dim)]
    solver.add(z3.Sum([x*x for x in x0]) > z3.Q(1, 1000, ctx=ctx))

    G_acc = np.eye(nx_dim)
    for m_target in sequence:
      if m_target > nu_bar:
        return False
      for j in range(1, m_target):
        M_z3 = to_rational(G_acc.T @ Q_v[j] @ G_acc, ctx)
        solver.add(z3.Sum([x0[r]*M_z3[r][c]*x0[c]
                   for r in range(nx_dim) for c in range(nx_dim)]) >= 0)
      if m_target < nu_bar:
        M_trig_z3 = to_rational(G_acc.T @ Q_v[m_target] @ G_acc, ctx)
        solver.add(z3.Sum([x0[r]*M_trig_z3[r][c]*x0[c]
                   for r in range(nx_dim) for c in range(nx_dim)]) < 0)
      G_acc = A_cal[m_target] @ G_acc

    return solver.check() == z3.sat

  current_layer = [prefix]
  for length in range(len(prefix) + 1, ell + 1):
    next_layer = []
    for seq in current_layer:
      for k_next in K_set:
        if (seq[-1], k_next) not in valid_base:
          continue
        candidate = seq + (k_next,)
        if check_sequence_dual_tier(candidate):
          next_layer.append(candidate)
    current_layer = next_layer
    if not current_layer:
      break

  feasible_nodes = set(current_layer)
  edges = []
  for node_a in feasible_nodes:
    for k_next in K_set:
      if (node_a[-1], k_next) not in valid_base:
        continue
      candidate_trans = node_a + (k_next,)
      if check_sequence_dual_tier(candidate_trans):
        node_b = node_a[1:] + (k_next,)
        edges.append((node_a, node_b, node_a[0]))

  return feasible_nodes, edges, stats["skip_count"]

# ============================================================================
# 3. CLASSE PRINCIPAL
# ============================================================================


class TrafficModelBuilderDual:
  def __init__(self, A, B, K, Xi, Psi, h, nu_bar_time):
    self.nx = A.shape[0]
    self.h = h
    self.nu_bar = int(round(nu_bar_time / h))
    sys_d = cont2discrete(
        (A, B, np.zeros((1, self.nx)), np.zeros((1, B.shape[1]))), h, method='zoh')
    self.Ad, self.Bd = sys_d[0], sys_d[1]
    self.A_cal = [None] * (self.nu_bar + 1)
    self.Q_v = [None] * (self.nu_bar + 1)
    I = np.eye(self.nx)
    self.A_cal[0] = I
    for v in range(1, self.nu_bar + 1):
      Ad_pow = np.linalg.matrix_power(self.Ad, v)
      S_v = sum(np.linalg.matrix_power(self.Ad, p) for p in range(v))
      self.A_cal[v] = Ad_pow + S_v @ self.Bd @ K
      diff = I - self.A_cal[v]
      self.Q_v[v] = (self.A_cal[v].T @ Psi @ self.A_cal[v]
                     ) - (diff.T @ Xi @ diff)

  def _check_z3_base(self, sequence):
    solver = z3.Solver()
    x0 = [z3.Real(f'x0_{i}') for i in range(self.nx)]
    solver.add(z3.Sum([x*x for x in x0]) > z3.Q(1, 1000))
    G_acc = np.eye(self.nx)
    for m_target in sequence:
      for j in range(1, m_target):
        M_z3 = [[z3.Q(Fraction(val).limit_denominator(10**10).numerator,
                      Fraction(val).limit_denominator(10**10).denominator)
                 for val in row] for row in (G_acc.T @ self.Q_v[j] @ G_acc)]
        solver.add(z3.Sum([x0[r]*M_z3[r][c]*x0[c]
                   for r in range(self.nx) for c in range(self.nx)]) >= 0)
      if m_target < self.nu_bar:
        M_trig_z3 = [[z3.Q(Fraction(val).limit_denominator(10**10).numerator,
                           Fraction(val).limit_denominator(10**10).denominator)
                     for val in row] for row in (G_acc.T @ self.Q_v[m_target] @ G_acc)]
        solver.add(z3.Sum([x0[r]*M_trig_z3[r][c]*x0[c]
                   for r in range(self.nx) for c in range(self.nx)]) < 0)
      G_acc = self.A_cal[m_target] @ G_acc
    return solver.check() == z3.sat

  def perform_discovery(self, n_samples, ell, config_path, prob_results, max_workers):
    """Gera um cache de sequências válidas via amostragem paralela na esfera unitária."""
    print(
        f"--- Fase de Descoberta: {n_samples} amostras (Multiprocessing) ---")
    discovered_set = set()

    samples_per_worker = [n_samples // max_workers] * max_workers
    for i in range(n_samples % max_workers):
      samples_per_worker[i] += 1

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
      futures = [executor.submit(_worker_simulation, n, self.nx, ell, config_path, prob_results, self.h)
                 for n in samples_per_worker]
      for future in as_completed(futures):
        discovered_set.update(future.result())

    print(f"   {len(discovered_set)} sequências 'exatas' cacheadas.")
    return discovered_set

  def build_graph(self, ell, K_set, n_samples, config_path, prob_results, max_workers=None, verbose=True):
    if max_workers is None:
      max_workers = max(1, multiprocessing.cpu_count() - 1)

    known_valid = self.perform_discovery(
        n_samples, ell, config_path, prob_results, max_workers)

    layer_1 = [(k,) for k in K_set if self._check_z3_base((k,))]
    layer_2_prefixes = []
    valid_base = set()
    for seq in layer_1:
      for k_next in K_set:
        candidate = seq + (k_next,)
        if self._check_z3_base(candidate):
          layer_2_prefixes.append(candidate)
          valid_base.add((seq[-1], k_next))

    if verbose:
      print(f"=== Distribuindo construção exata para l={ell} ===")

    G = nx.DiGraph()
    all_nodes, all_edges = set(), []
    total_skip_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
      futures = [executor.submit(_worker_dual_expand, p, ell, K_set, self.A_cal, self.Q_v,
                                 valid_base, self.nx, self.nu_bar, known_valid) for p in layer_2_prefixes]
      for future in as_completed(futures):
        nodes, edges, skip_count = future.result()
        all_nodes.update(nodes)
        all_edges.extend(edges)
        total_skip_count += skip_count

    G.add_nodes_from(all_nodes)
    for u, v, w in all_edges:
      G.add_node(v)
      G.add_edge(u, v, weight=u[0], label=str(u[0]))

    if verbose:
      print(
          f"   Aceleração: {total_skip_count} verificações puladas via cache de simulação.")
      print(
          f"Relatório Final: {G.number_of_nodes()} nós | {G.number_of_edges()} arestas.")
    return G


# Importação do Simulador do utilizador
try:
  from Simulator import recurrence_model_setm
except ImportError:
  pass

# ============================================================================
# 1. FUNÇÃO WORKER GLOBAL (Witness-Driven BFS)
# ============================================================================


def _worker_witness_expand(prefixes_with_witnesses, ell, K_set, A_cal, Q_v, valid_base, nx_dim, nu_bar, h, config_path, prob_results):

  # Cache local de matrizes Z3 (convertidas uma única vez por worker)
  ctx = z3.Context()
  Q_v_z3 = []
  for M in Q_v:
    if M is None:
      Q_v_z3.append(None)
      continue
    Q_v_z3.append([[z3.Q(Fraction(val).limit_denominator(10**10).numerator,
                         Fraction(val).limit_denominator(10**10).denominator, ctx=ctx)
                    for val in row] for row in M])

  # A_cal não precisa ser racional no solver se usarmos M_proj pre-calculado
  A_cal_z3 = [None] * len(A_cal)

  def check_sequence_with_witness(sequence, parent_witness):
    # --- TIER 1: BYPASS VIA TESTEMUNHO (Simulação) ---
    # Verificamos se o x0 que validou o pai também valida o filho
    if parent_witness is not None:
      event_times = recurrence_model_setm(
          parent_witness, config_path, prob_results)
      if event_times is not None and len(event_times) > len(sequence):
        intervals = np.diff(event_times)
        k_seq = tuple(
            np.round(intervals[:len(sequence)] / h).astype(int).tolist())
        if k_seq == sequence:
          return True, parent_witness  # Sucesso! Bypass completo.

    # --- TIER 2: PROVA REAL (Z3) ---
    solver = z3.Solver(ctx=ctx)
    x0_vars = [z3.Real(f'x0_{i}', ctx=ctx) for i in range(nx_dim)]
    solver.add(z3.Sum([x*x for x in x0_vars]) > z3.Q(1, 1000, ctx=ctx))

    G_acc = np.eye(nx_dim)
    for m_target in sequence:
      if m_target > nu_bar:
        return False, None
      # Condições de não-disparo
      for j in range(1, m_target):
        M_proj = G_acc.T @ Q_v[j] @ G_acc
        M_z3 = [[z3.Q(Fraction(val).limit_denominator(10**10).numerator,
                      Fraction(val).limit_denominator(10**10).denominator, ctx=ctx)
                 for val in row] for row in M_proj]
        solver.add(z3.Sum([x0_vars[r]*M_z3[r][c]*x0_vars[c]
                   for r in range(nx_dim) for c in range(nx_dim)]) >= 0)

      # Condição de disparo
      if m_target < nu_bar:
        M_proj_trig = G_acc.T @ Q_v[m_target] @ G_acc
        M_trig_z3 = [[z3.Q(Fraction(val).limit_denominator(10**10).numerator,
                           Fraction(val).limit_denominator(10**10).denominator, ctx=ctx)
                     for val in row] for row in M_proj_trig]
        solver.add(z3.Sum([x0_vars[r]*M_trig_z3[r][c]*x0_vars[c]
                   for r in range(nx_dim) for c in range(nx_dim)]) < 0)

      G_acc = A_cal[m_target] @ G_acc

    if solver.check() == z3.sat:
      m = solver.model()
      witness = np.array([float(m[v].as_fraction()) for v in x0_vars])
      return True, witness
    return False, None

  # Expansão BFS local mantendo dicionário de testemunhos {sequencia: x0}
  # Dict[tuple, np.ndarray]
  current_layer_witnesses = prefixes_with_witnesses
  total_smt_calls = 0
  total_bypasses = 0

  for length in range(3, ell + 1):
    next_layer = {}
    for seq, wit in current_layer_witnesses.items():
      for k_next in K_set:
        if (seq[-1], k_next) not in valid_base:
          continue
        candidate = seq + (k_next,)

        # Tenta validar via Witness primeiro
        is_valid, new_wit = check_sequence_with_witness(candidate, wit)
        if is_valid:
          next_layer[candidate] = new_wit
          if new_wit is wit:
            total_bypasses += 1
          else:
            total_smt_calls += 1
    current_layer_witnesses = next_layer
    if not next_layer:
      break

  feasible_nodes = set(current_layer_witnesses.keys())
  edges = []
  # Geração de arestas (ell + 1)
  for node_a, wit_a in current_layer_witnesses.items():
    for k_next in K_set:
      if (node_a[-1], k_next) not in valid_base:
        continue
      candidate_trans = node_a + (k_next,)
      is_valid, _ = check_sequence_with_witness(candidate_trans, wit_a)
      if is_valid:
        edges.append((node_a, node_a[1:] + (k_next,), node_a[0]))

  return feasible_nodes, edges, total_smt_calls, total_bypasses

# ============================================================================
# 2. CLASSE PRINCIPAL
# ============================================================================


class TrafficModelBuilderWitness:
  def __init__(self, A, B, K, Xi, Psi, h, nu_bar_time):
    self.nx = A.shape[0]
    self.h = h
    self.nu_bar = int(round(nu_bar_time / h))
    sys_d = cont2discrete(
        (A, B, np.zeros((1, self.nx)), np.zeros((1, B.shape[1]))), h, method='zoh')
    self.Ad, self.Bd = sys_d[0], sys_d[1]
    self.A_cal = [None] * (self.nu_bar + 1)
    self.Q_v = [None] * (self.nu_bar + 1)
    I = np.eye(self.nx)
    self.A_cal[0] = I
    for v in range(1, self.nu_bar + 1):
      Ad_pow = np.linalg.matrix_power(self.Ad, v)
      S_v = sum(np.linalg.matrix_power(self.Ad, p) for p in range(v))
      self.A_cal[v] = Ad_pow + S_v @ self.Bd @ K
      diff = I - self.A_cal[v]
      self.Q_v[v] = (self.A_cal[v].T @ Psi @ self.A_cal[v]
                     ) - (diff.T @ Xi @ diff)

  def _check_z3_base(self, sequence):
    solver = z3.Solver()
    x0 = [z3.Real(f'x0_{i}') for i in range(self.nx)]
    solver.add(z3.Sum([x*x for x in x0]) > z3.Q(1, 1000))
    G_acc = np.eye(self.nx)
    for m_target in sequence:
      for j in range(1, m_target):
        M_proj = G_acc.T @ self.Q_v[j] @ G_acc
        M_z3 = [[z3.Q(Fraction(val).limit_denominator(10**10).numerator,
                      Fraction(val).limit_denominator(10**10).denominator) for val in row] for row in M_proj]
        solver.add(z3.Sum([x0[r]*M_z3[r][c]*x0[c]
                   for r in range(self.nx) for c in range(self.nx)]) >= 0)
      if m_target < self.nu_bar:
        M_proj_trig = G_acc.T @ self.Q_v[m_target] @ G_acc
        M_trig_z3 = [[z3.Q(Fraction(val).limit_denominator(10**10).numerator,
                           Fraction(val).limit_denominator(10**10).denominator) for val in row] for row in M_proj_trig]
        solver.add(z3.Sum([x0[r]*M_trig_z3[r][c]*x0[c]
                   for r in range(self.nx) for c in range(self.nx)]) < 0)
      G_acc = self.A_cal[m_target] @ G_acc
    if solver.check() == z3.sat:
      m = solver.model()
      return True, np.array([float(m[v].as_fraction()) for v in x0])
    return False, None

  def build_graph(self, ell, K_set, config_path, prob_results, max_workers=None, verbose=True):
    # Camada 1 e 2 (Base)
    layer_1_w = {}
    for k in K_set:
      sat, wit = self._check_z3_base((k,))
      if sat:
        layer_1_w[(k,)] = wit

    layer_2_w = {}
    valid_base = set()
    for seq, wit in layer_1_w.items():
      for k_next in K_set:
        candidate = seq + (k_next,)
        sat, wit2 = self._check_z3_base(candidate)
        if sat:
          layer_2_w[candidate] = wit2
          valid_base.add((seq[-1], k_next))

    # Distribuição Híbrida
    if max_workers is None:
      max_workers = max(1, multiprocessing.cpu_count() - 1)

    # Divide as tarefas (grupos de prefixos para cada worker)
    prefixes = list(layer_2_w.keys())
    chunks = [prefixes[i::max_workers] for i in range(max_workers)]
    tasks = [{p: layer_2_w[p] for p in chunk} for chunk in chunks if chunk]

    all_nodes, all_edges = set(), []
    total_smt, total_bypasses = 0, 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
      futures = [executor.submit(_worker_witness_expand, t, ell, K_set, self.A_cal, self.Q_v,
                                 valid_base, self.nx, self.nu_bar, self.h, config_path, prob_results)
                 for t in tasks]
      for f in as_completed(futures):
        nodes, edges, smt, bypass = f.result()
        all_nodes.update(nodes)
        all_edges.extend(edges)
        total_smt += smt
        total_bypasses += bypass

    G = nx.DiGraph()
    G.add_nodes_from(all_nodes)
    for u, v, w in all_edges:
      G.add_node(v)
      G.add_edge(u, v, weight=u[0], label=str(u[0]))

    if verbose:
      print(
          f"   Estatísticas: {total_bypasses} Bypasses via Testemunho | {total_smt} Chamadas SMT.")
      print(
          f"Relatório Final: {G.number_of_nodes()} nós | {G.number_of_edges()} arestas.")
    return G


# ============================================================================
# 1. FUNÇÃO DE CONVERSÃO RÁPIDA (Pre-calculada)
# ============================================================================


def fast_rational_matrix(matrix, ctx):
  """Converte matriz numpy para Z3 Q uma única vez por worker."""
  return [[z3.Q(Fraction(float(val)).limit_denominator(10**6).numerator,
                Fraction(float(val)).limit_denominator(10**6).denominator, ctx=ctx)
           for val in row] for row in matrix]


def _worker_entropy_branch(prefix, ell, K_set, A_cal_raw, Q_v_raw, valid_base, nx_dim, nu_bar):
  """
  Worker Otimizado: Usa propagação de estado em vez de projeção de matrizes.
  """
  ctx = z3.Context()
  # Converte as matrizes apenas UMA vez por processo worker
  A_z3 = [fast_rational_matrix(
      m, ctx) if m is not None else None for m in A_cal_raw]
  Q_z3 = [fast_rational_matrix(
      m, ctx) if m is not None else None for m in Q_v_raw]

  def check_z3(sequence):
    solver = z3.Solver(ctx=ctx)
    # x_states[i] guardará o estado inicial de cada evento
    x_curr = [z3.Real(f'x0_{i}', ctx=ctx) for i in range(nx_dim)]
    # Homogeneidade
    solver.add(z3.Sum([xi * xi for xi in x_curr]) > z3.Q(1, 1000, ctx=ctx))

    for m_target in sequence:
        # Condições de não-disparo: x_curr^T * Q_j * x_curr >= 0
      for j in range(1, m_target):
        Qj = Q_z3[j]
        term = z3.Sum([x_curr[r] * Qj[r][c] * x_curr[c]
                       for r in range(nx_dim) for c in range(nx_dim)])
        solver.add(term >= 0)

      # Condição de disparo: x_curr^T * Q_m * x_curr < 0
      if m_target < nu_bar:
        Qm = Q_z3[m_target]
        term_trig = z3.Sum([x_curr[r] * Qm[r][c] * x_curr[c]
                            for r in range(nx_dim) for c in range(nx_dim)])
        solver.add(term_trig < 0)

      # Propagação do Estado: x_next = A_m * x_curr
      Am = A_z3[m_target]
      x_next = [z3.Sum([Am[r][c] * x_curr[c]
                       for c in range(nx_dim)]) for r in range(nx_dim)]
      x_curr = x_next

    return solver.check() == z3.sat

  # Expansão BFS local a partir do prefixo
  current_layer = [prefix]
  for _ in range(len(prefix), ell):
    next_layer = []
    for seq in current_layer:
      last_k = seq[-1]
      for k_next in K_set:
        if (last_k, k_next) in valid_base:
          candidate = seq + (k_next,)
          if check_z3(candidate):
            next_layer.append(candidate)
    current_layer = next_layer
    if not current_layer:
      break

  return set(current_layer)

# ============================================================================
# 2. CLASSE DE PROJETO
# ============================================================================


class EntropyProjectBuilder:
  def __init__(self, A, B, K, Xi, Psi, h, nu_bar_time):
    self.nx = A.shape[0]
    self.h = h
    self.nu_bar = int(round(nu_bar_time / h))

    from scipy.signal import cont2discrete
    sys_d = cont2discrete(
        (A, B, np.zeros((1, self.nx)), np.zeros((1, B.shape[1]))), h, method='zoh')
    self.Ad, self.Bd = sys_d[0], sys_d[1]

    self.A_cal = [None] * (self.nu_bar + 1)
    self.Q_v = [None] * (self.nu_bar + 1)
    I = np.eye(self.nx)
    for v in range(1, self.nu_bar + 1):
      Ad_pow = np.linalg.matrix_power(self.Ad, v)
      S_v = sum(np.linalg.matrix_power(self.Ad, p) for p in range(v))
      self.A_cal[v] = Ad_pow + S_v @ self.Bd @ K
      diff = I - self.A_cal[v]
      self.Q_v[v] = (self.A_cal[v].T @ Psi @ self.A_cal[v]
                     ) - (diff.T @ Xi @ diff)

  def get_topological_entropy(self, ell, K_set):
    # 1. Filtro de Camada 2 (Poda na Main Thread)
    valid_base = set()
    l2_prefixes = []
    for k1 in K_set:
      if self._quick_check((k1,)):
        for k2 in K_set:
          if self._quick_check((k1, k2)):
            valid_base.add((k1, k2))
            l2_prefixes.append((k1, k2))

    # 2. Multiprocessing com Propagação de Estado
    nodes_ell = set()
    max_workers = max(1, multiprocessing.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
      futures = [executor.submit(_worker_entropy_branch, p, ell, K_set,
                                 self.A_cal, self.Q_v, valid_base, self.nx, self.nu_bar)
                 for p in l2_prefixes]
      for f in as_completed(futures):
        nodes_ell.update(f.result())

    if not nodes_ell:
      return 0.0

    # 3. Matriz Esparsa
    seq_list = list(nodes_ell)
    idx = {s: i for i, s in enumerate(seq_list)}
    rows, cols = [], []
    for s in seq_list:
      suffix = s[1:]
      for kn in K_set:
        target = suffix + (kn,)
        if target in idx:
          rows.append(idx[s])
          cols.append(idx[target])

    if not rows:
      return 0.0
    adj = csr_matrix((np.ones(len(rows)), (rows, cols)),
                     shape=(len(seq_list), len(seq_list)))
    try:
      vals = eigs(adj, k=1, which='LM', return_eigenvectors=False)
      return np.log2(np.abs(vals[0]))
    except:
      return 0.0

  def _quick_check(self, seq):
    from z3 import Solver, Real, Q, sat
    solver = Solver()
    x = [Real(f'x_{i}') for i in range(self.nx)]
    solver.add(z3.Sum([xi*xi for xi in x]) > Q(1, 1000))
    x_curr = x
    for m in seq:
      for j in range(1, m):
        Qj = self.Q_v[j]
        solver.add(z3.Sum([x_curr[r]*Qj[r][c]*x_curr[c]
                   for r in range(self.nx) for c in range(self.nx)]) >= 0)
      if m < self.nu_bar:
        Qm = self.Q_v[m]
        solver.add(z3.Sum([x_curr[r]*Qm[r][c]*x_curr[c]
                   for r in range(self.nx) for c in range(self.nx)]) < 0)
      Am = self.A_cal[m]
      x_curr = [z3.Sum([Am[r][c] * x_curr[c]
                       for c in range(self.nx)]) for r in range(self.nx)]
    return solver.check() == sat


class IsochronousPartitionModel:
  def __init__(self, A: np.ndarray, B: np.ndarray, K: np.ndarray,
               Xi: np.ndarray, Psi: np.ndarray, h: float, nu_bar_time: float):
    self.nx = A.shape[0]
    self.h = h
    self.nu_bar = int(round(nu_bar_time / h))

    # Discretização (ZOH)
    sys_d = cont2discrete(
        (A, B, np.zeros((1, self.nx)), np.zeros((1, B.shape[1]))), h, method='zoh')
    self.Ad, self.Bd = sys_d[0], sys_d[1]

    self.A_cal = [None] * (self.nu_bar + 1)
    self.Q_v = [None] * (self.nu_bar + 1)
    I = np.eye(self.nx)

    for v in range(1, self.nu_bar + 1):
      Ad_pow_v = np.linalg.matrix_power(self.Ad, v)
      S_v = sum([np.linalg.matrix_power(self.Ad, p) for p in range(v)])
      A_cal_v = Ad_pow_v + S_v @ self.Bd @ K
      self.A_cal[v] = A_cal_v
      diff = I - A_cal_v
      self.Q_v[v] = (A_cal_v.T @ Psi @ A_cal_v) - (diff.T @ Xi @ diff)

  def _to_rational(self, matrix):
    return [[Q(Fraction(val).limit_denominator(10**10).numerator,
               Fraction(val).limit_denominator(10**10).denominator) for val in row] for row in matrix]

  def _add_region_constraint(self, solver, x_vec, k_index):
    """Adiciona ao solver as restrições que definem a região isócrona R_k."""
    # Não disparou antes de k
    for m in range(1, k_index):
      M_z3 = self._to_rational(self.Q_v[m])
      quad = sum(x_vec[r] * M_z3[r][c] * x_vec[c]
                 for r in range(self.nx) for c in range(self.nx))
      solver.add(quad >= 0)
    # Dispara exatamente em k
    if k_index < self.nu_bar:
      M_z3_trig = self._to_rational(self.Q_v[k_index])
      quad_trig = sum(x_vec[r] * M_z3_trig[r][c] * x_vec[c]
                      for r in range(self.nx) for c in range(self.nx))
      solver.add(quad_trig < 0)

  def build_partition_graph(self) -> nx.DiGraph:
    """
    Constrói o grafo de transição entre regiões (One-step Abstraction).
    Complexidade: O(nu_bar^2) chamadas ao solver SMT.
    """
    G = nx.DiGraph()
    # Os nós são simplesmente os índices dos intervalos possíveis
    nodes = list(range(1, self.nu_bar + 1))
    G.add_nodes_from(nodes)

    print(
        f"Verificando transições entre {self.nu_bar} regiões isócronas...")

    for i in nodes:
      for j in nodes:
        solver = Solver()
        x0 = [Real(f'x0_{r}') for r in range(self.nx)]
        # Evitar origem
        solver.add(sum([xr * xr for xr in x0]) > Q(1, 1000))
        solver.add(sum([xr * xr for xr in x0]) <= 1)

        # 1. x0 deve estar na região Ri
        self._add_region_constraint(solver, x0, i)

        # 2. x_next = A_cal_i * x0
        Phi = self.A_cal[i]
        x_next = [sum(float(Phi[r, c]) * x0[c]
                      for c in range(self.nx)) for r in range(self.nx)]

        # 3. x_next deve estar na região Rj
        self._add_region_constraint(solver, x_next, j)

        if solver.check() == sat:
          G.add_edge(i, j)

    return G
