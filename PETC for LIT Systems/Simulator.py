# -*- coding: utf-8 -*-
"""
Simulator.py
Functional implementation for simulation using C++ backend.
Provides strict I/O handling and native OpenMP parallel processing support.
Ensures determinism and bypasses Windows CLI limits using binary state transfer.
"""

import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.sparse.linalg as sla
from fractions import Fraction as PyFraction
from z3 import Solver, Real, sat, Q
from scipy.signal import cont2discrete
import networkx as nx
from pyvis.network import Network
import numpy as np
import json
import os
import tempfile
from Utils.CppInterface import CppSimulator
import Utils.Numeric as nm

# Single internal instance to manage the executable path
_backend = CppSimulator(exe_name="ETCforLinearSystemMain")


class NumpyEncoder(json.JSONEncoder):
  """
  A custom JSON encoder for serializing NumPy data types.

  This encoder ensures that NumPy arrays and scalars are converted to standard
  Python lists and native types (float, int) for JSON compatibility.
  """

  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    if isinstance(obj, (np.float64, np.float32, np.longdouble)):
      return float(obj)
    if isinstance(obj, (np.int64, np.int32, np.int16)):
      return int(obj)
    return super(NumpyEncoder, self).default(obj)


def projective_eps_net(eps):
  """
  Generates a uniform sampling of the projective space S^1 (unit circle in 2D).

  This function creates an epsilon-net on the unit circle, useful for covering
  the state space in homogeneous systems.

  Args:
      eps (float): Precision parameter for the epsilon-net.

  Returns:
      tuple: (X, theta) where X contains Cartesian coordinates and theta angular values.
  """
  N = int(np.ceil(np.pi / eps))
  theta = np.linspace(0, np.pi, N, endpoint=False)
  X = np.vstack([np.cos(theta), np.sin(theta)]).T
  return X, theta


def _prepare_temp_config(config_path, results_dict):
  """
  Merges the base configuration with optimization results and writes to a temporary JSON file.

  This function handles the mapping of Greek symbols to ASCII keys expected by the C++ backend
  and ensures atomic file operations to avoid race conditions.

  Args:
      config_path (str): Path to the original experiment JSON.
      results_dict (dict): Dictionary containing controller K and ETM matrices.

  Returns:
      str: Path to the temporary configuration file.
  """
  with open(config_path, 'r') as f:
    full_data = json.load(f)

  full_data['results'] = {
      "controller": {
          "K": np.array(results_dict['controller']['K']).tolist()
      },
      "etm": {
          "Xi": np.array(results_dict['etm']['Ξ']).tolist(),
          "Psi": np.array(results_dict['etm']['Ψ']).tolist()
      }
  }

  tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
  try:
    json.dump(full_data, tmp, cls=NumpyEncoder)
    tmp.flush()
    os.fsync(tmp.fileno())
    return tmp.name
  finally:
    tmp.close()


def open_loop(x0, config_path, u_constant=0.0):
  """
  Executes an open-loop simulation using the high-performance C++ backend.

  This function simulates the system dynamics without feedback control.

  Args:
      x0 (list/np.ndarray): Initial state vector.
      config_path (str): Path to the experiment configuration.
      u_constant (float): Constant input for open-loop simulation.

  Returns:
      tuple: (y, t, u) arrays containing output, time, and input history.
  """
  x0_clean = np.array(x0, copy=True).flatten()

  results = _backend.run(
      json_config_path=config_path,
      x0=x0_clean,
      u_constant=u_constant,
      closed_loop=False
  )
  if not results:
    return None, None, None
  return results['y'], results['t'], results['u']


def closed_loop_setm(x0, config_path, results_dict):
  """
  Executes a closed-loop simulation with a Static Event-Triggered Mechanism (SETM).

  This function utilizes the C++ backend to simulate the system under event-triggered
  control using Runge-Kutta integration.

  Args:
      x0 (list/np.ndarray): Initial state vector.
      config_path (str): Path to the base JSON.
      results_dict (dict): Optimization results containing K, Xi, and Psi.

  Returns:
      tuple: (y, t, u, event_times) arrays containing simulation data.
  """
  if results_dict is None:
    raise ValueError("Optimization results are required for closed-loop.")

  temp_path = _prepare_temp_config(config_path, results_dict)
  x0_clean = np.array(x0, copy=True).flatten()

  try:
    sim_data = _backend.run(
        json_config_path=temp_path,
        x0=x0_clean,
        u_constant=0.0,
        closed_loop=True
    )
    if not sim_data:
      return None, None, None, None

    return sim_data['y'], sim_data['t'], sim_data['u'], sim_data['event_times']
  finally:
    if os.path.exists(temp_path):
      os.remove(temp_path)


def recurrence_model_setm(x0, config_path, results_dict):
  """
  Simulates a single trajectory using the discrete-time recurrence map.

  This function is optimized for high-speed detection of event times without
  full state trajectory logging.

  Args:
      x0 (list/np.ndarray): Initial state vector.
      config_path (str): Path to the base JSON.
      results_dict (dict): Optimization results containing K, Xi, and Psi.

  Returns:
      np.ndarray: Array of event timestamps.
  """
  if results_dict is None:
    raise ValueError(
        "Optimization results are required for recurrence map.")

  temp_path = _prepare_temp_config(config_path, results_dict)
  x0_clean = np.array(x0, copy=True).flatten()

  try:
    sim_data = _backend.run(
        json_config_path=temp_path,
        x0=x0_clean,
        u_constant=0.0,
        closed_loop=False,
        recurrence=True
    )

    if not sim_data:
      return None

    return sim_data['event_times']
  finally:
    if os.path.exists(temp_path):
      os.remove(temp_path)


def build_symbolic_sequences_parallel(initial_states, config_path, results_dict):
  """
  Performs parallel recurrence mapping using native C++/OpenMP threads.

  This function efficiently computes symbolic sequences for a batch of initial states,
  leveraging the C++ backend for performance and bypassing CLI limitations via binary
  data transfer.

  Args:
      initial_states (np.ndarray): Array of sampled initial states [N x dim].
      config_path (str): Path to the base JSON configuration.
      results_dict (dict): Optimization results containing K, Xi, and Psi.

  Returns:
      list: A list of numpy arrays, each representing an event sequence (k_seq).
  """
  if results_dict is None:
    raise ValueError(
        "Optimization results are required for parallel mapping.")

  temp_path = _prepare_temp_config(config_path, results_dict)
  h = 0.0

  with open(config_path, 'r') as f:
    h = json.load(f)["design_params"]["h"]

  try:
    all_event_times = _backend.run_parallel_recurrence(
        temp_path, initial_states)

    all_k_seqs = []
    for event_times in all_event_times:
      if len(event_times) > 1:
        inter_event_times = np.diff(event_times)
        k_seq = np.round(inter_event_times / h).astype(int).tolist()
        all_k_seqs.append(k_seq)
      else:
        all_k_seqs.append([])

    return all_k_seqs
  finally:
    if os.path.exists(temp_path):
      os.remove(temp_path)


def visualize_graph(G, sequence=None, filename="symbolic_model_final.html"):
  """
  Generates an interactive HTML visualization of the symbolic graph.

  The visualization includes navigation controls, node highlighting for specific
  sequences, and topological analysis features.

  Args:
      G (networkx.DiGraph): The symbolic graph.
      sequence (list, optional): A sequence of nodes to highlight as a trajectory.
      filename (str): Output HTML filename.
  """
  net = Network(notebook=True, directed=True, width="100%",
                height="100vh", bgcolor="#fdfdfd", cdn_resources="in_line")

  sccs = list(nx.strongly_connected_components(G))
  attractor_nodes = set()
  for scc in sccs:
    if len(scc) > 1 or (len(scc) == 1 and G.has_edge(list(scc)[0], list(scc)[0])):
      attractor_nodes.update(scc)

  node_id_map = {node: str(i) for i, node in enumerate(G.nodes)}

  sequence_ids = []
  if sequence:
    sequence_ids = [node_id_map[node]
                    for node in sequence if node in node_id_map]

  for node, node_id in node_id_map.items():
    is_seq = sequence and node in sequence
    is_attr = node in attractor_nodes

    if is_seq:
      color = "#007AFF"      # iOS Blue
      size = 40
      font = {'size': 20, 'color': "#000000",
              'face': 'Segoe UI', 'vadjust': -35}
      border_width = 3
    elif is_attr:
      color = "#FF9500"      # Orange
      size = 30
      font = {'size': 16, 'color': "#555555",
              'face': 'Segoe UI', 'vadjust': -30}
      border_width = 2
    else:
      color = "#E5E5EA"      # Light Gray
      size = 15
      font = {'size': 10, 'color': "#AAAAAA", 'face': 'arial'}
      border_width = 1

    net.add_node(
        node_id,
        label=str(node),
        title=f"State: {node}",
        shape="dot",
        size=size,
        color={'background': color, 'border': 'white',
               'highlight': {'border': '#333', 'background': color}},
        font=font,
        borderWidth=border_width
    )

  for u, v, data in G.edges(data=True):
    u_id, v_id = node_id_map[u], node_id_map[v]
    weight = data.get("weight")
    label_text = str(weight) if weight is not None else ""

    is_in_sequence = False
    if sequence:
      for i in range(len(sequence) - 1):
        if u == sequence[i] and v == sequence[i + 1]:
          is_in_sequence = True
          break

    if is_in_sequence:
      color = "#FF2D55"
      width = 4
      font = {
          'size': 22, 'color': '#000000', 'background': 'white',
          'strokeWidth': 0, 'align': 'middle', 'multi': 'html'
      }
      dashes = False
    else:
      color = "#C7C7CC"
      width = 1
      font = {
          'size': 12, 'color': '#8E8E93', 'background': 'white',
          'strokeWidth': 0, 'align': 'middle'
      }
      dashes = True

    net.add_edge(u_id, v_id, label=label_text, color=color,
                 width=width, font=font, dashes=dashes, arrows="to")

  options = {
      "nodes": {"font": {"strokeWidth": 2, "strokeColor": "#ffffff"}},
      "edges": {
          "smooth": {"type": "curvedCW", "roundness": 0.2},
          "font": {"align": "middle"}
      },
      "physics": {
          "forceAtlas2Based": {
              "gravitationalConstant": -120,
              "centralGravity": 0.008,
              "springLength": 230,
              "damping": 0.9
          },
          "solver": "forceAtlas2Based",
          "stabilization": {"iterations": 200, "updateInterval": 50}
      },
      "interaction": {
          "hover": True,
          "navigationButtons": True,
          "keyboard": True,
          "zoomView": True
      }
  }
  net.set_options(f"var options = {json.dumps(options)}")
  net.save_graph(filename)

  with open(filename, "r", encoding="utf-8") as f:
    html_content = f.read()

  custom_ui = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
        
        #thesis-ui {{
            position: absolute; top: 20px; right: 20px; width: 260px;
            padding: 20px; 
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 16px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.5);
            font-family: 'Inter', sans-serif;
            z-index: 1000;
            transition: all 0.3s ease;
        }}
        
        #thesis-ui:hover {{
            box-shadow: 0 15px 50px rgba(0, 0, 0, 0.15);
            transform: translateY(-2px);
        }}

        h3 {{ margin: 0 0 15px 0; font-size: 16px; color: #1c1c1e; font-weight: 600; letter-spacing: -0.5px; }}
        
        .legend-row {{ display: flex; align-items: center; margin-bottom: 8px; font-size: 13px; color: #3a3a3c; }}
        .dot {{ width: 10px; height: 10px; border-radius: 50%; margin-right: 12px; }}
        .line {{ width: 20px; height: 3px; margin-right: 12px; border-radius: 2px; }}
        
        .divider {{ height: 1px; background: rgba(0,0,0,0.08); margin: 15px 0; }}
        
        .ctrl-panel {{ display: flex; justify-content: space-between; align-items: center; }}
        
        button {{
            background: #007AFF; color: white; border: none; padding: 8px 16px;
            border-radius: 8px; font-weight: 600; font-size: 13px; cursor: pointer;
            transition: background 0.2s;
            flex: 1; margin: 0 4px;
        }}
        button:hover {{ background: #0056D2; }}
        button:active {{ transform: scale(0.96); }}
        
        #step-counter {{ 
            text-align: center; margin-top: 10px; font-size: 12px; 
            color: #8E8E93; font-variant-numeric: tabular-nums; 
        }}
    </style>

    <div id="thesis-ui">
        <h3>Symbolic Control Viewer</h3>
        
        <div class="legend-row"><span class="dot" style="background:#007AFF; box-shadow: 0 0 8px rgba(0,122,255,0.4);"></span> Trajectory</div>
        <div class="legend-row"><span class="dot" style="background:#FF9500;"></span> Attractor</div>
        <div class="legend-row"><span class="line" style="background:#FF2D55;"></span> Transition (k)</div>
        <div class="legend-row"><span class="line" style="background:#C7C7CC; border:1px dashed #aaa;"></span> Others</div>
        
        <div class="divider"></div>
        
        <div class="ctrl-panel">
            <button onclick="prevStep()">← Prev</button>
            <button onclick="nextStep()">Next →</button>
        </div>
        <div id="step-counter">Initial State</div>
    </div>

    <script>
        var sequenceIds = {json.dumps(sequence_ids)};
        var currentStep = 0;
        
        // Animation Configuration
        var focusOptions = {{
            scale: 1.5,
            offset: {{x:0, y:0}},
            animation: {{
                duration: 1000,
                easingFunction: "easeInOutQuart"
            }}
        }};

        network.once("afterDrawing", function() {{
            if(sequenceIds.length > 0) focusNode(0);
        }});

        function updateUI() {{
            document.getElementById('step-counter').innerText = 
                "Step " + (currentStep + 1) + " of " + sequenceIds.length;
        }}

        function focusNode(index) {{
            if(sequenceIds.length === 0) return;
            var nodeId = sequenceIds[index];
            
            network.selectNodes([nodeId]);
            network.focus(nodeId, focusOptions);
            updateUI();
        }}

        function nextStep() {{
            if (currentStep < sequenceIds.length - 1) {{
                currentStep++;
                focusNode(currentStep);
            }}
        }}

        function prevStep() {{
            if (currentStep > 0) {{
                currentStep--;
                focusNode(currentStep);
            }}
        }}

        // Keyboard Shortcuts
        document.addEventListener('keydown', function(event) {{
            if (event.key === "ArrowRight") nextStep();
            if (event.key === "ArrowLeft") prevStep();
        }});
    </script>
    </body>
    """

  final_html = html_content.replace('</body>', custom_ui)
  with open(filename, "w", encoding="utf-8") as f:
    f.write(final_html)


class SequenceReconstructor:
  """
  Reconstructs initial states (x0) compatible with a predefined event sequence in ETC/SETM systems.

  This class uses the Z3 theorem prover to solve quadratic triggering constraints and find a valid
  initial state for a given symbolic sequence.
  """

  def __init__(self, A_c, B_c, K, Xi, Psi, h, iet_max):
    """
    Initializes the reconstructor and pre-computes transition and triggering matrices.

    Args:
        A_c, B_c: Continuous system matrices.
        K: Controller gain.
        Xi, Psi: Triggering mechanism weight matrices.
        h: Discretization step.
        iet_max: Maximum inter-event time.
    """
    self.nx = A_c.shape[0]
    self.h = h
    self.m_max = int(round(iet_max / h))

    sys_d = cont2discrete((A_c, B_c, np.zeros((1, self.nx)),
                           np.zeros((1, B_c.shape[1]))), h, method='zoh')
    self.Ad, self.Bd = sys_d[0], sys_d[1]

    self.Phi_cache = [None] * (self.m_max + 1)
    self.M_cache = [None] * (self.m_max + 1)

    I = np.eye(self.nx)
    self.Phi_cache[0] = I

    for m in range(1, self.m_max + 1):
      Ad_pow_m = np.linalg.matrix_power(self.Ad, m)
      S_m = np.zeros_like(self.Ad)
      for p in range(m):
        S_m += np.linalg.matrix_power(self.Ad, p)
      Phi_m = Ad_pow_m + S_m @ self.Bd @ K
      self.Phi_cache[m] = Phi_m
      diff = I - Phi_m
      M_val = (Phi_m.T @ Psi @ Phi_m) - (diff.T @ Xi @ diff)
      self.M_cache[m] = M_val

  def _to_rational(self, matrix):
    return [[Q(PyFraction(val).limit_denominator(10**10).numerator,
               PyFraction(val).limit_denominator(10**10).denominator)
             for val in row] for row in matrix]

  def find_compatible_state(self, sequence):
    """
    Finds a non-zero initial state vector x0 that generates the exact provided event sequence.

    Args:
        sequence (list[int]): List of inter-event intervals (in steps h).

    Returns:
        np.array: Valid state vector x0, or None if the sequence is impossible.
    """
    solver = Solver()
    x0 = [Real(f'x0_{i}') for i in range(self.nx)]
    solver.add(sum([x * x for x in x0]) > Q(1, 1000))
    G_acc = np.eye(self.nx)

    for m_target in sequence:
      if m_target > self.m_max:
        return None

      for k in range(1, m_target):
        M_k = self.M_cache[k]
        M_proj = G_acc.T @ M_k @ G_acc
        M_z3 = self._to_rational(M_proj)

        quad_form = sum(x0[r] * M_z3[r][c] * x0[c]
                        for r in range(self.nx) for c in range(self.nx))
        solver.add(quad_form >= 0)

      if m_target < self.m_max:
        M_target_mat = self.M_cache[m_target]
        M_proj_trig = G_acc.T @ M_target_mat @ G_acc
        M_z3_trig = self._to_rational(M_proj_trig)

        quad_form_trig = sum(x0[r] * M_z3_trig[r][c] * x0[c]
                             for r in range(self.nx) for c in range(self.nx))
        solver.add(quad_form_trig < 0)

      G_acc = self.Phi_cache[m_target] @ G_acc

    if solver.check() == sat:
      model = solver.model()
      return np.array([float(model[v].as_fraction()) for v in x0])
    else:
      return None


def build_symbolic_exactly_graph(target_l, K_set, h, A, B, K, Xi, Psi, iet_max):
  """
  Constructs the symbolic graph by progressively increasing the sequence length.

  This function reduces computational complexity by pruning infeasible branches early
  in the construction process.
  """
  print(f"=== [Init] Pre-calculating dynamics (Z3 Backend) ===")
  reconstructor = SequenceReconstructor(A, B, K, Xi, Psi, h, iet_max)
  valid_sequences = []
  print(f"--- Layer 1 ---")
  for k in K_set:
    seq = (k,)
    if reconstructor.find_compatible_state(seq) is not None:
      valid_sequences.append(seq)

  for length in range(2, target_l + 1):
    print(f"--- Layer {length} ---")
    next_valid_sequences = []
    candidates_count = len(valid_sequences) * len(K_set)

    for seq in valid_sequences:
      for k_next in K_set:
        candidate = seq + (k_next,)

        if reconstructor.find_compatible_state(candidate) is not None:
          next_valid_sequences.append(candidate)

    dropped = candidates_count - len(next_valid_sequences)
    print(
        f"   Candidates: {candidates_count} | Valid: {len(next_valid_sequences)} | Pruned: {dropped}")

    if not next_valid_sequences:
      print("WARNING: No valid sequences found at this length!")
      break

    valid_sequences = next_valid_sequences

  feasible_nodes = valid_sequences
  feasible_nodes_set = set(feasible_nodes)
  G = nx.DiGraph()
  G.add_nodes_from(feasible_nodes)

  print(
      f"=== [Edges] Generating edges (Extension to size {target_l + 1}) ===")
  count_edges = 0

  for node_a in feasible_nodes:
    suffix = node_a[1:]

    for k_next in K_set:
      node_b = suffix + (k_next,)

      if node_b in feasible_nodes_set:
        transition_seq = node_a + (k_next,)
        if reconstructor.find_compatible_state(transition_seq) is not None:
          G.add_edge(
              node_a,
              node_b,
              weight=node_a[0],
              label=str(node_a[0])
          )
          count_edges += 1

  print("\n=== Final Report ===")
  print(f"Final Nodes: {len(feasible_nodes)}")
  print(f"Edges:       {count_edges}")

  return G


class SequenceRegionAnalyzer:
  """
  Analyzes and determines the analytical angular regions (cones) in the phase plane.

  This class identifies the set of initial angles that generate a specific symbolic
  sequence for linear homogeneous systems.
  """

  def __init__(self, reconstructor):
    """
    Args:
        reconstructor: Instance of SequenceReconstructor containing 
                       pre-computed Phi and M matrices.
    """
    self.rec = reconstructor

  def _get_angular_interval(self, M):
    """
    Computes the angular intervals theta in [0, pi) where the quadratic form x'Mx >= 0.
    """
    a = M[1, 1]
    b = M[0, 1] + M[1, 0]
    c = M[0, 0]

    delta = b**2 - 4*a*c

    if delta < 0:
      return [(0, np.pi)] if c >= 0 else []

    t1 = (-b - np.sqrt(delta)) / (2 * a) if a != 0 else -c/b
    t2 = (-b + np.sqrt(delta)) / (2 * a) if a != 0 else -c/b

    angles = sorted([np.arctan(t1) % np.pi, np.arctan(t2) % np.pi])

    mid_test = angles[0] + (angles[1] - angles[0]) / 2
    test_vec = np.array([np.cos(mid_test), np.sin(mid_test)])

    if test_vec.T @ M @ test_vec >= 0:
      return [(angles[0], angles[1])]
    else:
      return [(0, angles[0]), (angles[1], np.pi)]

  def find_region(self, sequence):
    """
    Identifies the valid angular intervals [theta_min, theta_max] for a given symbolic sequence.

    Args:
        sequence (tuple): The target sequence of inter-event steps (k).

    Returns:
        list: List of tuples representing valid angular intervals in radians.
    """
    G_acc = np.eye(2)
    final_intervals = [(0, np.pi)]

    for m_target in sequence:
      for k in range(1, m_target):
        M_eff = G_acc.T @ self.rec.M_cache[k] @ G_acc
        new_interval = self._get_angular_interval(M_eff)
        final_intervals = self._intersect(
            final_intervals, new_interval)

      if m_target < self.rec.m_max:
        M_eff_trig = G_acc.T @ self.rec.M_cache[m_target] @ G_acc
        new_interval = self._get_angular_interval(-M_eff_trig)
        final_intervals = self._intersect(
            final_intervals, new_interval)

      G_acc = self.rec.Phi_cache[m_target] @ G_acc

    return final_intervals

  def _intersect(self, intervals_a, intervals_b):
    res = []
    for start_a, end_a in intervals_a:
      for start_b, end_b in intervals_b:
        s = max(start_a, start_b)
        e = min(end_a, end_b)
        if s < e:
          res.append((s, e))
    return res


def verify_region_robustness(target_seq, region_list, config, prob_results, experiment_file, n_samples=100, eps=1e-6):
  """
  Validates the analytical invariance region using statistical sampling and high-fidelity simulation.

  This function checks if sampled points within the calculated region produce the expected
  symbolic sequence.

  Args:
      target_seq (tuple): The symbolic sequence to verify.
      region_list (list): List of (theta_min, theta_max) tuples.
      config (dict): Configuration dictionary containing design_params.
      prob_results (dict): Dictionary containing controller and ETM matrices.
      experiment_file (str): Path to the simulation configuration file.
      n_samples (int): Number of points to sample within the region.
      eps (float): Safety margin to mitigate boundary numerical sensitivities.
  """
  if not region_list:
    print("Error: The provided region list is empty.")
    return

  sampling_period = config["design_params"]['h']
  theta_min, theta_max = region_list[0]

  theta_min_safe = theta_min + eps
  theta_max_safe = theta_max - eps

  if theta_min_safe >= theta_max_safe:
    print("Warning: Epsilon margin exceeds region width. Using center point only.")
    sampled_angles = np.array([(theta_min + theta_max) / 2])
  else:
    sampled_angles = np.linspace(theta_min_safe, theta_max_safe, n_samples)

  stats = {"success": 0, "failure": 0}
  failures = []

  print(f"--- Initiating Robustness Verification (eps={eps}) ---")
  print(f"Target Sequence: {target_seq}")
  print(
      f"Safe Range: [{np.degrees(theta_min_safe):.4f}°, {np.degrees(theta_max_safe):.4f}°]\n")

  for i, theta in enumerate(sampled_angles):
    x0_sample = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)

    try:
      events = recurrence_model_setm(
          x0_sample, experiment_file, prob_results)
      inter_event_times = nm.compute_deltas(events)

      sim_seq = np.round(
          np.array(inter_event_times[1:]) / sampling_period).astype(int)
      obtained_seq = tuple(sim_seq[:len(target_seq)])

      if obtained_seq == target_seq:
        stats["success"] += 1
      else:
        stats["failure"] += 1
        failures.append({
            "sample_idx": i,
            "angle_deg": np.degrees(theta),
            "got": obtained_seq
        })

    except Exception as e:
      stats["failure"] += 1
      print(f"Execution error at sample {i}: {e}")

  total_effective = len(sampled_angles)
  accuracy = (stats["success"] / total_effective) * 100

  print("-" * 45)
  print(f"VERIFICATION COMPLETE")
  print(f"Reliability: {accuracy:.2f}%")
  print(f"OK: {stats['success']} | FAIL: {stats['failure']}")
  print("-" * 45)

  if failures:
    print("\nAnalysis of Divergences:")
    for fail in failures[:3]:
      print(
          f"Sample {fail['sample_idx']}: θ = {fail['angle_deg']:.6f}° -> Got {fail['got']}")


def compute_analytical_metrics(G, h, reconstructor):
  """
  Generates a comprehensive analytical report of ETC traffic metrics.

  This function calculates global limits, recurrence limits, behavioral entropy,
  and performance metrics based on the symbolic graph structure.
  """

  if G.number_of_nodes() == 0:
    print("Error: Empty graph.")
    return {}

  print(f"\n{'='*60}")
  print(f"FULL ANALYTICAL TRAFFIC REPORT (Gleizer & Mazo Jr., 2023)")
  print(f"{'='*60}")
  print(
      f"Graph Structure: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

  def get_k(node):
    try:
      val = node[-1] if isinstance(node, (tuple, list)) else node
      return int(val)
    except:
      return int(node)

  def fmt(val, precision=5):
    return f"{val:.{precision}f}" if val is not None else "N/A"

  def get_matrix_from_cache(cache, k):
    try:
      if isinstance(cache, dict):
        return cache.get(k)
      elif isinstance(cache, (list, tuple, np.ndarray)):
        if 0 <= k < len(cache):
          return cache[k]
        if k < len(cache):
          return cache[k]
      return None
    except:
      return None

  inf_val = None
  sup_val = None
  inf_liminf = None
  sup_limsup = None
  entropy = None
  is_chaotic = False
  complexity_class = "Unknown"
  inf_lim_avg = None
  sup_lim_avg = None
  rob_inf_lim_avg = None
  rob_sup_lim_avg = None
  num_stable = 0
  num_unstable = 0
  non_trivial_sccs = 0
  max_scc_size = 0
  cycle_metrics = []
  savings_pct = None

  try:
    h_scalar = float(h)
  except:
    h_scalar = 0.005

  # --- GLOBAL LIMITS ---
  try:
    all_ks = [get_k(n) for n in G.nodes()]
    if all_ks:
      inf_val = min(all_ks) * h_scalar
      sup_val = max(all_ks) * h_scalar
  except Exception as e:
    print(f"Global limits error: {e}")

  # --- RECURRENCE LIMITS (SCCs) ---
  try:
    sccs = list(nx.strongly_connected_components(G))
    recurrent_ks = []

    for scc in sccs:
      if len(scc) > 1 or (len(scc) == 1 and G.has_edge(list(scc)[0], list(scc)[0])):
        non_trivial_sccs += 1
        max_scc_size = max(max_scc_size, len(scc))
        for n in scc:
          recurrent_ks.append(get_k(n))

    if recurrent_ks:
      inf_liminf = min(recurrent_ks) * h_scalar
      sup_limsup = max(recurrent_ks) * h_scalar
  except Exception as e:
    print(f"SCC error: {e}")

  # --- BEHAVIORAL ENTROPY ---
  try:
    if G.number_of_nodes() < 200:
      try:
        adj = nx.adjacency_matrix(G).todense()
      except:
        adj = nx.adjacency_matrix(G).toarray()
      vals = la.eigvals(adj)
      rho = float(max(abs(vals)))
    else:
      try:
        adj = nx.adjacency_matrix(G).astype(float)
      except:
        adj = nx.adjacency_matrix(G).asfptype()
      vals = sla.eigs(adj, k=1, which='LM', return_eigenvectors=False)
      rho = float(abs(vals[0]))

    entropy = np.log2(rho) if rho > 1.0 else 0.0
    is_chaotic = entropy > 1e-6

    complexity_class = "CHAOTIC" if is_chaotic else "ORDERLY"
    if is_chaotic and max_scc_size > 10:
      complexity_class += " (Complex Attractor)"

  except Exception as e:
    print(f"Entropy warning: {e}")

  # --- CYCLE ANALYSIS & ROBUST METRICS ---
  try:
    dim = 2
    if hasattr(reconstructor, 'M_cache') and reconstructor.M_cache:
      cache = reconstructor.M_cache
      sample_matrix = None
      if isinstance(cache, dict):
        if cache:
          sample_matrix = next(iter(cache.values()))
      elif isinstance(cache, (list, tuple)):
        if len(cache) > 0:
          sample_matrix = cache[0] if cache[0] is not None else (
              cache[1] if len(cache) > 1 else None)

      if sample_matrix is not None:
        dim = sample_matrix.shape[0]

    cycles = list(nx.simple_cycles(G))

    for cycle in cycles:
      ks = [get_k(n) for n in cycle]
      if not ks:
        continue

      total_time = sum(ks) * h_scalar
      mean_iet = float(total_time / len(ks))

      M_sigma = np.eye(dim)

      for k_val in ks:
        k_int = int(k_val)
        M_k = get_matrix_from_cache(reconstructor.M_cache, k_int)

        if M_k is not None:
          M_sigma = M_k @ M_sigma
        else:
          pass

      eigvals = np.linalg.eigvals(M_sigma)
      spec_rad = float(max(np.abs(eigvals)))
      is_stable = spec_rad < (1.0 - 1e-9)

      cycle_metrics.append({
          "mean": mean_iet,
          "stable": is_stable
      })

    if cycle_metrics:
      all_means = [c["mean"] for c in cycle_metrics]
      inf_lim_avg = min(all_means)
      sup_lim_avg = max(all_means)

      if inf_lim_avg > 1e-9:
        savings_pct = (1.0 - (h_scalar / inf_lim_avg)) * 100.0

      stable_means = [c["mean"] for c in cycle_metrics if c["stable"]]
      if stable_means:
        rob_inf_lim_avg = min(stable_means)
        rob_sup_lim_avg = max(stable_means)

      num_stable = len(stable_means)
      num_unstable = len(cycle_metrics) - num_stable

  except Exception as e:
    print(f"Cycle analysis failed: {e}")

  # --- REPORT PRINTING ---
  print(f"\n--- 1. Global & Limit Metrics ---")
  print(f"Inf (MIST):       {fmt(inf_val)} s")
  print(f"Sup (MaxIST):     {fmt(sup_val)} s")
  print(f"InfLimInf:        {fmt(inf_liminf)} s")
  print(f"SupLimSup:        {fmt(sup_limsup)} s")

  print(f"\n--- 2. Complexity & Chaos ---")
  print(f"Behavioral Entropy: {fmt(entropy, 4)} bits")
  print(f"Classification:     {complexity_class}")
  print(f"Non-Trivial SCCs:   {non_trivial_sccs}")

  print(f"\n--- 3. Performance Metrics (Average IET) ---")
  print(f"{'Metric':<20} | {'Value':<12} | {'Description'}")
  print("-" * 60)
  print(f"{'InfLimAvg':<20} | {fmt(inf_lim_avg):<12} | Worst-case (Theoretical)")
  print(f"{'SupLimAvg':<20} | {fmt(sup_lim_avg):<12} | Best-case (Theoretical)")

  if savings_pct is not None:
    savings_str = f"{savings_pct:.2f}%"
    print(f"{'Min Savings':<20} | {savings_str:<12} | vs. Periodic (h={h_scalar}s)")

  print("-" * 60)

  print(f"\n--- 4. Robustness Analysis (Stability) ---")
  print(f"Total Cycles Found: {len(cycle_metrics)}")
  print(f"  - Stable:         {num_stable}")
  print(f"  - Unstable:       {num_unstable}")

  if rob_inf_lim_avg is not None:
    print(f"\nROBUST InfLimAvg:   {fmt(rob_inf_lim_avg)} s")
    print("Note: Guaranteed performance on physically observable orbits.")
  else:
    status = "Undefined"
    if not cycle_metrics:
      status += " (No cycles found)"
    elif num_stable == 0:
      status += " (All cycles unstable/chaotic)"
    print(f"\nROBUST Metric:      {status}")

  print(f"{'='*60}\n")

  return {
      "Inf": inf_val,
      "Sup": sup_val,
      "InfLimInf": inf_liminf,
      "InfLimAvg": inf_lim_avg,
      "SupLimAvg": sup_lim_avg,
      "Entropy": entropy,
      "IsChaotic": is_chaotic,
      "RobInfLimAvg": rob_inf_lim_avg,
      "SavingsPct": savings_pct
  }


def plot_cycle_distribution(G, h):
  """
  Plots a histogram of average inter-event times for all simple cycles in the symbolic graph.

  This visualization displays the density of periodic traffic patterns.
  """
  try:
    cycles = list(nx.simple_cycles(G))

    if not cycles:
      print("No cycles found in the graph.")
      return

    cycle_means = []
    for cycle in cycles:
      total_time = 0
      for node in cycle:
        try:
          k = node[-1] if isinstance(node, (tuple, list)) else node
        except:
          k = node
        total_time += float(k) * h

      mean_val = total_time / len(cycle)
      cycle_means.append(mean_val)

    fig, ax = plt.subplots(figsize=(10, 6))

    counts, bins, patches = ax.hist(
        cycle_means,
        bins=30,
        color='#003366',
        edgecolor='white',
        alpha=0.7,
        label='Cycle Means'
    )

    min_mean = min(cycle_means)
    max_mean = max(cycle_means)

    ax.axvline(min_mean, color='#B22222', linestyle='--', linewidth=2,
               label=f'InfLimAvg (Worst Case): {min_mean:.5f}s')
    ax.axvline(max_mean, color='#228B22', linestyle='--', linewidth=2,
               label=f'SupLimAvg (Best Case): {max_mean:.5f}s')

    ax.set_xlabel('Average Inter-Event Time [s]', fontsize=14)
    ax.set_ylabel('Count of Simple Cycles', fontsize=14)
    ax.set_title(
        f'Distribution of Periodic Traffic Patterns (N={len(cycles)})', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()

    print(f"Total Simple Cycles Found: {len(cycles)}")

  except Exception as e:
    print(f"Error plotting cycle distribution: {e}")
