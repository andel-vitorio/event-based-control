# -*- coding: utf-8 -*-
"""
Simulator.py
Functional implementation for simulation using C++ backend.
Provides strict I/O handling and native OpenMP parallel processing support.
Ensures determinism and bypasses Windows CLI limits using binary state transfer.
"""

import networkx as nx
from pyvis.network import Network
import numpy as np
import json
import os
import tempfile
from Utils.CppInterface import CppSimulator

# Single internal instance to manage the executable path
_backend = CppSimulator(exe_name="ETCforLinearSystemMain")


class NumpyEncoder(json.JSONEncoder):
  """
  Custom JSON Encoder to handle NumPy types during serialization.
  Ensures all matrix types are converted to standard Python lists.
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
  Generates a uniform sampling of the projective space S^1.

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
  Internal helper to merge base config with optimization results
  and write to a guaranteed closed temporary file.

  Args:
      config_path (str): Path to the original experiment JSON.
      results_dict (dict): Dictionary containing controller K and ETM matrices.
  """
  with open(config_path, 'r') as f:
    full_data = json.load(f)

  # Map Greek symbols (Ξ, Ψ) to C++ expected keys (Xi, Psi)
  # This prevents key errors in the backend ConfigLoader
  full_data['results'] = {
      "controller": {
          "K": np.array(results_dict['controller']['K']).tolist()
      },
      "etm": {
          "Xi": np.array(results_dict['etm']['Ξ']).tolist(),
          "Psi": np.array(results_dict['etm']['Ψ']).tolist()
      }
  }

  # Use delete=False to manage lifecycle manually and avoid sharing violations
  tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
  try:
    json.dump(full_data, tmp, cls=NumpyEncoder)
    tmp.flush()
    os.fsync(tmp.fileno())  # Force write to disk
    return tmp.name
  finally:
    tmp.close()


def open_loop(x0, config_path, u_constant=0.0):
  """
  Performs an Open-Loop simulation using the C++ backend.

  Args:
      x0 (list/np.ndarray): Initial state vector.
      config_path (str): Path to the experiment configuration.
      u_constant (float): Constant input for open-loop simulation.
  """
  # Ensure x0 is a clean copy to prevent mutation
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
  Performs a Closed-Loop SETM simulation (RK5).

  Args:
      x0 (list/np.ndarray): Initial state vector.
      config_path (str): Path to the base JSON.
      results_dict (dict): Optimization results containing K, Xi, and Psi.
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
  Simulates a single trajectory using the Discrete-Time Recurrence Map.
  Optimized for high-speed event detection.

  Args:
      x0 (list/np.ndarray): Initial state vector.
      config_path (str): Path to the base JSON.
      results_dict (dict): Optimization results containing K, Xi, and Psi.
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

    # Return only the event timestamps
    return sim_data['event_times']
  finally:
    if os.path.exists(temp_path):
      os.remove(temp_path)


def build_symbolic_sequences_parallel(initial_states, config_path, results_dict):
  """
  Performs massive parallel recurrence mapping using native C++/OpenMP threads.
  Replaces joblib/multiprocessing for Windows systems.
  Bypasses CLI character limits using binary temporary files.

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

  # Extract h for symbolic discretization
  with open(config_path, 'r') as f:
    h = json.load(f)["design_params"]["h"]

  try:
    # Call the C++ backend using native threads and binary data transfer
    all_event_times = _backend.run_parallel_recurrence(
        temp_path, initial_states)

    all_k_seqs = []
    for event_times in all_event_times:
      if len(event_times) > 1:
        inter_event_times = np.diff(event_times)
        # Symbolic discretization: k = round(delta / h)
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
  Generates a high-fidelity interactive visualization of the symbolic graph.
  Includes slide controls and full navigation capabilities.

  Args:
      G (networkx.DiGraph): The symbolic graph.
      sequence (list, optional): A sequence of nodes to highlight as a trajectory.
      filename (str): Output HTML filename.
  """
  net = Network(notebook=True, directed=True, width="100%",
                height="100vh", bgcolor="#fdfdfd", cdn_resources="in_line")

  # Topological Analysis (Attractors)
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

  # Node Styling
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

  # Edge Styling
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

  # Physics and Interaction Settings
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

  # Inject Custom UI
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
