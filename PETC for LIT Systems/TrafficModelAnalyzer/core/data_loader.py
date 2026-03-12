import os
import re
from pathlib import Path
from typing import List, Dict, Any


def scan_traffic_directory(base_dir: str) -> List[Dict[str, Any]]:
  """
  Scans the specified base directory for traffic log CSV files, parsing their 
  metadata from the directory structure and filename.

  Args:
      base_dir (str): The root directory to scan (e.g., containing 'escalar', etc.).

  Returns:
      List[Dict[str, Any]]: A list of dictionaries containing file metadata:
          - 'path' (str): Absolute path to the file.
          - 'filename' (str): The name of the file.
          - 'subdirectory' (str): The immediate parent directory name.
          - 'kmax' (int | None): The extracted k_max value, or None if not found.
          - 'is_emulation' (bool): True if 'emulation' tag is in the filename.
  """
  parsed_files = []
  base_path = Path(base_dir)

  if not base_path.is_dir():
    return parsed_files

  for root, _, files in os.walk(base_path):
    for file in files:
      if not file.endswith(".csv"):
        continue

      filepath = Path(root) / file
      filename = filepath.name
      subdirectory = filepath.parent.name

      # Parse K-Max tag (e.g., 'kmax20' -> 20)
      kmax_match = re.search(r'kmax(\d+)', filename, re.IGNORECASE)
      kmax_value = int(kmax_match.group(1)) if kmax_match else None

      # Parse emulation tag
      is_emulation = 'emulation' in filename.lower()

      parsed_files.append({
          'path': str(filepath),
          'filename': filename,
          'subdirectory': subdirectory,
          'kmax': kmax_value,
          'is_emulation': is_emulation
      })

  return parsed_files
