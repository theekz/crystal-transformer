"""
Crystal Transformer: Multi-Fidelity Transfer Learning for Photocatalytic Bandgap Prediction
-----------------------------------------------------------------------------------------
CODE 01: Graph Preprocessing, Topology Optimization, and Dataset Split Preparation
Grounded in Paper Section III-A (Datasets), III-B (Graph Construction), & III-C (Topology)

Author: Thushara Subasinghe, Kavindi D.M.N., Perera E.T.B.
Journal Submission: Computational Materials Science (Elsevier)
"""

import os
import random
import numpy as np
import torch
from torch_geometric.data import Data
from pymatgen.core import Structure
from tqdm import tqdm

# ==============================================================================
# 1. TOPOLOGY OPTIMIZATION CONFIGURATION (Paper Section III-C)
# ==============================================================================
# To reproduce the Cutoff Sensitivity Analysis (Rc in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0] Å),
# simply uncomment your target Rc parameter below and run this script.

# Rc = 1.0  # (Å) - Very sparse, misses meaningful chemical environments
# Rc = 1.5  # (Å)
Rc = 2.0    # (Å) - OPTIMAL threshold selected for the Crystal Transformer [Optimal CCR]
# Rc = 2.5  # (Å)
# Rc = 3.0  # (Å)
# Rc = 4.0  # (Å) - Excessively dense, adds computational noise and validation memory overhead
# Rc = 6.0  # (Å)
# Rc = 8.0  # (Å)

# Set seed for reproducible split indices
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ==============================================================================
# 2. CRYSTAL GRAPH CONSTRUCTION NATIVE PIPELINE (Paper Section III-B)
# ==============================================================================
def crystal_structure_to_pyg_graph(pymatgen_structure, bandgap_value=None):
    """
    Translates a crystal structure into a PyTorch Geometric directed multigraph.
    Preserves periodic boundary conditions (PBC) during atomic distance lookup.
    """
    # Node features: Atomic numbers Zi normalized by 100 for network stability
    atomic_numbers = [site.specie.number for site in pymatgen_structure]
    x = torch.tensor(atomic_numbers, dtype=torch.float).view(-1, 1) / 100.0

    # Retrieve all atomic neighbors within coordination cutoff radius (Rc)
    # Pymatgen natively handles Periodic Boundary Conditions (PBC) calculations
    all_neighbors = pymatgen_structure.get_all_neighbors(Rc, include_index=True)

    edge_indices = []
    edge_distances = []

    for i, neighbors in enumerate(all_neighbors):
        for neighbor in neighbors:
            j = neighbor[5]      # Index of neighbor atom
            dist = neighbor[6]   # Interatomic distance d_ij
            
            # Record directed edge i -> j
            edge_indices.append([i, j])
            edge_distances.append([dist])

    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_distances, dtype=torch.float)
    else:
        # Edge case: handling isolated atoms if cutoff is extremely small (e.g. Rc = 1.0 Å)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)

    # Compile into PyTorch Geometric Data object
    pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    if bandgap_value is not None:
        pyg_data.y = torch.tensor([bandgap_value], dtype=torch.float)

    return pyg_data


# ==============================================================================
# 3. REPOSITORY PATH PREPARATION & BATCH PROCESSING
# ==============================================================================
def process_cif_directory(cif_dir, metadata_bg_dict, output_filepath):
    """
    Iterates over a folder of CIF files, converts them to graphs, and saves
    the processed list of PyG graphs to disk.
    """
    processed_graphs = []
    cif_files = [f for cif_files in os.listdir(cif_dir) if cif_files.endswith('.cif')]
    
    if not cif_files:
        print(f"⚠️ No CIF files found in: {cif_dir}")
        return

    print(f"Generating crystal graphs for {len(cif_files)} structures (Rc = {Rc} Å)...")
    for filename in tqdm(cif_files):
        cif_path = os.path.join(cif_dir, filename)
        try:
            # Load Structure natively with Pymatgen
            structure = Structure.from_file(cif_path)
            
            # Map structural ID to bandgap value from database metadata dictionary
            material_id = os.path.splitext(filename)
            bandgap = metadata_bg_dict.get(material_id, None)
            
            # Generate graph object
            pyg_graph = crystal_structure_to_pyg_graph(structure, bandgap)
            pyg_graph.id = material_id
            processed_graphs.append(pyg_graph)
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {str(e)}")
            continue

    # Save complete dataset to processed folder
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    torch.save(processed_graphs, output_filepath)
    print(f"✅ Successfully processed dataset saved at: {output_filepath}\n")


# ==============================================================================
# 4. DATASET CLEANING & REPRODUCIBLE SPLITTING (Paper Section III-A)
# ==============================================================================
def split_and_save_data(processed_data_path, split_type="PBE"):
    """
    Loads saved processed graphs and partitions them into reproducible splits.
    PBE Splits: Train/Validation (90:10 ratio)
    mBJ Splits: Train/Validation/Test (80:10:10 ratio)
    """
    dataset = torch.load(processed_data_path)
    random.shuffle(dataset)  # Shuffles with global seed 42
    n_samples = len(dataset)

    output_dir = os.path.dirname(processed_data_path)

    if split_type == "PBE":
        # 90:10 Random Split
        split_idx = int(0.9 * n_samples)
        train_set = dataset[:split_idx]
        val_set = dataset[split_idx:]
        
        torch.save(train_set, os.path.join(output_dir, f"PBE_train_Rc{Rc}.pt"))
        torch.save(val_set, os.path.join(output_dir, f"PBE_val_Rc{Rc}.pt"))
        print(f"📊 {split_type} Splits Created: Train={len(train_set)}, Val={len(val_set)}")

    elif split_type == "mBJ":
        # 80:10:10 Random Split
        train_idx = int(0.8 * n_samples)
        val_idx = int(0.9 * n_samples)
        
        train_set = dataset[:train_idx]
        val_set = dataset[train_idx:val_idx]
        test_set = dataset[val_idx:]
        
        torch.save(train_set, os.path.join(output_dir, f"mBJ_train_Rc{Rc}.pt"))
        torch.save(val_set, os.path.join(output_dir, f"mBJ_val_Rc{Rc}.pt"))
        torch.save(test_set, os.path.join(output_dir, f"mBJ_test_Rc{Rc}.pt"))
        print(f"📊 {split_type} Splits Created: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")


if __name__ == "__main__":
    # GitHub relative file structures
    # Users should populate data/raw_pbe/ and data/raw_mbj/ with database downloads
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # -------------------------------------------------------------
    # Example Database dictionaries mapping: {material_id: bandgap_value}
    # These dictionaries are populated from JARVIS/Materials Project queries
    # -------------------------------------------------------------
    dummy_pbe_metadata = {}  # Populate with your real AFLOW/Materials Project metadata mappings
    dummy_mbj_metadata = {}  # Populate with your real JARVIS-mBJ metadata mappings
    
    # Pre-processing PBE
    PBE_RAW_DIR = os.path.join(BASE_DIR, "data", "raw_pbe")
    PBE_PROCESSED_FILE = os.path.join(BASE_DIR, "data", "processed", f"PBE_full_graphs_Rc{Rc}.pt")
    if os.path.exists(PBE_RAW_DIR):
        process_cif_directory(PBE_RAW_DIR, dummy_pbe_metadata, PBE_PROCESSED_FILE)
        split_and_save_data(PBE_PROCESSED_FILE, split_type="PBE")
    
    # Pre-processing mBJ
    mBJ_RAW_DIR = os.path.join(BASE_DIR, "data", "raw_mbj")
    mBJ_PROCESSED_FILE = os.path.join(BASE_DIR, "data", "processed", f"mBJ_full_graphs_Rc{Rc}.pt")
    if os.path.exists(mBJ_RAW_DIR):
        process_cif_directory(mBJ_RAW_DIR, dummy_mbj_metadata, mBJ_PROCESSED_FILE)
        split_and_save_data(mBJ_PROCESSED_FILE, split_type="mBJ")
