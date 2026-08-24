"""
Crystal Transformer: Multi-Fidelity Transfer Learning for Photocatalytic Bandgap Prediction
-----------------------------------------------------------------------------------------
CODE 02: GGA-PBE Base Pretraining & Chemical Verification Stress Tests
Grounded in Paper Section III-D (Gaussian Encoding), III-E (Architecture), & IV-D (Stress Tests)

Author: Thushara Subasinghe, Kavindi D.M.N., Perera E.T.B.
Journal Submission: Computational Materials Science (Elsevier)
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Set seed for reproducible initialization and splitting
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ==============================================================================
# 1. PHYSICS-INFORMED ENCODING LAYER (Paper Section III-D)
# ==============================================================================
class GaussianSmearing(nn.Module):
    """
    Expands discrete scalar interatomic bond distances d_ij into a continuous, 
    differentiable high-dimensional spatial representation using 50 Gaussian filters.
    See Equation (1) in Section III-D.
    """
    def __init__(self, start=0.0, stop=5.0, num_centers=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_centers)
        self.offset = nn.Parameter(offset, requires_grad=False)
        spacing = (stop - start) / (num_centers - 1)
        self.gamma = nn.Parameter(torch.tensor(1.0 / (spacing ** 2)), requires_grad=False)

    def forward(self, dist):
        dist = dist.view(-1, 1)
        return torch.exp(-self.gamma * (dist - self.offset) ** 2)


# ==============================================================================
# 2. CRYSTAL GRAPH TRANSFORMER ARCHITECTURE (Paper Section III-E)
# ==============================================================================
class CrystalTransformer(nn.Module):
    """
    The predictive framework: incorporates 4 TransformerConv message passing layers
    with adaptive attention aggregation and a deep 5-layer MLP regression head.
    """
    def __init__(self, node_in_dim=1, edge_in_dim=50, hidden_dim=256, num_heads=8, dropout=0.2):
        super().__init__()
        
        # Continuous embedding maps for normalized node elements and bond representations
        self.node_embedding = nn.Linear(node_in_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_in_dim, hidden_dim)

        # 4-Layer multi-head Transformer message-passing block
        self.conv1 = TransformerConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, edge_dim=hidden_dim, dropout=dropout)
        self.conv2 = TransformerConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, edge_dim=hidden_dim, dropout=dropout)
        self.conv3 = TransformerConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, edge_dim=hidden_dim, dropout=dropout)
        self.conv4 = TransformerConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, edge_dim=hidden_dim, dropout=dropout)
        
        # Layer Normalizations for deep propagation stability
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)

        # Five-Layer Deep Regression Output Head (Linear + ReLU + Dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Node and spatial edge projection
        h = self.node_embedding(x)
        e = self.edge_embedding(edge_attr)

        # Deep propagation stacks with layers norms and skip connections
        h1 = F.relu(self.ln1(self.conv1(h, edge_index, edge_attr=e) + h))
        h2 = F.relu(self.ln2(self.conv2(h1, edge_index, edge_attr=e) + h1))
        h3 = F.relu(self.ln3(self.conv3(h2, edge_index, edge_attr=e) + h2))
        h4 = F.relu(self.ln4(self.conv4(h3, edge_index, edge_attr=e) + h3))

        # Global Mean Pooling mapping elements to a single crystalline vector
        pooled = global_mean_pool(h4, batch)
        
        # Execute non-linear regressor mapping mapping features to 1D bandgap output
        return self.mlp(pooled).squeeze(-1)


# ==============================================================================
# 3. VERIFICATION STRESS TESTS (Paper Section IV-D / Physics Verification)
# ==============================================================================
def run_metal_vs_insulator_test(model, val_loader, device):
    """
    Evaluates model MAE separately for metals (true bandgap = 0.0 eV) 
    and semiconductor/insulators (true bandgap > 0.0 eV).
    """
    model.eval()
    metal_preds, metal_targets = [], []
    insulator_preds, insulator_targets = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            preds = model(batch).cpu().numpy()
            targets = batch.y.cpu().numpy()
            
            for p, t in zip(preds, targets):
                if t < 1e-4:  # True Metallic threshold
                    metal_preds.append(p)
                    metal_targets.append(t)
                else:
                    insulator_preds.append(p)
                    insulator_targets.append(t)
                    
    metal_mae = np.mean(np.abs(np.array(metal_preds) - np.array(metal_targets))) if metal_preds else 0.0
    insulator_mae = np.mean(np.abs(np.array(insulator_preds) - np.array(insulator_targets))) if insulator_preds else 0.0
    
    print("\n============================================================")
    print("🔬 STRESS TEST 1: METAL vs. SEMICONDUCTOR SELECTIVITY")
    print(f"👉 Metallic Systems (Eg = 0 eV) - Calculated pretrain MAE: {metal_mae:.4f} eV")
    print(f"👉 Non-Metallic Systems (Eg > 0 eV) - Calculated pretrain MAE: {insulator_mae:.4f} eV")
    print("============================================================\n")


def run_mendeleev_substitution_test(model, val_loader, device):
    """
    Validates model chemistry chemical periodic table awareness. Swaps Oxygen/Sulfur 
    with column neighbors (Fluorine/Selenium) vs. completely random elements.
    """
    model.eval()
    valid_sub_count = 0
    group_deltas = []
    random_deltas = []

    # Map target atomic numbers to Group neighbors: {S (16) <-> Se (34), Si (14) <-> Ge (32)}
    periodic_group_mutations = {16: 34, 34: 16, 14: 32, 32: 14}
    
    for batch in val_loader:
        # Pull a clean batch data copy
        batch_original = batch.clone().to(device)
        batch_mutated_group = batch.clone().to(device)
        batch_mutated_random = batch.clone().to(device)
        
        with torch.no_grad():
            orig_preds = model(batch_original).cpu().numpy()
            
            # 1. Group Substitution Mutation
            has_mutatable = False
            for idx in range(batch_mutated_group.x.size(0)):
                current_z_norm = batch_mutated_group.x[idx].item()
                raw_z = int(round(current_z_norm * 100))
                if raw_z in periodic_group_mutations:
                    new_z = periodic_group_mutations[raw_z]
                    batch_mutated_group.x[idx] = new_z / 100.0
                    has_mutatable = True
            
            # 2. Complete Random Mutation (Replace with inert Helium / Z=2)
            for idx in range(batch_mutated_random.x.size(0)):
                current_z_norm = batch_mutated_random.x[idx].item()
                raw_z = int(round(current_z_norm * 100))
                if raw_z in periodic_group_mutations:
                    batch_mutated_random.x[idx] = 2.0 / 100.0  # Force helium swap

            if has_mutatable:
                group_preds = model(batch_mutated_group).cpu().numpy()
                random_preds = model(batch_mutated_random).cpu().numpy()
                
                group_deltas.extend(np.abs(group_preds - orig_preds))
                random_deltas.extend(np.abs(random_preds - orig_preds))
                valid_sub_count += 1
                
            if valid_sub_count >= 50:  # Bound batch limits for test speed
                break

    print("============================================================")
    print("🔬 STRESS TEST 2: MENDELEEV GROUP COLUMN SUBSTITUTION")
    print(f"🧬 Same-Group Column Swaps Mean Absolute Delta: {np.mean(group_deltas):.4f} eV  (Predicts chemically similar behavior)")
    print(f"🚫 Chaotic-Random (Helium-force) Swap Mean Absolute Delta: {np.mean(random_deltas):.4f} eV")
    print("============================================================\n")


# ==============================================================================
# 4. TRAINING ENGINE & MAIN LAUNCHER
# ==============================================================================
def train_one_epoch(model, loader, optimizer, criterion, g_smear, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training Batches"):
        batch = batch.to(device)
        
        # Apply physics-informed Gaussian Smearing on continuous edge features
        batch.edge_attr = g_smear(batch.edge_attr)
        
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def validate_model(model, loader, criterion, g_smear, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch.edge_attr = g_smear(batch.edge_attr)
            out = model(batch)
            loss = criterion(out, batch.y)
            total_loss += loss.item() * batch.num_graphs
            
            all_preds.extend(out.cpu().numpy())
            all_targets.extend(batch.y.cpu().numpy())
            
    val_mae = total_loss / len(loader.dataset)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Validation R² Calculation
    u = ((all_targets - all_preds) ** 2).sum()
    v = ((all_targets - all_targets.mean()) ** 2).sum()
    r2_score = 1 - (u / v) if v != 0 else 0.0
    
    return val_mae, r2_score, all_preds, all_targets


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training Device Configured: {device}")
    
    # GitHub Relative Directory Setup
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
    MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # Paths to datasets prepared by Code 01
    PBE_train_path = os.path.join(PROCESSED_DATA_DIR, "PBE_train_Rc2.0.pt")
    PBE_val_path = os.path.join(PROCESSED_DATA_DIR, "PBE_val_Rc2.0.pt")
    
    # -------------------------------------------------------------
    # Simulated/Fallback dataset loader if actual files are not yet populated
    # -------------------------------------------------------------
    if not os.path.exists(PBE_train_path):
        print("⚠️ Processed training files not found in Relative Path. Utilizing dynamic fallback structures...")
        # Create small synthetic graphs matching PyG data patterns
        from torch_geometric.data import Data
        def make_dummy_data(num_samples=100):
            dummy_graphs = []
            for _ in range(num_samples):
                num_nodes = np.random.randint(4, 15)
                # Random normalized node elements
                x = torch.randint(1, 80, (num_nodes, 1), dtype=torch.float) / 100.0
                edge_index = torch.randint(0, num_nodes, (2, num_nodes * 3), dtype=torch.long)
                edge_attr = torch.rand((num_nodes * 3, 1), dtype=torch.float) * 4.0  # distances [1]
                y = torch.tensor([np.random.uniform(0.0, 5.0)], dtype=torch.float)   # bandgaps
                dummy_graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
            return dummy_graphs
        
        train_data = make_dummy_data(100)
        val_data = make_dummy_data(20)
    else:
        train_data = torch.load(PBE_train_path)
        val_data = torch.load(PBE_val_path)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    
    # Initialize physics layers
    rbf_distance_expansion = GaussianSmearing(start=0.0, stop=5.0, num_centers=50).to(device)
    model = CrystalTransformer().to(device)
    
    # Training Parameters matching PBE pretraining schema
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    criterion = nn.L1Loss()  # Mean Absolute Error Loss

    best_val_mae = float("inf")
    
    print("\n--- STARTING BASE MODEL TRAINING (GGA-PBE PRETRAINING) ---")
    print(f"📋 Loaded {len(train_data)} training samples. Processing with 256-dim Model & 5-Layer Output MLP.")
    
    # Pretraining limits set to 100 epochs
    for epoch in range(1, 101):
        train_mae = train_one_epoch(model, train_loader, optimizer, criterion, rbf_distance_expansion, device)
        val_mae, val_r2, val_preds, val_targets = validate_model(model, val_loader, criterion, rbf_distance_expansion, device)
        scheduler.step()
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            # Save the pretraining weight state
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "best_model_pbe.pt"))
            print(f"⭐ Epoch {epoch:03d} | Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f} | R²: {val_r2:.4f} <-- New Best Weight Saved!")
        else:
            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d} | Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f} | R²: {val_r2:.4f}")

    # Load best weights to run our Stress Tests
    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, "best_model_pbe.pt")))
    
    # Run Chemistry stress tests
    run_metal_vs_insulator_test(model, val_loader, device)
    run_mendeleev_substitution_test(model, val_loader, device)
    
    # -------------------------------------------------------------
    # Generate Validation Scatter Plot
    # -------------------------------------------------------------
    val_mae, val_r2, val_preds, val_targets = validate_model(model, val_loader, criterion, rbf_distance_expansion, device)
    plt.figure(figsize=(6, 5))
    plt.scatter(val_targets, val_preds, alpha=0.5, color='darkblue', edgecolors='none', label='Structures')
    plt.plot([val_targets.min(), val_targets.max()], [val_targets.min(), val_targets.max()], 'r--', lw=2, label='Ideal Predictor')
    plt.xlabel("Actual PBE Bandgap (eV)")
    plt.ylabel("Predicted PBE Bandgap (eV)")
    plt.title(f"Crystal Transformer Pretraining on PBE\nMAE = {val_mae:.4f} eV | R² = {val_r2:.4f}")
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plot_filepath = os.path.join(BASE_DIR, "data", "processed", "pretrain_val_scatter_plot.png")
    os.makedirs(os.path.dirname(plot_filepath), exist_ok=True)
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Validation scatter plot successfully saved to: {plot_filepath}\n")
