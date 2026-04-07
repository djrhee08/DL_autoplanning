import os
import glob
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from pathlib import Path

# Import model and loss function defined in DoseCalculator_Attention.py
from DoseCalculator_Attention import VMATDosePredictorAttention, PhysicsInformedDoseLoss
from MLC2Aperture import DifferentiableMLCAperture, DifferentiableJawAperture

# =====================================================================
# 1. Custom Dataset & DataLoader (auto-parse file tree)
# =====================================================================
class VMATDataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir: top-level directory containing patient subdirectories (e.g. './data')
        """
        self.samples = []

        # Recursively search all subdirectories for _CT.npy files
        ct_files = glob.glob(os.path.join(root_dir, '**', '*_CT.npy'), recursive=True)
        print(root_dir)
        print(ct_files)

        for ct_path in ct_files:
            # Extract prefix (e.g. test_vmat_3arc_A 1)
            prefix = ct_path.replace('_CT.npy', '')

            dose_path = prefix + '_dose.npy'

            # MLC and jaw filenames have a trailing tag (e.g. _odd_start181), so use wildcard
            mlc_files = glob.glob(prefix + '_mlc_*.npy')
            jaw_files = glob.glob(prefix + '_jaw_*.npy')

            # Verify all four files exist as a matched set
            if os.path.exists(dose_path) and mlc_files and jaw_files:
                mlc_path = mlc_files[0]
                jaw_path = jaw_files[0]

                # Infer rotation direction (CW/CCW) from filename
                is_cw = True if 'start181' in mlc_path else False

                self.samples.append({
                    'ct': ct_path,
                    'dose': dose_path,
                    'mlc': mlc_path,
                    'jaw': jaw_path,
                    'is_cw': is_cw,
                    'name': os.path.basename(prefix)
                })

        print(f"Total valid arc samples found: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load numpy arrays
        ct = np.load(sample['ct'])       # [192, 192, 192]
        dose = np.load(sample['dose'])   # [192, 192, 192]
        mlc = np.load(sample['mlc'])     # [180, 60, 2]  raw leaf positions (mm), X1/X2 per leaf pair
        jaw = np.load(sample['jaw'])     # [180, 2, 2]   jaw boundaries (mm): [:,0,:]=[X1,X2], [:,1,:]=[Y1,Y2]

        # Add channel dimension to match PyTorch 3D CNN input format
        ct = np.expand_dims(ct, axis=0)      # [1, 192, 192, 192]
        dose = np.expand_dims(dose, axis=0)  # [1, 192, 192, 192]

        return {
            'ct':  torch.from_numpy(ct).float(),
            'dose': torch.from_numpy(dose).float(),
            'mlc': torch.from_numpy(mlc).float(),   # [180, 60, 2]
            'jaw': torch.from_numpy(jaw).float(),   # [180, 2, 2]
            'is_cw': sample['is_cw'],
            'name': sample['name']
        }


# =====================================================================
# 2. Main Training & Validation Loop
# =====================================================================
def main():
    # Hyperparameters
    current_dir = Path(__file__).resolve().parent
    data_dir = os.path.join(current_dir, 'data/processed')  # top-level data directory path
    batch_size = 1       # batch size of 1 recommended due to 3D VRAM constraints
    num_epochs = 100
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and split data (Train 70%, Val 15%, Test 15%)
    dataset = VMATDataset(root_dir=data_dir)
    total_size = len(dataset)
    
    train_size=4
    val_size=test_size=1
    """
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    """
    print(len(dataset))
    print(train_size, val_size, test_size)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # fixed seed for reproducibility
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model, aperture layers, and optimizer
    mlc_layer = DifferentiableMLCAperture(pixel_size=2.5, grid_size=160, tau=0.5).to(device)
    jaw_layer = DifferentiableJawAperture(pixel_size=2.5, grid_size=160, tau=0.5).to(device)
    model = VMATDosePredictorAttention().to(device)
    criterion = PhysicsInformedDoseLoss(alpha=1.0, beta=0.5).to(device)  # L1 + Gradient Penalty
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    save_dir = './checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    print("Starting Training...")
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0

        # Display progress bar with tqdm
        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Train")
        for batch in train_pbar:
            ct       = batch['ct'].to(device)
            dose_true = batch['dose'].to(device)
            mlc_pos  = batch['mlc'].to(device)   # [B, 180, 60, 2]
            jaw      = batch['jaw'].to(device)   # [B, 180, 2, 2]

            optimizer.zero_grad()

            B = mlc_pos.shape[0]
            mlc_aperture = mlc_layer(mlc_pos.view(B * 180, 60, 2)).view(B, 180, 1, 160, 160)
            jaw_aperture = jaw_layer(jaw.view(B * 180, 2, 2)).view(B, 180, 1, 160, 160)
            bev = torch.cat([jaw_aperture, mlc_aperture], dim=2)  # [B, 180, 2, 160, 160]

            mlc_npy = mlc_aperture.detach().cpu().numpy().squeeze()
            jaw_npy = jaw_aperture.detach().cpu().numpy().squeeze()

            np.save(os.path.join(current_dir, 'mlc.npy'), mlc_npy)
            np.save(os.path.join(current_dir, 'jaw.npy'), jaw_npy)

            dose_pred, _ = model(ct, bev)

            loss = criterion(dose_pred, dose_true)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Val  ")
            for batch in val_pbar:
                ct       = batch['ct'].to(device)
                dose_true = batch['dose'].to(device)
                mlc_pos  = batch['mlc'].to(device)
                jaw      = batch['jaw'].to(device)

                B = mlc_pos.shape[0]
                mlc_aperture = mlc_layer(mlc_pos.view(B * 180, 60, 2)).view(B, 180, 1, 160, 160)
                jaw_aperture = jaw_layer(jaw.view(B * 180, 2, 2)).view(B, 180, 1, 160, 160)
                bev = torch.cat([jaw_aperture, mlc_aperture], dim=2)

                dose_pred, _ = model(ct, bev)
                loss = criterion(dose_pred, dose_true)

                val_loss += loss.item()
                val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_vmat_attn_model.pth'))
            print(f" -> Best model saved at epoch {epoch+1}")

    # --- Test Phase ---
    print("\nTraining Complete. Running Test Phase on unseen data...")
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_vmat_attn_model.pth')))
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            ct       = batch['ct'].to(device)
            dose_true = batch['dose'].to(device)
            mlc_pos  = batch['mlc'].to(device)
            jaw      = batch['jaw'].to(device)

            B = mlc_pos.shape[0]
            mlc_aperture = mlc_layer(mlc_pos.view(B * 180, 60, 2)).view(B, 180, 1, 160, 160)
            jaw_aperture = jaw_layer(jaw.view(B * 180, 2, 2)).view(B, 180, 1, 160, 160)
            bev = torch.cat([jaw_aperture, mlc_aperture], dim=2)

            dose_pred, _ = model(ct, bev)
            loss = criterion(dose_pred, dose_true)
            test_loss += loss.item()

            # Test results can be exported to .npy or DICOM-RT here if needed

    avg_test_loss = test_loss / len(test_loader)
    print(f"Final Test Loss: {avg_test_loss:.4f}")

if __name__ == '__main__':
    main()
