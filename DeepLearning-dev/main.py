import os
import glob
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from pathlib import Path

from DoseCalculator_Attention import VMATDosePredictorAttention, PhysicsInformedDoseLoss
from MLC2Aperture import DifferentiableMLCAperture, DifferentiableJawAperture

# =====================================================================
# 1. Custom Dataset & DataLoader
# =====================================================================
class VMATDataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir: top-level directory containing patient subdirectories.

        Expects per-beam files produced by preprocessing-dev/main_total.py:
            {prefix}_CT.npy    [192, 192, 192]
            {prefix}_dose.npy  [192, 192, 192]
            {prefix}_mlc.npy   [180, 60, 2]  raw MLC leaf positions (mm)
            {prefix}_jaw.npy   [180,  2, 2]  raw jaw positions (mm)
        """
        self.samples = []

        ct_files = glob.glob(os.path.join(root_dir, '**', '*_CT.npy'), recursive=True)
        print(root_dir)
        print(ct_files)

        for ct_path in ct_files:
            prefix     = ct_path.replace('_CT.npy', '')
            dose_path  = prefix + '_dose.npy'

            # Support both new naming (*_mlc.npy) and legacy (*_mlc_*.npy)
            mlc_files = glob.glob(prefix + '_mlc.npy') + glob.glob(prefix + '_mlc_*.npy')
            jaw_files = glob.glob(prefix + '_jaw.npy') + glob.glob(prefix + '_jaw_*.npy')
            mu_files  = glob.glob(prefix + '_mu.npy')

            if os.path.exists(dose_path) and mlc_files and jaw_files and mu_files:
                self.samples.append({
                    'ct':   ct_path,
                    'dose': dose_path,
                    'mlc':  mlc_files[0],
                    'jaw':  jaw_files[0],
                    'mu':   mu_files[0],
                    'name': os.path.basename(prefix)
                })

        print(f"Total valid arc samples found: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        ct   = np.load(sample['ct'])              # [192, 192, 192]
        dose = np.load(sample['dose'])            # [192, 192, 192]
        mlc  = np.load(sample['mlc'])             # [180, 60, 2]
        jaw  = np.load(sample['jaw'])             # [180,  2, 2]
        mu   = np.load(sample['mu']).squeeze()    # [180, 1, 1] → [180]

        ct   = np.expand_dims(ct,   axis=0)       # [1, 192, 192, 192]
        dose = np.expand_dims(dose, axis=0)       # [1, 192, 192, 192]

        return {
            'ct':   torch.from_numpy(ct).float(),
            'dose': torch.from_numpy(dose).float(),
            'mlc':  torch.from_numpy(mlc).float(),   # [180, 60, 2]
            'jaw':  torch.from_numpy(jaw).float(),   # [180,  2, 2]
            'mu':   torch.from_numpy(mu).float(),    # [180]  — caller uses [:179]
            'name': sample['name']
        }


# =====================================================================
# 2. Main Training & Validation Loop
# =====================================================================
def main():
    current_dir   = Path(__file__).resolve().parent
    data_dir      = os.path.join(current_dir, '../preprocessing-dev/npy_total/')
    batch_size    = 1
    num_epochs    = 200
    learning_rate = 1e-3
    device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset    = VMATDataset(root_dir=data_dir)
    total_size = len(dataset)

    train_size = int(0.7 * total_size)
    val_size   = int(0.15 * total_size)
    test_size  = total_size - train_size - val_size

    print(len(dataset))
    print(train_size, val_size, test_size)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    # Aperture layers: average=True uses RayStation VMAT convention (179 mid-segment apertures)
    mlc_layer = DifferentiableMLCAperture(pixel_size=2.5, grid_size=160, tau=0.5).to(device)
    jaw_layer = DifferentiableJawAperture(pixel_size=2.5, grid_size=160, tau=0.5).to(device)

    # Model built with average=True must receive [B, 179, 2, 160, 160] BEV input
    model     = VMATDosePredictorAttention(average=True).to(device)
    criterion = PhysicsInformedDoseLoss(alpha=1.0, beta=0.5).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    save_dir      = os.path.join(current_dir, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    def build_bev(mlc_pos, jaw):
        """
        mlc_pos : [B, 180, 60, 2]
        jaw     : [B, 180,  2, 2]
        Returns : [B, 179, 2, 160, 160]  – averaged adjacent-CP apertures (normalised 0–1)
        MU scaling is applied inside model.forward(), not here, so that during
        autoplanning the MU gradients flow directly through the model.
        """
        mlc_ap = mlc_layer(mlc_pos, average=True)   # [B, 179, 1, 160, 160]
        jaw_ap = jaw_layer(jaw,     average=True)   # [B, 179, 1, 160, 160]
        return torch.cat([jaw_ap, mlc_ap], dim=2)   # [B, 179, 2, 160, 160]

    print("Starting Training...")
    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Train")
        for batch in train_pbar:
            ct        = batch['ct'].to(device)
            dose_true = batch['dose'].to(device)
            mlc_pos   = batch['mlc'].to(device)        # [B, 180, 60, 2]
            jaw       = batch['jaw'].to(device)        # [B, 180,  2, 2]
            mu        = batch['mu'][:, :179].to(device) # [B, 179]

            optimizer.zero_grad()

            bev       = build_bev(mlc_pos, jaw)        # [B, 179, 2, 160, 160]
            dose_pred, _ = model(ct, bev, mu)

            loss = criterion(dose_pred, dose_true)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Val  ")
            for batch in val_pbar:
                ct        = batch['ct'].to(device)
                dose_true = batch['dose'].to(device)
                mlc_pos   = batch['mlc'].to(device)
                jaw       = batch['jaw'].to(device)
                mu        = batch['mu'][:, :179].to(device)

                bev       = build_bev(mlc_pos, jaw)
                dose_pred, _ = model(ct, bev, mu)
                loss      = criterion(dose_pred, dose_true)

                val_loss += loss.item()
                val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(),
                       os.path.join(save_dir, 'best_vmat_attn_model.pth'))
            print(f" -> Best model saved at epoch {epoch+1}")

    # --- Test ---
    print("\nTraining Complete. Running Test Phase...")
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_vmat_attn_model.pth')))
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            ct        = batch['ct'].to(device)
            dose_true = batch['dose'].to(device)
            mlc_pos   = batch['mlc'].to(device)
            jaw       = batch['jaw'].to(device)
            mu        = batch['mu'][:, :179].to(device)

            bev       = build_bev(mlc_pos, jaw)
            dose_pred, _ = model(ct, bev, mu)
            loss      = criterion(dose_pred, dose_true)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Final Test Loss: {avg_test_loss:.4f}")


if __name__ == '__main__':
    main()
