"""
Entrenamiento local en Windows (RTX 2050, 4GB VRAM).
Requiere el dataset.csv con columnas:
image_path, x, y, z, q0, q1, q2, q3, route_type, trajectory_id, split
"""

import os
import sys
import time
import csv
from datetime import datetime
from typing import Tuple

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image

# ------------------------------------------------------------------
# Configuracion
# ------------------------------------------------------------------
DATASET_ROOT = r"C:\dataset_3"
LOGS_ROOT = r"C:\cats_codes"
CSV_PATH = os.path.join(DATASET_ROOT, "dataset.csv")

LOG_DIR = os.path.join(LOGS_ROOT, "Training_4")
CHECKPOINT_DIR = os.path.join(LOG_DIR, "checkpoints")
LOG_PATH = os.path.join(LOG_DIR, "training_log.txt")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

# Limites reales de la habitacion
X_MIN, X_MAX = -1.25, 8.75
Y_MIN, Y_MAX = -8.0, 2.0
Z_MIN, Z_MAX = 0.134733498096466, 4.34491205215454

POS_MIN = torch.tensor([X_MIN, Y_MIN, Z_MIN], dtype=torch.float32)
POS_MAX = torch.tensor([X_MAX, Y_MAX, Z_MAX], dtype=torch.float32)
POS_RANGE = POS_MAX - POS_MIN

BATCH_SIZE = 8
NUM_EPOCHS = 60
LEARNING_RATE = 1e-4
BETA_ROT = 10.0
IMAGE_SIZE = 224
NUM_WORKERS = 4
EARLY_STOP_PATIENCE = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class PoseDataset(Dataset):
    def __init__(self, csv_path: str, root_dir: str, split: str = "train",
                 transform=None, normalize_pos: bool = True):
        self.root_dir = root_dir
        self.transform = transform
        self.normalize_pos = normalize_pos
        self.samples = []

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] != split:
                    continue
                full_image_path = os.path.join(root_dir, row["image_path"])
                pose = [
                    float(row["x"]), float(row["y"]), float(row["z"]),
                    float(row["q0"]), float(row["q1"]),
                    float(row["q2"]), float(row["q3"]),
                ]
                self.samples.append((full_image_path, pose, row["route_type"]))

        if len(self.samples) == 0:
            raise RuntimeError(f"No se encontraron muestras para split={split}")

        print(f"Split '{split}': {len(self.samples)} muestras")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, pose, _route = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        pose = torch.tensor(pose, dtype=torch.float32)

        if self.normalize_pos:
            pose[:3] = 2 * (pose[:3] - POS_MIN) / POS_RANGE - 1

        return img, pose

    def route_types(self):
        return [s[2] for s in self.samples]


def denormalize_pos(pos_norm: torch.Tensor) -> torch.Tensor:
    return (pos_norm + 1) / 2 * POS_RANGE.to(pos_norm.device) + POS_MIN.to(pos_norm.device)


class AddGaussianNoise:
    def __init__(self, std: float = 0.02):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noisy = tensor + self.std * torch.randn_like(tensor)
        return torch.clamp(noisy, 0.0, 1.0)


train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),
    transforms.ToTensor(),
    AddGaussianNoise(std=0.02),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.08)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ------------------------------------------------------------------
# Modelo y loss
# ------------------------------------------------------------------
class PoseNetLight(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.backbone_out_dim = backbone.fc.in_features

        self.fc = nn.Sequential(
            nn.Linear(self.backbone_out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 7),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def pose_loss(pred: torch.Tensor, target: torch.Tensor, beta: float = BETA_ROT):
    t_pred, q_pred = pred[:, :3], pred[:, 3:]
    t_gt, q_gt = target[:, :3], target[:, 3:]
    q_pred = q_pred / (torch.norm(q_pred, p=2, dim=1, keepdim=True) + 1e-8)

    # q y -q representan la misma rotacion (doble cobertura del cuaternion).
    # Sin esto, el MSE castiga como "totalmente equivocada" una prediccion
    # correcta que cayo del lado opuesto del signo
    dot = torch.sum(q_pred * q_gt, dim=1, keepdim=True)
    q_gt_aligned = torch.where(dot < 0, -q_gt, q_gt)

    t_loss = nn.functional.mse_loss(t_pred, t_gt)
    q_loss = nn.functional.mse_loss(q_pred, q_gt_aligned)
    return t_loss + beta * q_loss, t_loss.item(), q_loss.item()


def quaternion_angle_error_deg(q1, q2):
    q1 = q1 / (np.linalg.norm(q1) + 1e-8)
    q2 = q2 / (np.linalg.norm(q2) + 1e-8)
    dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
    return np.degrees(2 * np.arccos(abs(dot)))


# ------------------------------------------------------------------
# Loop de entrenamiento
# ------------------------------------------------------------------
def run_epoch(model, dataloader, device, optimizer=None, scaler=None,
              epoch_idx=0, num_epochs=1, tag="Train"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss, running_t, running_q = 0.0, 0.0, 0.0
    n_batches = len(dataloader)
    start = time.time()

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for i, (images, poses) in enumerate(dataloader):
            images, poses = images.to(device), poses.to(device)

            if is_train:
                optimizer.zero_grad()

            with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                outputs = model(images)
                loss, t_loss, q_loss = pose_loss(outputs, poses)

            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            running_loss += loss.item() * images.size(0)
            running_t += t_loss * images.size(0)
            running_q += q_loss * images.size(0)

            if (i % 20) == 0:
                elapsed = time.time() - start
                print(f"[{tag}][Epoca {epoch_idx}/{num_epochs}] Batch {i}/{n_batches}  "
                      f"Loss: {loss.item():.6f} (t={t_loss:.6f}, q={q_loss:.6f})  "
                      f"Tiempo: {elapsed:.1f}s")

    n = len(dataloader.dataset)
    epoch_loss, epoch_t, epoch_q = running_loss / n, running_t / n, running_q / n
    print(f"[{tag}] Epoca {epoch_idx} terminada en {time.time()-start:.1f}s  "
          f"Loss: {epoch_loss:.6f}  (t={epoch_t:.6f}, q={epoch_q:.6f})")
    return epoch_loss

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Modo "a" (append): si se reanuda el entrenamiento tras un corte, el log
    # sigue creciendo en el mismo archivo en vez de borrarse.
    log_file = open(LOG_PATH, "a", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    print(f"\n{'#'*70}")
    print(f"# Sesion de entrenamiento iniciada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}\n")

    print("Usando dispositivo:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_dataset = PoseDataset(CSV_PATH, DATASET_ROOT, split="train", transform=train_transform)
    val_dataset = PoseDataset(CSV_PATH, DATASET_ROOT, split="valid", transform=eval_transform)
    test_dataset = PoseDataset(CSV_PATH, DATASET_ROOT, split="test", transform=eval_transform)

    route_counts = pd.Series(train_dataset.route_types()).value_counts()
    print("Distribucion de rutas en train:\n", route_counts)

    weight_per_route = 1.0 / route_counts
    sample_weights = pd.Series(train_dataset.route_types()).map(weight_per_route).values
    train_sampler = WeightedRandomSampler(
        weights=np.asarray(sample_weights, dtype=np.float64),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
                               num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=(NUM_WORKERS > 0))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=(NUM_WORKERS > 0))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=(NUM_WORKERS > 0))

    model = PoseNetLight(pretrained=True).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    best_model_path = os.path.join(CHECKPOINT_DIR, "posenet_light_best_local.pth")
    last_model_path = os.path.join(CHECKPOINT_DIR, "posenet_light_last_local.pth")

    start_epoch = 1
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    if os.path.exists(last_model_path):
        print("Checkpoint encontrado, reanudando entrenamiento...")
        checkpoint = torch.load(last_model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
        print(f"  Reanudando desde epoca {start_epoch}, mejor val_loss hasta ahora: {best_val_loss:.6f}")
    else:
        print("No se encontro checkpoint previo, comenzando desde cero.")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print("=" * 60)
        print(f"Comenzando epoca {epoch}/{NUM_EPOCHS}  (lr={optimizer.param_groups[0]['lr']:.2e})")

        run_epoch(model, train_loader, DEVICE, optimizer, scaler, epoch, NUM_EPOCHS, "Train")
        val_loss = run_epoch(model, val_loader, DEVICE, None, None, epoch, NUM_EPOCHS, "Val")

        scheduler.step(val_loss)

        improved = val_loss < best_val_loss
        epochs_without_improvement = 0 if improved else epochs_without_improvement + 1

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "val_loss": val_loss,
                    "best_val_loss": min(val_loss, best_val_loss),
                    "epochs_without_improvement": epochs_without_improvement},
                   last_model_path)

        if improved:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "val_loss": val_loss},
                       best_model_path)
            print(f"  Nuevo mejor modelo guardado: {best_model_path}")
        else:
            print(f"  Sin mejora en val_loss ({epochs_without_improvement}/{EARLY_STOP_PATIENCE} epocas)")

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping: val_loss no mejoro en {EARLY_STOP_PATIENCE} epocas seguidas. "
                  f"Deteniendo entrenamiento en la epoca {epoch}.")
            break

    print("Entrenamiento terminado. Mejor val_loss:", best_val_loss)

    # Evaluacion final en test (por ruta)
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    best_model = PoseNetLight(pretrained=False).to(DEVICE)
    best_model.load_state_dict(checkpoint["model_state_dict"])
    best_model.eval()

    route_list = test_dataset.route_types()
    results = {"transl_err": [], "rot_err": [], "route": []}

    with torch.no_grad():
        idx = 0
        for images, poses_gt in test_loader:
            bsz = images.size(0)
            images, poses_gt = images.to(DEVICE), poses_gt.to(DEVICE)
            poses_pred = best_model(images)

            t_pred = denormalize_pos(poses_pred[:, :3]).cpu().numpy()
            t_gt = denormalize_pos(poses_gt[:, :3]).cpu().numpy()
            q_pred = poses_pred[:, 3:].cpu().numpy()
            q_gt = poses_gt[:, 3:].cpu().numpy()
            q_pred = q_pred / (np.linalg.norm(q_pred, axis=1, keepdims=True) + 1e-8)

            dist = np.linalg.norm(t_pred - t_gt, axis=1)
            for j in range(bsz):
                results["transl_err"].append(dist[j])
                results["rot_err"].append(quaternion_angle_error_deg(q_pred[j], q_gt[j]))
                results["route"].append(route_list[idx])
                idx += 1

    df_results = pd.DataFrame(results)
    print("\n=== Resultados en TEST por tipo de ruta ===")
    print(df_results.groupby("route")[["transl_err", "rot_err"]].agg(["mean", "median"]))
    print("\n=== Resultados globales en TEST ===")
    print(f"Error traslacion medio: {df_results['transl_err'].mean():.4f} m")
    print(f"Error rotacion medio:   {df_results['rot_err'].mean():.3f} grados")


if __name__ == "__main__":
    main()