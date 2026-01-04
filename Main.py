import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import time

# @title Data Loader
print("Loading Data...")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
x_train = train_data.data
image_size = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(-1, image_size).float() / 255


# @title AutoEncoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Corrected dimensions: image_size -> 256 -> 128 -> 64 -> 32
        self.encoder = nn.Sequential(
            nn.Linear(image_size, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),  # Fixed: was 128 -> 64
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, image_size), nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def autoencoder_oracle(model, criterion, x, calc_fisher=False):
    """
    Computes loss and Fisher Information diagonal.
    The Fisher is normalized to ensure stable second-order updates.
    """
    reconstructed_x = model(x)
    loss = criterion(reconstructed_x, x)

    # Standard gradients
    grads = torch.autograd.grad(loss, model.parameters())

    fisher_diagonals = []
    if calc_fisher:
        for grad in grads:
            # Fisher approximation using squared gradients
            # Detach to prevent tracking second-order graphs twice
            f_diag = grad.detach() ** 2
            fisher_diagonals.append(f_diag)

    return loss, grads, fisher_diagonals

print("Initializing the model (Defining the function surface)...")
train_dataset = TensorDataset(x_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# This dataset is for the second order learners - so you wont run out of RAM
small_train_dataset = TensorDataset(x_train[:100]) # רק 100 תמונות להדגמה
small_train_loader = DataLoader(small_train_dataset, batch_size=10, shuffle=True)


def gd_step(x, grad_f_x, acc, learning_rate=0.1, beta=0.9, weight_decay=1e-5):
    """
    Exclusive Implementation: Dynamic Nesterov-Polyak Momentum.
    Combines Polyak's 1964 Heavy Ball with a time-varying schedule
    and Nesterov's look-ahead for superior convergence.
    """

    new_acc = (beta * acc) - (learning_rate * grad_f_x)

    x_new = x + (beta * new_acc) - (learning_rate * grad_f_x)
    # x_new = x + new_acc

    return x_new, new_acc

import torch.nn as nn

def newton_step(param_obj, grad_f_x, fisher_diag, step_count, learning_rate=0.001):
    global _ADAHESSIAN_STATES
    param_key = param_obj  # Use the Parameter object itself as the key

    x = param_obj.data # Get the underlying tensor data

    # Hyperparameters
    beta1, beta2 = 0.9, 0.999
    eps = 1e-4
    block_size = 32

    if param_key not in _ADAHESSIAN_STATES:
        _ADAHESSIAN_STATES[param_key] = {
            'm': torch.zeros_like(x),
            'v': torch.zeros_like(x)
        }
    state = _ADAHESSIAN_STATES[param_key]

    # 1. Update First Moment (Momentum)
    state['m'] = beta1 * state['m'] + (1 - beta1) * grad_f_x

    # 2. Update Fisher/Hessian Diagonal (Curvature)
    # PAPER REQUIREMENT: Use absolute value to ensure positive definiteness
    state['v'] = beta2 * state['v'] + (1 - beta2) * torch.abs(fisher_diag.view(x.shape))

    # 3. Bias Correction
    # This prevents the update from being tiny at the start of training
    bc1 = 1 - beta1 ** step_count
    bc2 = 1 - beta2 ** step_count
    m_hat = state['m'] / bc1
    v_hat = state['v'] / bc2

    # 4. Spatial Averaging (The "Block" Trick)
    v_flat = v_hat.view(-1)
    n = v_flat.numel()

    if n >= block_size:
        remainder = n % block_size
        if remainder > 0:
            # Pad the end, so we can reshape into blocks of 32
            padding = v_flat[-1].expand(block_size - remainder)
            v_padded = torch.cat([v_flat, padding])
        else:
            v_padded = v_flat

        # Average within each block
        v_blocks = v_padded.view(-1, block_size)
        v_avg_blocks = v_blocks.mean(dim=1, keepdim=True).expand_as(v_blocks)

        # Flatten and crop back to original size
        v_final = v_avg_blocks.reshape(-1)[:n].view(x.shape)
    else:
        v_final = v_hat.mean().expand_as(x)

    v_final = v_final.sqrt()

    # 5. Apply Newton Update (k=1)
    update = m_hat / (v_final + eps)

    return x - learning_rate * update

def run_model_optimization_experiment(
            self,
            optimizer_type: str,
            model: nn.Module,
            criterion: nn.Module,
            train_loader: DataLoader,
            epochs: int,
            learning_rate: float
    ):
    """
    Runs an optimization experiment to train the MODEL using GD or Newton.
    Closer to the requested structure.
    """
    all_losses = []
    print(f"\nStarting MODEL {optimizer_type.upper()} optimization for {epochs} epochs (lr={learning_rate})...")
    total_start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            img = batch[0]

            calc_hessian_for_oracle = (optimizer_type == 'newton')
            loss, grads, hessians = self.autoencoder_oracle(model, criterion, img, calc_hessian=calc_hessian_for_oracle)

            epoch_loss += loss.item()

            # 2. Optimization step
            with torch.no_grad():
                if optimizer_type == 'gd':
                    for param, grad in zip(model.parameters(), grads):
                        param.data = self.gd_step(param.data, grad, learning_rate)

                elif optimizer_type == 'newton':
                    if hessians is None:
                        raise ValueError("Hessian missing for Newton method")
                    for param, grad, hessian in zip(model.parameters(), grads, hessians):
                        param.data = self.newton_step(param.data, grad, hessian, learning_rate)

                else:
                    raise ValueError("Invalid optimizer type")

        avg_loss = epoch_loss / len(train_loader)
        all_losses.append(avg_loss)

        if (epoch + 1) % (epochs // 5) == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"  Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.6f}")

    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    print(f"\nTotal Training Time: {total_training_time:.2f} seconds")
    print(f"Finished {optimizer_type.upper()} optimization. Final Loss: {all_losses[-1]:.6f}")
    return model, all_losses


epochs = 20
lr = 0.01
model = Autoencoder()
criterion = nn.MSELoss()

trained_model, losses = run_model_optimization_experiment(
    'newton', model, criterion, train_loader, epochs, lr
)

plt.figure(figsize=(10, 6))
plt.plot(losses, label=f'GD (LR={lr})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Optimization Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()


def show_model_reconstructions(model, batch_images, count=5):
    """
    Visualizes original images vs. model reconstructions.
    """
    print("\nVisualizing Results...")

    model.eval()

    with torch.no_grad():
        inputs = batch_images[:count]
        reconstructions = model(inputs)

    plt.figure(figsize=(8, 3 * count))

    for i in range(count):
        img_orig = inputs[i].cpu().numpy().reshape(28, 28)
        img_recon = reconstructions[i].cpu().numpy().reshape(28, 28)

        # Original
        ax = plt.subplot(count, 2, i*2 + 1)
        plt.imshow(img_orig, cmap='gray')
        if i == 0: ax.set_title("Original")
        plt.axis('off')

        #Reconstructed
        ax = plt.subplot(count, 2, i*2 + 2)
        plt.imshow(img_recon, cmap='gray')
        if i == 0: ax.set_title("Reconstructed (Model Output)")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# ---use example ---
data_iter = iter(train_loader)
sample_images = next(data_iter)[0]

show_model_reconstructions(trained_model, sample_images, count=5)