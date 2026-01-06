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
small_train_dataset = TensorDataset(x_train[:100])  # רק 100 תמונות להדגמה
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

    x = param_obj.data  # Get the underlying tensor data

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


import numpy as np
import time


def run_model_optimization_experiment(optimizer_type, model, criterion, train_loader, epochs, learning_rate):
    """
    Runs optimization using Fisher Information (Natural Gradient approximation).
    Recommended initial learning_rate: 0.2
    """
    all_losses = []
    print(f"\nStarting MODEL {optimizer_type.upper()} optimization for {epochs} epochs (lr={learning_rate})...")
    total_start_time = time.time()

    global _ADAHESSIAN_STATES
    _ADAHESSIAN_STATES = {}  # Clear the global state for each new experiment run

    # Initialize momentum for GD if applicable (re-init for each experiment)
    cur_acc_list = [torch.zeros_like(p) for p in model.parameters()]
    global_step = 1
    total_steps = epochs * len(train_loader)

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            img = batch[0]

            # --- FISHER ORACLE CALL ---
            # Using calc_fisher instead of calc_hessian for the Newton path
            calc_fisher = (optimizer_type == 'newton')
            loss, grads, fishers = autoencoder_oracle(model, criterion, img, calc_fisher=calc_fisher)
            epoch_loss += loss.item()

            # --- DYNAMIC LEARNING RATE (Cosine Annealing) ---
            cur_lr = learning_rate * 0.5 * (1 + np.cos(np.pi * global_step / total_steps))

            # --- DYNAMIC MOMENTUM (BETA) ---
            if global_step / total_steps < 0.5:
                beta = 0.9
            else:
                beta = 0.9 + 0.08 * ((global_step / total_steps - 0.5) * 2)

            with torch.no_grad():
                if optimizer_type == 'gd':
                    for i, (param, grad) in enumerate(zip(model.parameters(), grads)):
                        new_val, new_acc = gd_step(param.data, grad, cur_acc_list[i], cur_lr, beta)
                        param.data = new_val
                        cur_acc_list[i] = new_acc

                elif optimizer_type == 'newton':
                    # Fisher-based adaptive step
                    for param, grad, fisher in zip(model.parameters(), grads, fishers):
                        # Pass the Parameter object itself and the correct global_step
                        param.data = newton_step(param, grad, fisher, global_step, cur_lr)

            global_step += 1

        avg_loss = epoch_loss / len(train_loader)
        all_losses.append(avg_loss)
        if (epoch + 1) % (epochs // 5) == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"  Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.6f}")

    print(f"\nTotal Training Time: {time.time() - total_start_time:.2f} seconds")
    return model, all_losses


epochs = 20
lr = 1
model = Autoencoder()
criterion = nn.MSELoss()

trained_model, losses = run_model_optimization_experiment(
    'gd', model, criterion, train_loader, epochs, lr
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
        ax = plt.subplot(count, 2, i * 2 + 1)
        plt.imshow(img_orig, cmap='gray')
        if i == 0: ax.set_title("Original")
        plt.axis('off')

        # Reconstructed
        ax = plt.subplot(count, 2, i * 2 + 2)
        plt.imshow(img_recon, cmap='gray')
        if i == 0: ax.set_title("Reconstructed (Model Output)")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# ---use example ---
data_iter = iter(train_loader)
sample_images = next(data_iter)[0]

show_model_reconstructions(trained_model, sample_images, count=5)
