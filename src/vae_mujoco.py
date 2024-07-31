import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 15 * 20, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim * 2)  # 2 for mean and log variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * 15 * 20),
            nn.ReLU(),
            nn.Unflatten(1, (128, 15, 20)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h[:, :self.latent_dim], h[:, self.latent_dim:]
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


def train(model, train_loader, optimizer, epoch, device, checkpoint_path):
    model.train()
    train_loss = 0
    for imgs in train_loader:
        imgs = imgs.to(device)
        optimizer.zero_grad()
        recon_imgs, mu, logvar = model(imgs)
        loss = vae_loss(recon_imgs, imgs, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    avg_loss = train_loss / len(train_loader.dataset)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    save_checkpoint(model, optimizer, epoch + 1, avg_loss, checkpoint_path)


def test(model, val_loader, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for imgs in val_loader:
            imgs = imgs.to(device)
            recon_imgs, mu, logvar = model(imgs)
            loss = vae_loss(recon_imgs, imgs, mu, logvar)
            test_loss += loss.item()
    avg_loss = test_loss / len(val_loader.dataset)
    print(f'Epoch {epoch + 1}, Test Loss: {avg_loss:.4f}')


def unnormalize_image(image):
    image = image.cpu().numpy()
    image = image.transpose(1, 2, 0)
    image = image * 255

    if np.any(np.isnan(image)):
        print("Found NaNs in the image array. Replacing with 0.")
        image = np.nan_to_num(image, nan=0)

    if np.any(np.isinf(image)):
        print("Found Infinities in the image array. Replacing with max uint8 value.")
        image = np.nan_to_num(image, posinf=255, neginf=0)

    return image.astype('uint8')


def generate_and_save_samples(model, num_samples, latent_dim, device, save_dir='generated_samples'):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        generated_images = model.decoder(z).cpu()

        for i in range(num_samples):
            image = generated_images[i]
            unnormalized_image = unnormalize_image(image)
            image_pil = Image.fromarray(unnormalized_image)
            image_path = os.path.join(save_dir, f'generated_image_{i + 1}.png')
            image_pil.save(image_path)
            print(f'Saved {image_path}')


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded: start from epoch {epoch}, loss {loss:.4f}")
        return epoch, loss
    else:
        print("No checkpoint found at specified path")
        return 0, None


if __name__ == '__main__':
    # Hyperparameters
    learning_rate = 0.001
    batch_size = 16
    num_epochs = 100
    latent_dim = 30
    validation_split = 0.2
    resize_ratio = 0.25
    checkpoint_path = 'vae_mujoco_model.pth'

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((int(480 * resize_ratio), int(640 * resize_ratio))),
        transforms.ToTensor(),
    ])

    # Load dataset
    dataset = CustomImageDataset(root_dir='data/mujoco/rgb', transform=transform)
    dataset_size = len(dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Training and testing loops
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load checkpoint if available
    start_epoch, _ = load_checkpoint(checkpoint_path, model, optimizer)

    # Run the training and testing loops
    for epoch in range(start_epoch, num_epochs):
        train(model, train_loader, optimizer, epoch, device, checkpoint_path)
        test(model, val_loader, device)

    # Generate new samples
    generate_and_save_samples(model, num_samples=10, latent_dim=latent_dim, device=device)
