import math
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
from model_with_time_embedding import Unet  # Импорт обновленной модели

# Гиперпараметры
batch_size = 64
learning_rate = 1e-4
epochs = 50
image_size = 128
num_timesteps = 1000  # Количество временных шагов для диффузии
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Подготовка датасета CelebA
transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.CelebA(root="./data", split="train", download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Инициализация модели, оптимизатора и функции потерь
model = Unet(num_features=64).to(device)  # num_features соответствует архитектуре модели
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()  # Используем MSE для предсказания шума

# Параметры диффузии
beta_schedule = torch.linspace(1e-4, 0.02, num_timesteps).to(device)  # Шаги бетта
alphas = 1.0 - beta_schedule
alphas_cumprod = torch.cumprod(alphas, dim=0)  # Кумулятивное произведение альфа
alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]])

# Функция зашумления
def add_noise(images, timestep, noise=None):
    if noise is None:
        noise = torch.randn_like(images).to(device)
    sqrt_alpha_cumprod = alphas_cumprod[timestep] ** 0.5
    sqrt_one_minus_alpha_cumprod = (1 - alphas_cumprod[timestep]) ** 0.5
    noisy_images = sqrt_alpha_cumprod * images + sqrt_one_minus_alpha_cumprod * noise
    return noisy_images, noise

# Функция обучения
def train_epoch(model, dataloader, optimizer, criterion, device, num_timesteps):
    model.train()
    epoch_loss = 0.0
    for images, _ in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        timesteps = torch.randint(0, num_timesteps, (images.size(0),), device=device).long()

        # Генерация шума и зашумление
        noisy_images, noise = add_noise(images, timesteps)

        optimizer.zero_grad()
        predicted_noise = model(noisy_images, timesteps)  # Передаем временной шаг в модель
        loss = criterion(predicted_noise, noise)  # Сравниваем предсказанный шум с истинным
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

# Цикл обучения
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train_loss = train_epoch(model, dataloader, optimizer, criterion, device, num_timesteps)
    print(f"Train Loss: {train_loss:.4f}")

    # Сохранение модели каждые 10 эпох
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"diffusion_model_epoch_{epoch + 1}.pth")
        print(f"Model saved at epoch {epoch + 1}")

# Финальное сохранение модели
torch.save(model.state_dict(), "diffusion_model_final.pth")
print("Training completed and model saved.")
