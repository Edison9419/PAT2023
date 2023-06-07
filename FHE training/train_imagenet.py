import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import alexnet
from encrypt import encrypt_init, encrypt_cal
import time
from tqdm import tqdm


def train(model, dataloader, criterion, num_epochs, device, encryptor, evaluator, decryptor, learning_rate, rate, offset):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_start = time.time()
        for inputs, labels in tqdm(dataloader):
            batch_all_start = time.time()
            inputs, labels = inputs.to(device), labels.to(device)
            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            batch_start = time.time()
            with torch.no_grad():
                for param in model.parameters():
                    param_start = time.time()
                    origin_shape = param.shape
                    param_flat = param.flatten()
                    update = param.grad * learning_rate * -1
                    grad_flat = update.flatten()
                    for i in range(param_flat.size(0)):
                        # it_start = time.time()
                        param_flat[i] = encrypt_cal(encryptor, evaluator, decryptor, param_flat[i].item(), grad_flat[i].item(), rate, offset)
                        # it_end = time.time()
                        # print(f"Iter Encrypt Time:{it_end - it_start}")
                    param_end = time.time()
                    print(f"Param Encrypt Time:{param_end - param_start}")
                param = param_flat.reshape(origin_shape)
            batch_end = time.time()
            print(f"Batch Encrypt Time:{batch_end - batch_start}")
            running_loss += loss.item()
            batch_all_end = time.time()
            print(f"Batch All Time:{batch_all_end - batch_all_start}")
        epoch_end = time.time()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}, Epoch Time: {epoch_end - epoch_start}")


if __name__ == "__main__":
    batch_size = 128
    num_epochs = 100
    learning_rate = 0.001

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = torchvision.datasets.ImageFolder('/home/edison/imagenet1k/train/', transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = alexnet(num_classes=1000).to(device)

    encryptor, evaluator, decryptor = encrypt_init()
    train(model, train_dataloader, nn.CrossEntropyLoss(), num_epochs, device, encryptor, evaluator, decryptor, learning_rate, 1e4, 1)
