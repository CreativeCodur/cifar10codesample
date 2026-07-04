"""
CIFAR-10 image classifier using a ResNet-9 architecture.
Trained on 20,000 images, tested on 10,000.
Obviously not a 100x developer of any kind, but took some decent time researching on this and
built this after working through occasional AI-verficiation / help and a few resrorces (many more i forgot about):
- PyTorch official CIFAR10 tutorial: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
- Aladdin Persson's ResNet video: https://www.youtube.com/watch?v=DkNIBBABpA4
- The ResNet9 structure comes from David Page's DAWNBench experiments: https://myrtle.ai/learn/how-to-train-your-resnet/
- I used smaller channel sizes (32/64/128/256 instead of 64/128/256/512) because
  my laptop can't handle the full version and I wanted to run this on CPU in a reasonable time (not 70 days and 50 minutes like when I tried 11000 test images instead of 10000)

The normalization values (mean and std for each RGB channel) are standard for CIFAR-10,
I got them from: https://github.com/kuangliu/pytorch-cifar/issues/19

Things I tried that didn't work as well:
- SGD with momentum instead of AdamW (got ~79% after 15 epochs, way slower to converge)
- no data augmentation (accuracy dropped to like 75%, overfitting was bad)
- Dropout of 0.5 instead of 0.3 (underfitting, test acc was lower)
- Tried 64/128/256/512 channels but it was wayy  too slow on CPU
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
BATCH_SIZE = 128
LEARNING_RATE = 0.003
WEIGHT_DECAY = 1e-4
EPOCHS = 15
DEVICE = torch.device('cpu')

print(f"Using device: {DEVICE}")
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),    # randomly shifts the image around
    transforms.RandomHorizontalFlip(),        # flips it half the time
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])









test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])




# download cifar10 and only use 20k for training
full_train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
full_test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)






train_subset = Subset(full_train_set, list(range(20000)))
test_subset = Subset(full_test_set, list(range(10000)))



train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)






print(f"Training on {len(train_subset)} images, testing on {len(test_subset)} images")




# conv -> batchnorm -> relu, with optional pooling
# I kept using this pattern everywhere so made it a function
def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)





class ResNet9(nn.Module):
    def __init__(self):
        super().__init__()

        # each pooling layer halves the spatial size
        # 32x32 -> 16x16 -> 8x8 -> 4x4

        self.prep = conv_block(3, 32)

        self.layer1 = conv_block(32, 64, pool=True)       # 32x32 -> 16x16
        self.res1 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))

        self.layer2 = conv_block(64, 128, pool=True)       # 16x16 -> 8x8

        self.layer3 = conv_block(128, 256, pool=True)      # 8x8 -> 4x4
        self.res2 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),   # 4x4x256 -> 1x1x256
            nn.Flatten(),                    # -> 256
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )




    def forward(self, x):
        out = self.prep(x)
        out = self.layer1(out)
        out = out + self.res1(out)      # skip connection - this is the key idea from ResNet
        out = self.layer2(out)
        out = self.layer3(out)
        out = out + self.res2(out)      # second skip connection
        out = self.classifier(out)
        return out



model = ResNet9().to(DEVICE)

criterion = nn.CrossEntropyLoss()



# switched from SGD to AdamW after reading this: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html



optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)



# slowly lowers lr over training so it doesn't overshoot near the end
# learned about this from: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# for graphing later
train_losses = []
train_accs = []
test_losses = []
test_accs = []

print("\nTraining...")
for epoch in range(EPOCHS):
    start = time.time()

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    scheduler.step()

    epoch_train_loss = running_loss / total
    epoch_train_acc = (correct / total) * 100

    # testing - nothing too extra here
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    epoch_test_loss = test_loss / test_total
    epoch_test_acc = (test_correct / test_total) * 100

    elapsed = time.time() - start

    train_losses.append(epoch_train_loss)
    train_accs.append(epoch_train_acc)
    test_losses.append(epoch_test_loss)
    test_accs.append(epoch_test_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} ({elapsed:.1f}s) - "
          f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
          f"Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.2f}%")

print("\nDone!")

# plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss', marker='o')
ax1.plot(range(1, EPOCHS + 1), test_losses, label='Test Loss', marker='o')
ax1.set_title('Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, EPOCHS + 1), train_accs, label='Train Acc', marker='o')
ax2.plot(range(1, EPOCHS + 1), test_accs, label='Test Acc', marker='o')
ax2.set_title('Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
