import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from U_net_model import Unet
from Dataset import *
from tqdm import tqdm
import numpy as np
import skimage.io as io

PATH = './model/unet_model.pt'

# Whether to use cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# mask only needs to be converted to tensor
y_transforms = transforms.ToTensor()


def train_model(model, criterion, optimizer, dataload, num_epochs=10):
    best_model = model
    min_loss = 1000
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in tqdm(dataload):
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss / step))
        if (epoch_loss / step) < min_loss:
            min_loss = (epoch_loss / step)
            best_model = model
    torch.save(best_model.state_dict(), PATH)
    return best_model


# Training model
def train():
    model = Unet(1, 1).to(device)
    batch_size = 1
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    train_dataset = TrainDataset("dataset/train/image", "dataset/train/label", transform=x_transforms,
                                 target_transform=y_transforms)
    dataloaders = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_model(model, criterion, optimizer, dataloaders)


# Save the output of the model
def test():
    model = Unet(1, 1)
    model.load_state_dict(torch.load(PATH))
    test_dataset = TestDataset("dataset/test", transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(test_dataset, batch_size=1)
    model.eval()
    import matplotlib.pyplot as plt
    plt.ion()
    with torch.no_grad():
        for index, x in enumerate(dataloaders):
            y = model(x)
            img_y = torch.squeeze(y).numpy()
            img_y = img_y[:, :, np.newaxis]  # it was made for RGB so might need debugin
            img = img_y[:, :]
            io.imsave("./dataset/test/" + str(index) + "_predict.png", img)
            plt.pause(0.01)
        plt.show()


if __name__ == '__main__':
    print("Start training")
train()
print("Training completed, save model")
print("-" * 20)
print("Start prediction")
test()
