import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import torchvision


class Mnist_tensorboard(pl.LightningModule):
    def __init__(self):
        # this is the init function where we will defnine the architecture
        super(Mnist_tensorboard, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.5))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.5))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            torch.nn.Dropout(p=0.5))

        self.fc1 = torch.nn.Linear(4 * 4 * 128, 625, bias=True)
        torch.nn.init.xavier_uniform(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5))
        self.dense1_bn = torch.nn.BatchNorm1d(625)
        self.fc2 = torch.nn.Linear(625, 10, bias=True)
        self.dense2_bn = torch.nn.BatchNorm1d(10)
        torch.nn.init.xavier_uniform_(self.fc2.weight)  # initialize parameters

    def prepare_data(self):

        MNIST(os.getcwd(), train=True, download=True)
        MNIST(os.getcwd(), train=False, download=True)

        #  getcwd() returns current working directory of a process.
        mnist_train = MNIST(os.getcwd(), train=True,
                            download=False, transform=transforms.ToTensor())

        self.train_set, self.val_set = random_split(mnist_train, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=128, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=128, num_workers=4)

    def test_dataloader(self):
        return DataLoader(MNIST(os.getcwd(), train=False, download=False, transform=transforms.ToTensor()), batch_size=128, num_workers=32)

    def forward(self, x):
        # evaluating the batch data as it moves forward in the netowrk
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.dense1_bn(self.fc1(out))
        out = self.dense2_bn(self.fc2(out))
        return F.softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, labels = batch
        pred = self.forward(x)

        correct = pred.argmax(dim=1).eq(labels).sum().item()
        total = len(labels)
        # calculating the loss
        train_loss = F.cross_entropy(pred, labels)

        # logs
        logs = {"train_loss": train_loss}

        output = {
            # REQUIRED: It ie required for us to return "loss"
            "loss": train_loss,
            # optional for logging purposes
            "log": logs,
            "correct": correct,
            "total": total
        }

        return output

    def training_epoch_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        print("Loss train= {}".format(avg_loss))
        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])
        tensorboard_logs = {'loss': avg_loss, "Accuracy": correct/total}
        print("Number of Correctly identified Training Set Images {} from a set of {}. \nAccuracy= {} ".format(
            correct, total, correct/total))
        return {'loss': avg_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        # The code that runs as we forward pass a validation batch
        x, y = batch
        y_hat = self(x)
        correct = y_hat.argmax(dim=1).eq(y).sum().item()
        total = len(y)
        return {'val_loss': F.cross_entropy(y_hat, y), "correct": correct, "total": total}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        # The code that runs as a validation epoch finished
        # Used for metric evaluation
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])
        print("Loss val= {}".format(avg_loss))
        print("Number of Correctly identified Validation Images {} from aset of {}. \nAccuracy= {} ".format(
            correct, total, correct/total))
        tensorboard_logs = {'val_loss': avg_loss, "Accuracy": correct/total}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        # The code that runs as we forward pass a test batch

        x, y = batch
        y_hat = self(x)
        correct = y_hat.argmax(dim=1).eq(y).sum().item()
        total = len(y)
        # returning the batch_dictionary
        return {'test_loss': F.cross_entropy(y_hat, y),
                "correct": correct,
                "total": total}

    def test_epoch_end(self, outputs):
        # OPTIONAL
        # The code that runs as a validation epoch finished
        # Used for metric evaluation
        testCorrect = sum([x["correct"] for x in outputs])
        testTotal = sum([x["total"] for x in outputs])
        print("Number of Correctly identified Testing Images {} from aset of {}. \nAccuracy= {} ".format(
            testCorrect, testTotal, testCorrect/testTotal))
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        print("Loss= {}".format(avg_loss))
        logs = {'test_loss': avg_loss}
        return {'test_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        # REQUIRED
        # Can return multiple optimizers and learning_rate schedulers
        return torch.optim.Adam(self.parameters(), lr=0.0005)


if __name__ == '__main__':

    # Using the Lightning trainer and specifing the requied parameters as arguments
    myTrainer = pl.Trainer(gpus=1, max_epochs=10, checkpoint_callback=False)

    myModel = Mnist_tensorboard()

    # Begin Training
    myTrainer.fit(myModel)
