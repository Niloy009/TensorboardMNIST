# importing necessary libraries
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
from pytorch_lightning.loggers import TensorBoardLogger


# defining the model
class mnistTensorboardWithLogger(pl.LightningModule):

    def __init__(self):
        # this is the init function where we will defnine the architecture
        super(mnistTensorboardWithLogger, self).__init__()

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

        # reference dummy image for logging graph
        self.reference_image = torch.rand((1, 1, 28, 28))

    def prepare_data(self):

        # This contains the manupulation on data that needs to be done only once such as downloading it

        # download the MNIST dataset
        MNIST(os.getcwd(), train=True, download=True)
        MNIST(os.getcwd(), train=False, download=True)

        # See here I have set download to false as it is already downloaded
        mnist_train = MNIST(os.getcwd(), train=True,
                            download=False, transform=transforms.ToTensor())

        # dividing into validation and training set
        self.train_set, self.val_set = random_split(mnist_train, [55000, 5000])

    def train_dataloader(self):
        # REQUIRED
        # This is an essential function. Needs to be included in the code

        return DataLoader(self.train_set, batch_size=128, num_workers=4)

    def val_dataloader(self):
        # OPTIONAL
        # loading validation dataset
        return DataLoader(self.val_set, batch_size=128, num_workers=4)

    def test_dataloader(self):
        # OPTIONAL
        # loading test dataset
        return DataLoader(MNIST(os.getcwd(), train=False, download=False, transform=transforms.ToTensor()), batch_size=128, num_workers=4)

    def forward(self, x):
        # evaluating the batch data as it moves forward in the netowrk
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        # out = self.fc1(out)
        out = self.dense1_bn(self.fc1(out))
        # out = self.fc2(out)
        out = self.dense2_bn(self.fc2(out))
        return F.softmax(out, dim=1)

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

        # Adding logs to TensorBoard
        self.logger.experiment.add_scalar(
            "Loss/Val", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar(
            "Accuracy/Val", correct/total, self.current_epoch)

        return {'val_loss': avg_loss}

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        # The code that runs as we forward pass a test batch

        x, y = batch
        y_hat = self(x)
        correct = y_hat.argmax(dim=1).eq(y).sum().item()
        total = len(y)
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

        # Logging Data to TensorBoard
        self.logger.experiment.add_scalar(
            "Loss/Test", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar(
            "Accuracy/Test", testCorrect/testTotal, self.current_epoch)

        return {'test_loss': avg_loss}

    def configure_optimizers(self):
        # REQUIRED
        # Can return multiple optimizers and learning_rate schedulers
        return torch.optim.Adam(self.parameters(), lr=0.0005)

    def custom_histogram_adder(self):
        # A custom defined function that adds Histogram to TensorBoard

        # Iterating over all parameters and logging them
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name, params, self.current_epoch)

    def training_step(self, batch, batch_idx):
        # REQUIRED
        # print(batch)
        if(batch_idx == 0):
            self.reference_image = (batch[0][0]).unsqueeze(0)
            # self.reference_image.resize((1,1,28,28))
            print(self.reference_image.shape)

        if(self.current_epoch == 1):
            sampleImg = torch.rand((1, 1, 28, 28))
            self.logger.experiment.add_graph(
                mnistTensorboardWithLogger(), sampleImg)
            # self.write.add_graph(mnistTensorboardWithLogger, sampleImg)

        # extracting input and output from the batch
        x, labels = batch
        pred = self.forward(x)

        correct = pred.argmax(dim=1).eq(labels).sum().item()
        total = len(labels)
        # calculating the loss
        train_loss = F.cross_entropy(pred, labels)

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

    def showActivations(self, x):
        # Evaluating the batch data as it moves forward in the netowrk
        # Custom made function for this model to log activations
        print(x.shape)
        plt.imshow(torch.Tensor.cpu(x[0][0]))

        # Logging the input image
        self.logger.experiment.add_image("input", torch.Tensor.cpu(
            x[0][0]), self.current_epoch, dataformats="HW")
        plt.show()
        plt.clf()
        out = self.layer1(x)
        outer = (torch.Tensor.cpu(out).detach())
        plt.figure(figsize=(20, 5))
        b = np.array([]).reshape(0, outer.shape[2])
        c = np.array([]).reshape(4*outer.shape[2], 0)

        # Plotting for layer 1
        i = 0
        j = 0
        while(i < 32):
            img = outer[0][i]
            b = np.concatenate((img, b), axis=0)
            j += 1
            if(j == 4):
                c = np.concatenate((c, b), axis=1)
                b = np.array([]).reshape(0, outer.shape[2])
                j = 0

            i += 1

        plt.imshow(c)
        plt.show()
        plt.clf()
        self.logger.experiment.add_image(
            "layer 1", c, self.current_epoch, dataformats="HW")

        out = self.layer2(out)
        outer = (torch.Tensor.cpu(out).detach())
        plt.figure(figsize=(10, 10))
        b = np.array([]).reshape(0, outer.shape[2])
        c = np.array([]).reshape(8*outer.shape[2], 0)

        # Plotting for layer2
        i = 0
        j = 0
        while(i < 64):
            img = outer[0][i]
            b = np.concatenate((img, b), axis=0)
            j += 1
            if(j == 8):
                c = np.concatenate((c, b), axis=1)
                b = np.array([]).reshape(0, outer.shape[2])
                j = 0

            i += 1

        self.logger.experiment.add_image(
            "layer 2", c, self.current_epoch, dataformats="HW")
        plt.imshow(c)
        plt.show()
        plt.clf()

        # print(out.shape)
        out = self.layer3(out)
        outer = (torch.Tensor.cpu(out).detach())
        plt.figure(figsize=(20, 5))
        b = np.array([]).reshape(0, outer.shape[2])
        c = np.array([]).reshape(8*outer.shape[2], 0)

        # Plotting for layer3
        j = 0
        i = 0
        while(i < 128):
            img = outer[0][i]
            b = np.concatenate((img, b), axis=0)
            j += 1
            if(j == 8):
                c = np.concatenate((c, b), axis=1)
                b = np.array([]).reshape(0, outer.shape[2])
                j = 0

            i += 1
        # print(c.shape)

        self.logger.experiment.add_image(
            "layer 3", c, self.current_epoch, dataformats="HW")
        plt.imshow(c)
        plt.show()

    def training_epoch_end(self, outputs):

        # Logging activations
        self.showActivations(self.reference_image)

        # Logging graph
        # if(self.current_epoch == 1):
        #     sampleImg = torch.rand((1, 1, 28, 28))
        #     self.logger.experiment.add_graph(
        #         mnistTensorboardWithLogger(), sampleImg)

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        print("Loss train= {}".format(avg_loss))
        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])

        # Loggig scalars
        self.logger.experiment.add_scalar(
            "Loss/Train", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar(
            "Accuracy/Train", correct/total, self.current_epoch)

        # Logging histograms
        self.custom_histogram_adder()

        print("Number of Correctly identified Training Set Images {} from a set of {}. \nAccuracy= {} ".format(
            correct, total, correct/total))
        return {'loss': avg_loss}


if __name__ == "__main__":

    # Using the Lightning trainer and specifing the requied parameters as arguments
    # logger = TensorBoardLogger('tb_logs', name='testing_for_activations')

    myTrainerWithLogger = pl.Trainer(
        gpus=1, max_epochs=1, checkpoint_callback=False)

    myModelWithLogger = mnistTensorboardWithLogger()

    myTrainerWithLogger.fit(myModelWithLogger)
