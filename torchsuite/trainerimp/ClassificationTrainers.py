from torchsuite.Trainer import Trainer
import torch

class TrainBinaryNet(Trainer):
    def calculate_acc(self, dataloader):
        acc_sum = 0
        total = 0
        for x, y in dataloader:
            if self.gpu_boole:
                x = x.cuda()
                y = y.cuda()
            outputs = self.net(x)
            acc_sum += torch.sum((outputs > 0.5).float() == y)
            total += y.shape[0]

        acc = acc_sum / total
        return acc.cpu().data.item()


class TrainMulticlassNet(Trainer):
    def calculate_acc(self, dataloader):
        acc_sum = 0
        total = 0
        for x, y in dataloader:
            if self.gpu_boole:
                x = x.cuda()
                y = y.cuda()
            outputs = self.net(x)
            acc_sum += torch.sum(torch.argmax(outputs, axis=1) == y)
            total += y.shape[0]

        acc = acc_sum / total
        return acc.cpu().data.item()

