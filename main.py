from utils import get_device, set_seed, model_summary
from utils.experiment import Experiment
from utils.backprop import Train, Test
from models import ResNet18, ResNet34
from datasets import CIFAR10

set_seed()
batch_size = 32

def print_summary(model):
    print(model_summary(model, input_size=(batch_size, 3, 32, 32)))

def get_cifar10_dataset_and_resnet18_model():
    dataset = CIFAR10(batch_size=batch_size)
    model = ResNet18()
    return dataset, model

def create_resnet18_experiment(epochs=20, criterion='crossentropy', scheduler='one_cycle'):
    model, dataset = get_cifar10_dataset_and_resnet18_model()
    return Experiment(model, dataset, epochs=epochs, scheduler=scheduler, criterion=criterion)

def create_resnet34_experiment(epochs=20, criterion='crossentropy', scheduler='one_cycle'):
    model = ResNet34()
    dataset = CIFAR10(batch_size=batch_size)
    return Experiment(model, dataset, epochs=epochs, scheduler=scheduler, criterion=criterion)

def main():
    resnet14 = create_resnet18_experiment()
    resnet14.execute()
    resnet14.plot_stats()
    resnet14.show_incorrect()
    resnet14.show_incorrect(cams=True)

if __name__ == '__main__':
    main()
