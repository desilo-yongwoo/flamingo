from torch.utils.data import DataLoader, random_split, Subset
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip
from torchvision.datasets import CIFAR10

def load_data(index=None):
    """Load FMNIST (training and test set)."""
    transform_train = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = CIFAR10(".", train=True, download=True, transform=transform_train)
    testset = CIFAR10(".", train=False, download=True, transform=transform_test)

    if index is None:
        train_subset, _ = random_split(trainset, [len(trainset)//64, len(trainset)- len(trainset)//64])

    else:
        chunk = len(trainset)//64
        train_subset = Subset(trainset, range(chunk*(index-1), min(chunk*index, len(trainset))))

    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    test_loader = DataLoader(testset, batch_size=128)
    number_examples = {"trainset": len(trainset), "testset": len(testset)}

    return train_loader, test_loader, number_examples
