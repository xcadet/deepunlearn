from torchvision import transforms

from .transforms import ConvertTo3Channels

MNIST_IMAGE_SIZE = 32 

MNIST_MEAN = [0.1307] * 3
MNIST_STD = [0.3081] * 3


def get_mnist_train_transform():
    transform = transforms.Compose(
        [
            ConvertTo3Channels(),
            transforms.Resize(size=(MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD),
        ]
    )
    return transform


def get_mnist_test_transform():
    transform = transforms.Compose(
        [
            ConvertTo3Channels(),
            transforms.Resize(size=(MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD),
        ]
    )
    return transform
