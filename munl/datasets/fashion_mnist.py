from torchvision import transforms

from .transforms import ConvertTo3Channels

FASHION_MNIST_IMAGE_SIZE = 32
FASHION_MNIST_MEAN = [0.2860] * 3
FASHION_MNIST_STD = [0.3530] * 3


def get_fashion_mnist_train_transform():
    transform = transforms.Compose(
        [
            ConvertTo3Channels(),
            transforms.Resize(
                size=(FASHION_MNIST_IMAGE_SIZE, FASHION_MNIST_IMAGE_SIZE)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=FASHION_MNIST_MEAN, std=FASHION_MNIST_STD),
        ]
    )
    return transform


def get_fashion_mnist_test_transform():
    transform = transforms.Compose(
        [
            ConvertTo3Channels(),
            transforms.Resize(
                size=(FASHION_MNIST_IMAGE_SIZE, FASHION_MNIST_IMAGE_SIZE)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=FASHION_MNIST_MEAN, std=FASHION_MNIST_STD),
        ]
    )
    return transform
