from torchvision import transforms

CIFAR10_IMAGE_SIZE = 32
CIFAR10_PADDING = 4
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def get_cifar10_train_transform():
    transform = transforms.Compose(
        [
            transforms.RandomCrop(CIFAR10_IMAGE_SIZE, padding=CIFAR10_PADDING),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )
    return transform


def get_cifar10_test_transform():
    transform = transforms.Compose(
        [
            transforms.Resize(size=(CIFAR10_IMAGE_SIZE, CIFAR10_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )
    return transform
