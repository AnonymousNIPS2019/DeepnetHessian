def get_mean_std(dataset):
    if dataset == 'MNIST':
        mean = (0.1307,)
        std = (0.3081,)
    elif dataset == 'FashionMNIST':
        mean = (0.5,)
        std  = (0.5,)
    elif dataset == 'CIFAR10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif dataset == 'CIFAR100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    else:
        raise Exception('Unknown dataset')
    
    return mean, std

