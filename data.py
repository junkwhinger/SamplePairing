import os
import torch
import torchvision.datasets as vdatasets
import torchvision.transforms as vtransforms


def get_dataloaders(data_dir, batch_size):

    train_data_dir = os.path.join(data_dir, "train")
    valid_data_dir = os.path.join(data_dir, "valid")
    test_data_dir = os.path.join(data_dir, "test")

    train_transform_fn = vtransforms.Compose([
        vtransforms.Resize((197, 197)),
        vtransforms.RandomHorizontalFlip(),
        vtransforms.ToTensor(),
        vtransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform_fn = vtransforms.Compose([
        vtransforms.Resize((197, 197)),
        vtransforms.ToTensor(),
        vtransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = vdatasets.ImageFolder(train_data_dir, train_transform_fn)
    valid_dataset = vdatasets.ImageFolder(valid_data_dir, test_transform_fn)
    test_dataset = vdatasets.ImageFolder(test_data_dir, test_transform_fn)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader
