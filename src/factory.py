from src.datasets import *
from torch.utils.data import DataLoader
from src.networks import *
from torchvision import models


def create_dataloader(dataset, root_dir, idx_file, gt_file, image_t, batch_size):
    # Create dataset
    if dataset == "test":
        ds = TestDataSet(root_dir, idx_file, transform=image_t)
        return DataLoader(ds, batch_size=batch_size, num_workers=4)

    if dataset == "soft_siamese":
        ds = SiameseDataSet(root_dir, idx_file, gt_file, ds_key="fov", transform=image_t)
    elif dataset == "binary_siamese":
        ds = SiameseDataSet(root_dir, idx_file, gt_file, ds_key="sim", transform=image_t)
    return DataLoader(ds, batch_size=batch_size, num_workers=4, shuffle=True)


def create_msls_dataloader(dataset, root_dir, cities, transform, batch_size,model=None):
    if dataset == "binary_MSLS":
        ds = MSLSDataSet(root_dir, cities, ds_key="sim", transform=transform)
    elif dataset == "soft_MSLS":
        ds = MSLSDataSet(root_dir, cities, ds_key="fov", transform=transform)
    return DataLoader(ds, batch_size=batch_size, num_workers=4, shuffle=True)


def get_backbone(name):
    if name == "resnet18":
        backbone = models.resnet18(pretrained=True)
    elif name == "resnet34":
        backbone = models.resnet34(pretrained=True)
    elif name == "resnet152":
        backbone = models.resnet152(pretrained=True)
    elif name == "resnet50":
        backbone = models.resnet50(pretrained=True)
    if name == "densenet161":
        backbone = models.densenet161(pretrained=True).features
        output_dim=2208
    elif name == "densenet121":
        backbone = models.densenet121(pretrained=True).features
        output_dim=2208
    elif name == "vgg16":
        backbone = models.vgg16(pretrained=True).features
        output_dim = 512
    elif name == "resnext":
        backbone = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
    if "resne" in name:
        backbone = torch.nn.Sequential(*(list(backbone.children())[:-2]))
        output_dim = 2048
    return backbone, output_dim


def create_model(name, pool, last_layer=None, norm=None, p_gem=3, mode="siamese"):
    backbone, output_dim = get_backbone(name)
    layers = len(list(backbone.children()))

    if last_layer is None:
        last_layer = layers
    elif "densenet" in name:
        last_layer=last_layer*2
    elif "vgg" in name:
        last_layer=last_layer*8-2
    aux = 0
    for c in backbone.children():

        if aux < layers - last_layer:
            print(aux, c._get_name(), "IS FROZEN")
            for p in c.parameters():
                p.requires_grad = False
        else:
            print(aux, c._get_name(), "IS TRAINED")
        aux += 1
    if mode=="siamese":
        return SiameseNet(backbone, pool, norm=norm, p=p_gem)
    else:
        return BaseNet(backbone, pool, norm=norm, p=p_gem)
