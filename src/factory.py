from .datasets import *
from torch.utils.data import DataLoader
from .networks import *
from .attention_networks import *
from torchvision import models
import sys

file_path = os.path.dirname(os.path.realpath(__file__))
project_path = os.path.abspath(os.path.join(file_path, os.pardir))
libs_path = os.path.join(project_path, 'libs')

sys.path.append(libs_path)

from deeplabv3plus_pytorch.network import modeling

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


def create_msls_dataloader(dataset, root_dir, cities, transform, batch_size):
    if dataset == "binary_MSLS":
        ds = MSLSDataSet(root_dir, cities, ds_key="sim", transform=transform)
    elif dataset == "soft_MSLS":
        ds = MSLSDataSet(root_dir, cities, ds_key="fov", transform=transform)
    return DataLoader(ds, batch_size=batch_size, num_workers=4, shuffle=True)


def tweak_in_channels(model: nn.Module, in_channels: int = 3) -> nn.Module:
    first_layer = list(model.children())[0]

    # if first layer isn't a convolutional layer, we can't tweak it
    if not isinstance(first_layer, torch.nn.Conv2d):
        raise ValueError("First layer of model must be a convolutional (Conv2d) layer")
    
    if first_layer.in_channels != 3:
        print("Warning: first layer of model already has in_channels != 3")
    
    # get all the hyperparameters needed to recreate the first layer
    kwargs = {
        "in_channels": in_channels,
        "out_channels": first_layer.out_channels,
        "kernel_size": first_layer.kernel_size,
        "stride": first_layer.stride,
        "padding": first_layer.padding,
        "bias": first_layer.bias is not None
    }

    # create a new first layer with the tweaked in_channels
    new_first_layer = torch.nn.Conv2d(**kwargs)

    # replace the first layer with the new one
    model[0] = new_first_layer

    # returning the model is redundant as the model is modified in-place
    return model



def get_backbone(vpr_name: str, seg_name = None, attention_name = None):
    if vpr_name == "resnet18":
        backbone = models.resnet18(pretrained=True)
    elif vpr_name == "resnet34":
        backbone = models.resnet34(pretrained=True)
    elif vpr_name == "resnet152":
        backbone = models.resnet152(pretrained=True)
    elif vpr_name == "resnet50":
        backbone = models.resnet50(pretrained=True)
    if vpr_name == "densenet161":
        backbone = models.densenet161(pretrained=True).features
        output_dim=2208
    elif vpr_name == "densenet121":
        backbone = models.densenet121(pretrained=True).features
        output_dim=2208
    elif vpr_name == "vgg16":
        backbone = models.vgg16(pretrained=True).features
        output_dim = 512
    elif vpr_name == "resnext":
        backbone = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
    if "resne" in vpr_name:
        backbone = torch.nn.Sequential(*(list(backbone.children())[:-2]))
        output_dim = 2048

    # if only one of seg_name or attention_name is "none", raise an error
    if attention_name is None:
        if seg_name is not None:
            print("Warning: if attention_name is None, seg_name will be ignored")
        return backbone, output_dim
    
    # first change the backbone to have the correct number of input channels
    if attention_name == "all":
        tweak_in_channels(backbone, in_channels=22)
    elif attention_name == "mask_preset":
        tweak_in_channels(backbone, in_channels=4)
    else:
        raise ValueError(f"attention_name {attention_name} not recognized")

    # then get the segmentation network
    if seg_name is None:
        # no segmentation network, so we can return the backbone only
        return backbone, output_dim
    
    semantic_net = get_seg_model(seg_name)
    
    
    if attention_name == "all":
        augmented_backbone = AugmentedVprBackbone(backbone, semantic_net)
    elif attention_name == "mask_preset":
        sem_mask = SemanticMask(semantic_net)
        augmented_backbone = AugmentedVprBackbone(backbone, sem_mask)    

    return augmented_backbone, output_dim


def get_seg_model(seg_name):
    if seg_name == "deeplabv3plus_resnet101":
        semantic_net = modeling.deeplabv3plus_resnet101(num_classes=19)
        weights = "/home/gregory/develop/generalized_contrastive_loss/weights/best_deeplabv3plus_resnet101_cityscapes_os16.pth"

        semantic_net.load_state_dict(torch.load(weights, map_location=torch.device('cpu'))["model_state"])
        return semantic_net
    else:
        raise ValueError(f"seg_name {seg_name} not recognized")

def create_model(name, pool, last_layer=None, norm=None, p_gem=3, mode="siamese", seg_name=None, attention=None):
    backbone, output_dim = get_backbone(name, seg_name, attention)
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
