import torch
from torch import nn
from torch.nn import functional as F


# preselected set of layer indices that we deem useful for the semantic mask
USEFUL_LAYERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
class SemanticMask(nn.Module):
    def __init__(self, semantic_net, useful_layers=None):
        super(SemanticMask, self).__init__()

        # if no useful layers are specified, use the default
        self.useful_layers = useful_layers if useful_layers is not None else USEFUL_LAYERS
        self.semantic_net = semantic_net

    def forward(self, x0):
        # run the semantic segmentation network
        semantic_activations = self.semantic_net.forward(x0)

        # zero out the one that we don't want
        nb_channels = semantic_activations.shape[1]
        for i in range(nb_channels):
            if i not in self.useful_layers:
                semantic_activations[:, i] = 0

        # average the channels pixel-wise, but keep the channel dimension
        mask = torch.mean(semantic_activations, dim=1).unsqueeze(1)

        return mask
    

# a wrapper class that adds some kind of augmentation to the input, beit a semantic mask or all 19 semantic channels
class AugmentedVprBackbone(nn.Module):
    def __init__(self, main_model, augmentation_net):
        super(AugmentedVprBackbone, self).__init__()
        
        # save the main model and the semantic segmentation network
        self.main_model = main_model
        self.augmentation_net = augmentation_net

    def forward(self, x0):
        # get the semantic segmentation mask
        augmentation = self.augmentation_net.forward(x0)
        
        # concatenate the mask to the input
        x0_with_mask = torch.cat((x0, augmentation), dim=1)

        # run the main model with the added mask        
        out = self.main_model.forward(x0_with_mask)
        return out