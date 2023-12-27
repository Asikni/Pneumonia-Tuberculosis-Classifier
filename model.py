import torch
import torchvision
from torchvision import transforms
from torch import nn


def create_google_net_model(num_classes: int=10):
    """Creates an EfficientNetB2 feature extractor model and transforms.

    Args:
        num_classes (int, optional): number of classes in the classifier head.
            Defaults to 3.
        seed (int, optional): random seed value. Defaults to 42.

    Returns:
        model (torch.nn.Module): EffNetB2 feature extractor model.
        transforms (torchvision.transforms): EffNetB2 image transforms.
    """
    # Create EffNetB2 pretrained weights, transforms and model
    weights = models.googlenet(pretrained=True)
    transforms = transforms.Compose([
    transforms.Resize(224),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])
    model =  models.googlenet(weights=weights)

    # Freeze all layers in base model

    for param in model.children():
      for param_child in param.parameters():  # Accessing children of the model
        param_child.requires_grad= False
    # Change classifier head with random seed for reproducibility


    # Recreate the classifier layer and seed it to the target device
    model.fc =  torch.nn.Linear(in_features=1024,
                    out_features=num_classes, # same number of output units as our number of classes
                    bias=True).to(device)

    return model, transforms
