import torch
import torchvision

from torch import nn

def create_resnet50_model(num_classes:int=9, # 4
                          seed:int=42):
    """Creates an ResNet50 feature extractor model and transforms.

    Args:
        num_classes (int, optional): number of classes in the classifier head. 
            Defaults to 3.
        seed (int, optional): random seed value. Defaults to 42.

    Returns:
        model (torch.nn.Module): ResNet50 feature extractor model. 
        transforms (torchvision.transforms): ResNet50 image transforms.
    """
    # 1, 2, 3. Create ResNet50 pretrained weights, transforms and model
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.resnet50(weights=weights)

    # 4. Freeze all layers in base model
    for param in model.parameters():
        param.requires_grad = True # Set to False for model's other than ResNet

    # 5. Change classifier head with random seed for reproducibility
    torch.manual_seed(seed)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=2048
                  , out_features=num_classes), # If using EffnetB2 in_features = 1408, EffnetB0 in_features = 1280, if ResNet50 in_features = 2048
    )
    
    return model, transforms
