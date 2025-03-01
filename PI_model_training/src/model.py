import torch
import torch.nn as nn

class Model(nn.Module):
    
    def _calculate_in_features(self, blocks, input_shape):
        """
        Determines the `in_features` for the linear layer by forwarding a dummy tensor
        through all layers up to the classifier.
        """
        model = nn.Sequential(*blocks)
        dummy_input = torch.zeros(1, *input_shape)  # Batch size = 1
        output = model(dummy_input)
        return output.numel()  # Total number of elements after flattening


    def _build_model_from_json(self, model, input_shape):
        layers = []
        feature_extractor_blocks = []
        
        for block_name, block_layers in model.items():
            block = []
            if block_name == "classifier":
                block.append(nn.Flatten())
            for layer in block_layers:
                layer_type = layer['type']
                params = layer['params'] or {}
                
                if layer_type == 'conv':
                    block.append(nn.Conv2d(**params))
                elif layer_type == 'relu':
                    block.append(nn.ReLU())
                elif layer_type == 'batchnorm':
                    block.append(nn.BatchNorm2d(**params))
                elif layer_type == 'maxpool':
                    block.append(nn.MaxPool2d(**params))
                elif layer_type == 'avgpool':
                    block.append(nn.AvgPool2d(**params))
                elif layer_type == 'dropout':
                    block.append(nn.Dropout(**params))
                elif layer_type == 'linear':
                    # Calculate in_features for the linear layer
                    in_features = self._calculate_in_features(feature_extractor_blocks, input_shape)
                    params['in_features'] = in_features
                    block.append(nn.Linear(**params))
            
            layers.append(nn.Sequential(*block))
            feature_extractor_blocks.extend(block)  # Keep track of all layers up to the classifier
        
        return nn.Sequential(*layers)
    
    def __init__(self, model, input_shape) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.model = self._build_model_from_json(model, input_shape)
    
    def forward(self, x):
        return self.model(x)
