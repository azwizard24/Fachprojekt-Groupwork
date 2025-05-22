from transformers import ConvNextPreTrainedModel, ConvNextConfig
import torch.nn as nn
import torch

class ConvNextBirdSet(ConvNextPreTrainedModel):
    def __init__(self, config: ConvNextConfig, num_classes=206):
        super().__init__(config)
        # build your backbone convnext model (like the original)
        self.num_classes = num_classes
        self.convnext = ...  # build backbone here
        self.classifier = nn.Linear(config.hidden_size, num_classes)
        self.preprocess = ... # your preprocessing module/layer if you want
        self.init_weights()

    def forward(self, x):
        # x should already be preprocessed spectrogram tensors
        x = self.convnext(x)
        x = self.classifier(x)
        return x


class BirdsetModule(nn.Module):
    def __init__(self, num_classes=206, checkpoint_path=None):
        super().__init__()
        # Initialize model normally (from HF pretrained weights or config)
        self.model = ConvNextBirdSet.from_pretrained(
            
            "chandanreddy/finetune_birdset",  # or any HF model id or config dir
            num_classes=num_classes
        )
        if checkpoint_path:
            # Load your local checkpoint state dict
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(state_dict)


    def forward(self, x):
        # optionally preprocess here, or expect input preprocessed outside
        preprocessed = self.model.preprocess(x)
        return self.model(preprocessed)
