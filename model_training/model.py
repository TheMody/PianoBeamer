import torch
import torchvision
from transformers import Mask2FormerForUniversalSegmentation



class custom_model(torch.nn.Module):
    def __init__(self, processor, model_name = "facebook/mask2former-swin-base-IN21k-ade-semantic", num_classes = 2):
        super(custom_model, self).__init__()
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        self.processor = processor
        self.linear = torch.nn.Linear(self.model.config.hidden_size, num_classes)
        self.resizeconv = torch.nn.Conv2d(
            num_classes + 3, 64, kernel_size=(7,7),  padding=(3,3), padding_mode='reflect'
        )
        self.conv2d = torch.nn.Conv2d(
            64, num_classes, kernel_size=(3,3),padding=(1,1), padding_mode='reflect'
        )

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values).pixel_decoder_last_hidden_state
        outputs = self.linear(outputs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        outputs = torchvision.transforms.Resize(pixel_values.shape[2:])(outputs)
        outputs = torch.cat([outputs,pixel_values],dim=1,)
        outputs = torch.relu(self.resizeconv(outputs))
        outputs = self.conv2d(outputs)

        return outputs
