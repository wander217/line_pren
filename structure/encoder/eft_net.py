import copy
import math

from torch import nn, Tensor
from typing import Any, Callable, List, Optional, Sequence, Dict, Tuple

from torch.utils.model_zoo import load_url
from structure.encoder.element import ConvNormActivation, MBConvConfig, MBConv, _eft_conf

__all__ = ["EfficientNet"]

model_urls: Dict = {
    # Weights ported from https://github.com/rwightman/pytorch-image-models/
    "eb0": "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth",
    "eb1": "https://download.pytorch.org/models/efficientnet_b1_rwightman-533bc792.pth",
    "eb2": "https://download.pytorch.org/models/efficientnet_b2_rwightman-bcdf34b7.pth",
    "eb3": "https://download.pytorch.org/models/efficientnet_b3_rwightman-cf984f9c.pth",
    "eb4": "https://download.pytorch.org/models/efficientnet_b4_rwightman-7eb33cd5.pth",
}

scale_compound: Dict = {
    "eb0": [1.0, 1.0, 0.2],
    "eb1": [1.0, 1.1, 0.2],
    "eb2": [1.1, 1.2, 0.3],
    "eb3": [1.2, 1.4, 0.3],
    "eb4": [1.4, 1.8, 0.4]
}

data_point: List = [3, 5, 7]


class EfficientNet(nn.Module):
    def __init__(
            self,
            inverted_residual_setting: List[MBConvConfig],
            stochastic_depth_prob: float = 0.2,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (isinstance(inverted_residual_setting, Sequence) and
                  all([isinstance(s, MBConvConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        block = MBConv if block is None else block
        norm_layer = nn.BatchNorm2d if norm_layer is None else norm_layer
        layers: List[nn.Module] = []

        # building first layer
        first_output_channels = inverted_residual_setting[0].input_channels
        layers.append(ConvNormActivation(3,
                                         first_output_channels,
                                         kernel_size=3,
                                         stride=2,
                                         norm_layer=norm_layer,
                                         activation_layer=nn.SiLU))

        # building inverted residual blocks
        total_stage_blocks = sum([cnf.num_layers for cnf in inverted_residual_setting])
        stage_block_id = 0
        self.out_channel: List = []
        for i, cnf in enumerate(inverted_residual_setting):
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if i + 1 in data_point:
                self.out_channel.append(cnf.out_channels)
        self.layers: nn.ModuleList = nn.ModuleList(layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> List:
        feature: List = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in data_point:
                feature.append(x)
        return feature

    def forward(self, x: Tensor) -> List:
        return self._forward_impl(x)


def _eft_model(
        arch: str,
        inverted_residual_setting: List[MBConvConfig],
        dropout: float,
        pretrained: bool,
        progress: bool,
        model_dir: str = "",
        **kwargs: Any
) -> nn.Module:
    model = EfficientNet(inverted_residual_setting, dropout, **kwargs)
    if pretrained:
        if model_urls.get(arch, None) is None:
            raise ValueError("No checkpoint is available for network type {}".format(arch))
        state_dict = load_url(model_urls[arch], model_dir, progress=progress)
        model.load_state_dict(state_dict)
    return model


def eft_builder(name: str,
                pretrained: bool = False,
                progress: bool = False,
                **kwargs: Any) -> nn.Module:
    if name not in scale_compound:
        print("Model is not supported for key {}".format(name))
    scp = scale_compound[name]
    inverted_residual_setting = _eft_conf(width_mult=scp[0], depth_mult=scp[1], **kwargs)
    return _eft_model(name, inverted_residual_setting, scp[2], pretrained, progress, **kwargs)
