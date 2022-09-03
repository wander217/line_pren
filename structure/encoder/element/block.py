from torch import nn, Tensor
from typing import Optional, Callable, List, Tuple
from functools import partial
import math
import torch
import torch.fx


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


@torch.fx.wrap
def stochastic_depth(x: Tensor, p: float, mode: str, training: bool = True) -> Tensor:
    if p < 0.0 or p > 1.0:
        raise ValueError("drop probability has to be between 0 and 1, but got {}".format(p))
    if mode not in ["batch", "row"]:
        raise ValueError("mode has to be either 'batch' or 'row', but got {}".format(mode))
    if not training or p == 0.0:
        return x

    survival_rate = 1.0 - p
    if mode == "row":
        size = [x.shape[0]] + [1] * (x.ndim - 1)
    else:
        size = [1] * x.ndim
    noise = torch.empty(size, dtype=x.dtype, device=x.device)
    noise = noise.bernoulli_(survival_rate).div_(survival_rate)
    return x * noise


class ConvNormActivation(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size=3,
            stride: int = 1,
            padding=None,
            groups: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
            activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
            dilation: int = 1,
            inplace: bool = True,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        layers = [nn.Conv2d(in_channels,
                            out_channels,
                            kernel_size,
                            stride,
                            padding,
                            dilation=dilation,
                            groups=groups,
                            bias=norm_layer is None)]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer(inplace=inplace))
        super().__init__(*layers)
        self.out_channels = out_channels


class SqueezeExcitation(nn.Module):
    def __init__(
            self,
            input_channels: int,
            squeeze_channels: int,
            activation: Callable[..., nn.Module] = nn.ReLU,
            scale_activation: Callable[..., nn.Module] = nn.Hardsigmoid,
    ) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, x: Tensor) -> Tensor:
        scale = self.avg_pool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, x: Tensor) -> Tensor:
        scale = self._scale(x)
        return scale * x


class StochasticDepth(nn.Module):
    """
    See :func:`stochastic_depth`.
    """

    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        return stochastic_depth(x, self.p, self.mode, self.training)

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'p=' + str(self.p)
        tmpstr += ', mode=' + str(self.mode)
        tmpstr += ')'
        return tmpstr


class MBConvConfig:
    # Stores information listed at Table 1 of the EfficientNet paper
    def __init__(self,
                 expand_ratio: float,
                 kernel: int,
                 stride: int,
                 input_channels: int,
                 out_channels: int,
                 num_layers: int,
                 width_mult: float,
                 depth_mult: float,
                 padding: Optional[int] = None) -> None:
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.num_layers = self.adjust_depth(num_layers, depth_mult)
        self.padding = padding

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += 'expand_ratio={expand_ratio}'
        s += ', kernel={kernel}'
        s += ', stride={stride}'
        s += ', input_channels={input_channels}'
        s += ', out_channels={out_channels}'
        s += ', num_layers={num_layers}'
        s += ')'
        return s.format(**self.__dict__)

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class MBConv(nn.Module):
    def __init__(self,
                 cnf: MBConvConfig,
                 stochastic_depth_prob: float,
                 norm_layer: Callable[..., nn.Module],
                 se_layer: Callable[..., nn.Module] = SqueezeExcitation) -> None:
        super().__init__()

        if isinstance(cnf.stride, int) and not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(ConvNormActivation(cnf.input_channels,
                                             expanded_channels,
                                             kernel_size=1,
                                             norm_layer=norm_layer,
                                             activation_layer=activation_layer))

        # depth wise
        layers.append(ConvNormActivation(expanded_channels,
                                         expanded_channels,
                                         kernel_size=cnf.kernel,
                                         stride=cnf.stride,
                                         groups=expanded_channels,
                                         norm_layer=norm_layer,
                                         padding=cnf.padding,
                                         activation_layer=activation_layer))

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels,
                               squeeze_channels,
                               activation=partial(nn.SiLU, inplace=True)))

        # project
        layers.append(ConvNormActivation(expanded_channels,
                                         cnf.out_channels,
                                         kernel_size=1,
                                         norm_layer=norm_layer,
                                         activation_layer=None))

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channel = cnf.out_channels

    def forward(self, x: Tensor) -> Tensor:
        output: Tensor = self.block(x)
        if self.use_res_connect:
            output = self.stochastic_depth(output)
            output += x
        return output
