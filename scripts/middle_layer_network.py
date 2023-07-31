from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gym import spaces
import torch as th
from torch import nn

from stable_baselines3 import PPO
from sb3_contrib.ppo_recurrent import MlpLstmPolicy
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.torch_layers import MlpExtractor


class ImportantObsNetwork(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        net_arch: List[int],
        important_obs_layers_dims: int,
        activation_fn: Type[nn.Module],
    ) -> None:
        super().__init__()
        self.activation_fn = activation_fn
        self.layer: List[nn.Module] = []
        last_layer_dim:int = feature_dim
        for idx, curr_layer_dim in enumerate(net_arch):
            if idx == len(net_arch) - 1:
                self.layer.append(nn.Linear(last_layer_dim, curr_layer_dim))
            else:
                self.layer.append(nn.Linear(last_layer_dim + important_obs_layers_dims, curr_layer_dim))
            last_layer_dim = curr_layer_dim
        self.layer = nn.ModuleList(self.layer)

    def forward(self, x: th.Tensor, important_obs: th.Tensor):
        if (important_obs.device != x.device):
            important_obs = important_obs.to(x.device)

        
        for idx, layer in enumerate(self.layer):
            if (layer.weight.device != x.device):
                layer = layer.to(x.device)
            if idx == len(self.layer) - 1:
                input = x.to(th.float32)
            else:
                input = th.cat((x, important_obs), dim=1).to(th.float32)
            x = layer(input)
            x = self.activation_fn()(x)
        return x


class MiddleLayerNetwork(MlpExtractor):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[List[int], Union[Dict[str, List[int]], Dict[str, int]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
    ):
        device = get_device(device)
        super().__init__(
            feature_dim,
            [{k: v for k, v in net_arch.items() if k!="important_obs"}],
            activation_fn,
            device,
        )

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        if isinstance(net_arch, dict):
            # Note: if key is not specificed, assume linear network
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
            important_obs_layers_dims = net_arch.get("important_obs", 0)  # Layer sizes of the important obs
        else:
            pi_layers_dims = vf_layers_dims = net_arch
            important_obs_layers_dims: int = 0

        # Policy network
        self.policy_net = ImportantObsNetwork(
            feature_dim, pi_layers_dims, important_obs_layers_dims, activation_fn 
        )
        # Value network
        self.value_net = ImportantObsNetwork(feature_dim, vf_layers_dims, important_obs_layers_dims, activation_fn)

    def forward(self, features: th.Tensor, important_obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features, important_obs), self.forward_critic(features, important_obs)

    def forward_actor(self, features: th.Tensor, important_obs: th.Tensor) -> th.Tensor:
        return self.policy_net(features, important_obs)

    def forward_critic(self, features: th.Tensor, important_obs: th.Tensor) -> th.Tensor:
        return self.value_net(features, important_obs)


class MiddleLayerActorCriticPolicy(MlpLstmPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            use_sde = use_sde,
            log_std_init = log_std_init,
            activation_fn = activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        self.activation_fn = activation_fn

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MiddleLayerNetwork(
            self.lstm_output_dim,
            net_arch= self.net_arch,
            activation_fn=self.activation_fn,
        )