#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torch.distributions import Normal
import numpy as np

class Flatten(nn.Module):
    def forward(self, inp):
        # print(inp.size())
        return inp.reshape(inp.size(0), -1)


class PerceptiveActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[512, 512, 512],
        critic_hidden_dims=[512, 512, 512],
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = get_activation(activation)

        self.perceptive_dims = kwargs["perceptive_dims"]
        self.tot_perceptive_dims = 1
        for perceptive_dim in self.perceptive_dims:
            self.tot_perceptive_dims *= perceptive_dim
        mlp_input_dim_a = num_actor_obs - self.tot_perceptive_dims + 800 -3
        mlp_input_dim_c = num_critic_obs - self.tot_perceptive_dims + 800 -3
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # Perceptive inputs
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),    # Output: (Batch, 8, 80, 80)
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),    # Output: (Batch, 16, 40, 40)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),    # Output: (Batch, 32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),    # Output: (Batch, 32, 10, 10)
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),    # Output: (Batch, 32, 5, 5)
            # nn.ReLU(),
            # nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),    # Output: (Batch, 32, 3, 3)
            nn.ReLU(),
            nn.Flatten(),
        )

        self.linear_reduction = nn.Linear(32*5*5, 3)

        # # Freeze encoder
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        
        # for param in self.fc_mu.parameters():
        #     param.requires_grad = False

        # # Load the model weights here
        # checkpoint = torch.load('logs/rsl_rl/franka_reach_camera/encoder_sup_latent288/model_550.pt')
        # encoder_state_dict = {k.replace('encoder.', ''): v for k, v in checkpoint['model_state_dict'].items() if k.startswith('encoder.')}
        # actor_state_dict = {k.replace('actor.', ''): v for k, v in checkpoint['model_state_dict'].items() if k.startswith('actor.')}
        # critic_state_dict = {k.replace('critic.', ''): v for k, v in checkpoint['model_state_dict'].items() if k.startswith('critic.')}
        # self.encoder.load_state_dict(encoder_state_dict)
        # self.actor.load_state_dict(actor_state_dict)
        # self.critic.load_state_dict(critic_state_dict)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Encoder CNN: {self.encoder}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # Add vision encoder latent space
        self.obstacle_position_pred = None
        self.obstacle_position_observation = None
        self.obs_perceptive = None

        self.episode = 0

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        obs_proprio, obs_obstacle_position, obs_perceptive = observations[:, :-(self.tot_perceptive_dims+3)], observations[:, -(self.tot_perceptive_dims+3):-self.tot_perceptive_dims], observations[:, -self.tot_perceptive_dims:]
        latent = self.encoder(obs_perceptive.reshape(observations.shape[0], *self.perceptive_dims))
        obs_position_pred = self.linear_reduction(latent)
        input = torch.cat((obs_proprio, latent), dim=1)
        mean = self.actor(input)
        self.distribution = Normal(mean, mean * 0.0 + self.std)
        self.obstacle_position_pred = obs_position_pred
        self.obstacle_position_observation = obs_obstacle_position
        self.obs_perceptive = obs_perceptive.reshape(observations.shape[0], *self.perceptive_dims)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        obs_proprio, obs_obstacle_position, obs_perceptive = observations[:, :-(self.tot_perceptive_dims+3)], observations[:, -(self.tot_perceptive_dims+3):-self.tot_perceptive_dims], observations[:, -self.tot_perceptive_dims:]
        # Add saving of the perspective observations
        # np.save(f"training_images/obs_perception_ {self.episode}", obs_perceptive.reshape(observations.shape[0], *self.perceptive_dims).cpu().numpy())
        self.episode += 1
        latent = self.encoder(obs_perceptive.reshape(observations.shape[0], *self.perceptive_dims))
        obs_position_pred = self.linear_reduction(latent)
        input = torch.cat((obs_proprio, latent), dim=1)
        actions_mean = self.actor(input)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        obs_proprio, obs_obstacle_position, obs_perceptive = critic_observations[:, :-(self.tot_perceptive_dims+3)], critic_observations[:, -(self.tot_perceptive_dims+3):-self.tot_perceptive_dims], critic_observations[:, -self.tot_perceptive_dims:]
        latent = self.encoder(obs_perceptive.reshape(critic_observations.shape[0], *self.perceptive_dims))
        input = torch.cat((obs_proprio, latent), dim=1)
        value = self.critic(input)
        return value


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
