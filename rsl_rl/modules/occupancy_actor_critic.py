#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np


class Flatten(nn.Module):
    def forward(self, inp):
        # print(inp.size())
        return inp.reshape(inp.size(0), -1)


class OccupancyActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
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
        mlp_input_dim_a = num_actor_obs - self.tot_perceptive_dims + 32 -3
        mlp_input_dim_c = num_critic_obs - self.tot_perceptive_dims + 32 -3
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
            nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1),  # [batch, 32, D/2, H/2, W/2]
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),              # [batch, 64, D/4, H/4, W/4]
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),             # [batch, 128, D/8, H/8, W/8]
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),            # [batch, 256, D/16, H/16, W/16]
            nn.ReLU(inplace=True),
        )

        self.fc_input_dim = 256 * 2 * 2 * 2

        # Fully connected layers for mu and logvar
        self.fc_mu = nn.Linear(self.fc_input_dim, 32)

        # # Freeze encoder
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        
        # for param in self.fc_mu.parameters():
        #     param.requires_grad = False

        # # Load the model weights here
        # checkpoint = torch.load('models/vae3d_best.pth')
        # encoder_state_dict = {k.replace('encoder.', ''): v for k, v in checkpoint.items() if k.startswith('encoder.encoder.')}
        # fc_mu_state_dict = {k.replace('encoder.fc_mu.', ''): v for k, v in checkpoint.items() if k.startswith('encoder.fc_mu.')}
        # self.encoder.load_state_dict(encoder_state_dict)
        # self.fc_mu.load_state_dict(fc_mu_state_dict)

        # # Load the actor and critic weights here
        # checkpoint_ac = torch.load('logs/rsl_rl/franka_reach_occupancy/warmup_no_obstacle/model_200.pt')
        # actor_state_dict = {k.replace('actor.', ''): v for k, v in checkpoint_ac['model_state_dict'].items() if k.startswith('actor.')}
        # critic_state_dict = {k.replace('critic.', ''): v for k, v in checkpoint_ac['model_state_dict'].items() if k.startswith('critic.')}
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
        self.vision_fc_mu = None
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
    
    def create_occupancy_grids(self, obs_obstacle_position, 
                           cam_x=2.5, cam_y=0.0, cam_z=0.65,
                           org_x=1.7, org_y=-0.64, org_z=-0.64,
                           sphere_radius=0.08, voxel_size=0.02, 
                           grid_dim=64):
        # obs_obstacle_position: (B,3)
        B = obs_obstacle_position.shape[0]

        # Separate coordinates
        obs_x = obs_obstacle_position[:, 0]
        obs_y = obs_obstacle_position[:, 1]
        obs_z = obs_obstacle_position[:, 2]

        # Compute obstacle position relative to camera
        obs_in_cam_x = cam_x - obs_x
        obs_in_cam_y = cam_y - obs_y
        obs_in_cam_z = cam_z - obs_z

        # Shift coordinates
        shifted_x = obs_in_cam_x - org_x
        shifted_y = obs_in_cam_y - org_y
        shifted_z = obs_in_cam_z - org_z

        # Convert to voxel coordinates
        cx = (shifted_x / voxel_size).to(torch.int)
        cy = (shifted_y / voxel_size).to(torch.int)
        cz = (shifted_z / voxel_size).to(torch.int)

        radius_in_voxels = sphere_radius / voxel_size

        # Create a 3D grid of coordinates
        device = obs_obstacle_position.device
        X = torch.arange(grid_dim, dtype=torch.float32, device=device)
        Y = torch.arange(grid_dim, dtype=torch.float32, device=device)
        Z = torch.arange(grid_dim, dtype=torch.float32, device=device)

        # meshgrid with indexing='ij' to match (D,H,W) order
        X, Y, Z = torch.meshgrid(X, Y, Z, indexing='ij')  # Each: (64,64,64)

        # Add batch dimension for broadcasting
        # Shape: (1,64,64,64) for each coordinate
        X = X.unsqueeze(0)
        Y = Y.unsqueeze(0)
        Z = Z.unsqueeze(0)

        # Broadcast centers (cx,cy,cz) to shape (B,1,1,1) for subtraction
        cx = cx.view(B, 1, 1, 1).float()
        cy = cy.view(B, 1, 1, 1).float()
        cz = cz.view(B, 1, 1, 1).float()

        # Compute distances (B,64,64,64)
        dist = torch.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2)

        # Create occupancy grids: 1.0 inside sphere radius, 0.0 outside
        occupancy_grid = (dist <= radius_in_voxels).float()

        return occupancy_grid.unsqueeze(1)

    def update_distribution(self, observations):
        obs_proprio, obs_obstacle_position, obs_perceptive = observations[:, :-(self.tot_perceptive_dims+3)], observations[:, -(self.tot_perceptive_dims+3):-self.tot_perceptive_dims], observations[:, -self.tot_perceptive_dims:]
        # # Add saving of the perspective observations
        # np.save(f"training_images/occupancy_perception_{self.episode}", obs_perceptive.reshape(observations.shape[0], *self.perceptive_dims).cpu().numpy())
        # np.save(f"training_images/obstacle_perception_{self.episode}", obs_obstacle_position.cpu().numpy())
        # self.episode += 1
        # Add obstacle input here
        # Add the obstacle sphere here from the obstacle location
        # obs_perceptive = self.create_occupancy_grids(obs_obstacle_position)
        latent = self.encoder(obs_perceptive.reshape(observations.shape[0], *self.perceptive_dims))
        latent = latent.view(latent.size(0), -1)
        fc_mu = self.fc_mu(latent)
        input = torch.cat((obs_proprio, fc_mu), dim=1)
        mean = self.actor(input)
        self.distribution = Normal(mean, mean * 0.0 + self.std)
        self.vision_fc_mu = fc_mu
        self.obstacle_position_observation = obs_obstacle_position
        self.obs_perceptive = obs_perceptive.reshape(observations.shape[0], *self.perceptive_dims)
        

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        obs_proprio, obs_obstacle_position, obs_perceptive = observations[:, :-(self.tot_perceptive_dims+3)], observations[:, -(self.tot_perceptive_dims+3):-self.tot_perceptive_dims], observations[:, -self.tot_perceptive_dims:]
        # Add obstacle input here
        # Add the obstacle sphere here from the obstacle location
        # obs_perceptive = self.create_occupancy_grids(obs_obstacle_position)
        latent = self.encoder(obs_perceptive.reshape(observations.shape[0], *self.perceptive_dims))
        latent = latent.view(latent.size(0), -1)
        fc_mu = self.fc_mu(latent)
        input = torch.cat((obs_proprio, fc_mu), dim=1)
        actions_mean = self.actor(input)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        obs_proprio, obs_obstacle_position, obs_perceptive = critic_observations[:, :-(self.tot_perceptive_dims+3)], critic_observations[:, -(self.tot_perceptive_dims+3):-self.tot_perceptive_dims], critic_observations[:, -self.tot_perceptive_dims:]
        # Add obstacle input here
        # Add the obstacle sphere here from the obstacle location
        # obs_perceptive = self.create_occupancy_grids(obs_obstacle_position)
        latent = self.encoder(obs_perceptive.reshape(critic_observations.shape[0], *self.perceptive_dims))
        latent = latent.view(latent.size(0), -1)
        fc_mu = self.fc_mu(latent)
        input = torch.cat((obs_proprio, fc_mu), dim=1)
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
