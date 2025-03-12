import numpy as np
import torch
import torch.nn as nn
from level_replay.distributions import Categorical
from level_replay.utils import init
import torch.nn.functional as F

init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0))

init_dense_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.
                             constant_(x, 0))

init_relu_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), nn.init.calculate_gain('relu'))

init_tanh_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), np.sqrt(2))


class OvercookedPolicy(nn.Module):
    def __init__(self, obs_shape, num_actions, args):
        super(OvercookedPolicy, self).__init__()

        num_filters = 25
        size_hidden_layers = 64

        self.num_convs_layer = args.num_convs_layer
        self.num_hidden_layer = args.num_hidden_layer
        self.obs_shape = obs_shape
        self.use_lstm = args.use_lstm
        self.num_lstm_layer = args.num_lstm_layer

        # set the initial conv layers
        self.conv_and_mlp = nn.Sequential(
            init_dense_(nn.Conv2d(in_channels=self.obs_shape[0], out_channels=num_filters, kernel_size=5, padding=2)),
            nn.LeakyReLU(negative_slope=0.2))

        ## add the N number of conv layers
        for i in range(self.num_convs_layer - 1):
            padding = 1 if i < self.num_convs_layer - 2 else 0
            self.conv_and_mlp.add_module(
                f"conv_{i}",
                init_dense_(
                    nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, padding=padding)))
            self.conv_and_mlp.add_module(f"leaky_relu_{i}", nn.LeakyReLU(negative_slope=0.2))

        ## Set the layers
        self.conv_and_mlp.add_module(f"Flatten", nn.Flatten())

        ## Set the hidden dence layers
        self.fc_layers = nn.Sequential()
        num_inputs = self.num_flat_features(self.conv_and_mlp(torch.Tensor(np.zeros(self.obs_shape).reshape((1,self.obs_shape[0],self.obs_shape[1],self.obs_shape[2])))))
        for i in range(self.num_hidden_layer):
            self.fc_layers.add_module(f"linear_{i}", init_dense_(nn.Linear(num_inputs, size_hidden_layers)))
            self.fc_layers.add_module(f"leaky_relu_{i}", nn.LeakyReLU(negative_slope=0.2))
            num_inputs = size_hidden_layers

        ## Set the hidden dence layers
        self.E3T_fc_layers = nn.Sequential()
        E3T_num_inputs = self.num_flat_features(self.conv_and_mlp(torch.Tensor(np.zeros(self.obs_shape).reshape((1,self.obs_shape[0],self.obs_shape[1],self.obs_shape[2]))))) + num_actions
        for i in range(self.num_hidden_layer):
            self.E3T_fc_layers.add_module(f"linear_{i}", init_dense_(nn.Linear(E3T_num_inputs, size_hidden_layers)))
            self.E3T_fc_layers.add_module(f"leaky_relu_{i}", nn.LeakyReLU(negative_slope=0.2))
            E3T_num_inputs = size_hidden_layers

        self.lstm = nn.LSTM(size_hidden_layers, size_hidden_layers, args.num_lstm_layer, batch_first=False)
        self.critic = init_(nn.Linear(size_hidden_layers, 1))
        self.dist = Categorical(size_hidden_layers, num_actions)

        #apply_init_(self.modules())

        self.train()

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    @property
    def is_recurrent(self):
        return False


    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return 64


    def forward(self, inputs, rnn_hx, rnn_cx, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hx, rnn_cx, masks, latent=None, deterministic=False):
        x = self.conv_and_mlp(inputs)
        if latent != None:
            x = torch.cat((x, latent), dim=1)
            x = self.E3T_fc_layers(x)
        else:
            x = self.fc_layers(x)

        if self.use_lstm:
            rnn_hx = rnn_hx.repeat(self.num_lstm_layer, 1, 1)
            rnn_cx = rnn_cx.repeat(self.num_lstm_layer, 1, 1)

            x, (rnn_hx, rnn_cx) = self.lstm(x.unsqueeze(0), (rnn_hx, rnn_cx))
            x = x.squeeze(0)
            rnn_hx = rnn_hx[-1].squeeze(0)
            rnn_cx = rnn_cx[-1].squeeze(0)
        value = self.critic(x)
        dist = self.dist(x)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        # action_log_probs = dist.log_probs(action)
        action_log_dist = dist.logits
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_dist, rnn_hx, rnn_cx

    def act_human(self, inputs, rnn_hx, rnn_cx, masks, latent=None, deterministic=False):
        x = self.conv_and_mlp(inputs)
        if latent != None:
            x = torch.cat((x, latent), dim=1)
            x = self.E3T_fc_layers(x)
        else:
            x = self.fc_layers(x)

        if self.use_lstm:
            rnn_hx = rnn_hx.repeat(self.num_lstm_layer, 1, 1)
            rnn_cx = rnn_cx.repeat(self.num_lstm_layer, 1, 1)

            x, (rnn_hx, rnn_cx) = self.lstm(x.unsqueeze(0), (rnn_hx, rnn_cx))
            x = x.squeeze(0)
            rnn_hx = rnn_hx[-1].squeeze(0)
            rnn_cx = rnn_cx[-1].squeeze(0)
        value = self.critic(x)
        dist = self.dist(x)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        # action_log_probs = dist.log_probs(action)
        action_log_dist = dist.logits
        dist_entropy = dist.entropy().mean()
        action_probs = dist.probs

        return value, action, action_log_dist, rnn_hx, rnn_cx, action_probs

    def get_value(self, inputs, rnn_hx, rnn_cx, masks, latent=None):
        x = self.conv_and_mlp(inputs)
        if latent != None:
            x = torch.cat((x, latent), dim=1)
            x = self.E3T_fc_layers(x)
        else:
            x = self.fc_layers(x)

        if self.use_lstm:
            rnn_hx = rnn_hx.repeat(self.num_lstm_layer, 1, 1)
            rnn_cx = rnn_cx.repeat(self.num_lstm_layer, 1, 1)

            x, (rnn_hx, rnn_cx) = self.lstm(x.unsqueeze(0), (rnn_hx, rnn_cx))
            x = x.squeeze(0)

        return self.critic(x)

    def evaluate_actions(self, inputs, rnn_hx, rnn_cx, masks, action, latent=None):
        x = self.conv_and_mlp(inputs)
        if latent != None:
            x = torch.cat((x, latent), dim=1)
            x = self.E3T_fc_layers(x)
        else:
            x = self.fc_layers(x)

        if self.use_lstm:
            rnn_hx = rnn_hx.repeat(self.num_lstm_layer, 1, 1)
            rnn_cx = rnn_cx.repeat(self.num_lstm_layer, 1, 1)

            x, (rnn_hx, rnn_cx) = self.lstm(x.unsqueeze(0), (rnn_hx, rnn_cx))
            x = x.squeeze(0)

        value = self.critic(x)
        dist = self.dist(x)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hx, rnn_cx


class OvercookedPolicy_E3T(nn.Module):
    def __init__(self, obs_shape, num_actions, args):
        super(OvercookedPolicy_E3T, self).__init__()

        self.latent_dim = args.latent_dim

        self.context_network = Overcooked_context(obs_shape, num_actions, args)
        self.human_latent_prob_predict = Overcooked_human_prob(obs_shape, num_actions, args)
        self.decoder_network = Overcooked_decoder(obs_shape, num_actions, args)
        self.logit_layer = nn.Linear(self.latent_dim, num_actions)
        self.policy_network = OvercookedPolicy(obs_shape, num_actions, args)

        #apply_init_(self.modules())

        self.train()

    @property
    def is_recurrent(self):
        return False

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return 64

    def forward(self, inputs, rnn_hx, rnn_cx, masks):
        raise NotImplementedError

    def act(self, inputs, pre_obs, action_reward, rnn_hx, rnn_cx, masks, deterministic=False):
        encoded_context = self.context_network(pre_obs, action_reward)
        human_latent_pre = self.human_latent_prob_predict(encoded_context)
        policy_latent_d = self.decoder_network(inputs, human_latent_pre)
        action_logits = self.logit_layer(policy_latent_d)

        action_softmax = F.softmax(action_logits, dim=-1)

        value, action, action_log_dist, rnn_hx, rnn_cx = self.policy_network.act(inputs, rnn_hx, rnn_cx, masks, latent=action_softmax, deterministic=deterministic)

        return value, action, action_log_dist, rnn_hx, rnn_cx

    def act_human(self, inputs, context_human, rnn_hx, rnn_cx, masks, deterministic=False):

        value, action, action_log_dist, rnn_hx, rnn_cx, action_probs = self.policy_network.act_human(inputs, rnn_hx, rnn_cx, masks, latent=context_human, deterministic=deterministic)

        return value, action, action_log_dist, rnn_hx, rnn_cx, action_probs

    def get_value(self, inputs, pre_obs, action_reward, rnn_hx, rnn_cx, masks):
        encoded_context = self.context_network(pre_obs, action_reward)
        human_latent_pre = self.human_latent_prob_predict(encoded_context)
        policy_latent_d = self.decoder_network(inputs, human_latent_pre)
        action_logits = self.logit_layer(policy_latent_d)
        action_softmax = F.softmax(action_logits, dim=-1)
        critic = self.policy_network.get_value(inputs, rnn_hx, rnn_cx, masks, latent=action_softmax)

        return critic

    def evaluate_actions(self, inputs, rnn_hx, rnn_cx, masks, action, pre_obs, action_reward):
        encoded_context = self.context_network(pre_obs, action_reward)
        human_latent_pre = self.human_latent_prob_predict(encoded_context)
        policy_latent_d = self.decoder_network(inputs, human_latent_pre)
        action_logits = self.logit_layer(policy_latent_d)
        action_softmax = F.softmax(action_logits, dim=-1)
        value, action_log_probs, dist_entropy, rnn_hx, rnn_cx = self.policy_network.evaluate_actions(inputs, rnn_hx, rnn_cx, masks, action, latent=action_softmax)

        return value, action_log_probs, dist_entropy, rnn_hx, rnn_cx


class Overcooked_context(nn.Module):
    def __init__(self, obs_shape, num_actions, args):
        super(Overcooked_context, self).__init__()

        num_filters = 25
        size_hidden_layers = 64
        self.latent_dim = args.latent_dim
        self.past_length = args.past_length
        self.num_actions = num_actions

        self.num_convs_layer = args.num_convs_layer
        self.num_hidden_layer = args.num_hidden_layer
        self.obs_shape = obs_shape

        # set the initial conv layers
        self.conv_and_mlp = nn.Sequential(
            init_dense_(nn.Conv2d(in_channels=self.obs_shape[0], out_channels=num_filters, kernel_size=5, padding=2)),
            nn.LeakyReLU(negative_slope=0.2))

        ## add the N number of conv layers
        for i in range(self.num_convs_layer - 1):
            padding = 1 if i < self.num_convs_layer - 2 else 0
            self.conv_and_mlp.add_module(
                f"conv_{i}",
                init_dense_(
                    nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, padding=padding)))
            self.conv_and_mlp.add_module(f"leaky_relu_{i}", nn.LeakyReLU(negative_slope=0.2))

        ## Set the layers
        self.conv_and_mlp.add_module(f"Flatten", nn.Flatten())

        ## embedding one_hot_vec
        self.word_embd_layer = init_dense_(nn.Linear(self.num_actions, self.num_actions))

        ## Set the hidden dence layers
        self.fc_dense_encoder = nn.Sequential()
        num_inputs = self.num_flat_features(self.conv_and_mlp(torch.Tensor(np.zeros(self.obs_shape).reshape((1,self.obs_shape[0],self.obs_shape[1],self.obs_shape[2]))))) + self.num_actions
        for i in range(self.num_hidden_layer - 1):
            self.fc_dense_encoder.add_module(f"linear_{i}", init_dense_(nn.Linear(num_inputs, size_hidden_layers)))
            self.fc_dense_encoder.add_module(f"leaky_relu_{i}", nn.LeakyReLU(negative_slope=0.2))
            num_inputs = size_hidden_layers
        self.fc_dense_encoder.add_module(f"linear_last", init_dense_(nn.Linear(size_hidden_layers, self.latent_dim)))
        self.fc_dense_encoder.add_module(f"leaky_relu_last", nn.LeakyReLU(negative_slope=0.2))

        # ## MLP
        self.mlp = nn.Sequential()
        num_inputs = self.past_length * size_hidden_layers
        for i in range(self.num_hidden_layer):
            self.mlp.add_module(f"linear_{i}", init_dense_(nn.Linear(num_inputs, self.latent_dim)))
            self.mlp.add_module(f"leaky_relu_{i}", nn.LeakyReLU(negative_slope=0.2))
            num_inputs = self.latent_dim

        #apply_init_(self.modules())

        self.train()

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, inputs, latent):
        batch_size = inputs.shape[0]

        if inputs.shape[1] != 1:
            inputs = inputs.reshape(-1, inputs.shape[2], inputs.shape[3], inputs.shape[4])

        x = self.conv_and_mlp(inputs)

        if latent != None:
            latent = torch.reshape(latent, (batch_size * self.past_length, -1))
            latent_last_dim = latent.shape[-1]

            if latent_last_dim <= 2:
                one_hot_labels = F.one_hot(latent, num_classes=self.num_actions)
                latent = self.word_embd_layer(one_hot_labels.float())
                latent = latent.reshape(-1, latent_last_dim * self.num_actions)
            x = torch.cat((x, latent), dim=1)
            x = F.leaky_relu(x)
            x = self.fc_dense_encoder(x)

        if self.past_length > 1:
            x = torch.reshape(x, (batch_size, -1))
            x = self.mlp(x)

        return x


class Overcooked_human_prob(nn.Module):
    def __init__(self, obs_shape, num_actions, args):
        super(Overcooked_human_prob, self).__init__()

        num_layers = 2
        self.latent_dim = 64

        # set the initial conv layers
        self.human_prob_layers = nn.Sequential()
        self.human_prob_layers.add_module(f"Flatten", nn.Flatten())

        for i in range(num_layers):
            self.human_prob_layers.add_module(f"linear_{i}", init_dense_(nn.Linear(self.latent_dim, self.latent_dim)))
            self.human_prob_layers.add_module(f"tanh_{i}", nn.Tanh())

        self.train()

    def forward(self, inputs):
        x = self.human_prob_layers(inputs)
        return x

class Overcooked_decoder(nn.Module):
    def __init__(self, obs_shape, num_actions, args):
        super(Overcooked_decoder, self).__init__()

        num_filters = 25
        size_hidden_layers = 64
        self.latent_dim = args.latent_dim

        self.num_convs_layer = args.num_convs_layer
        self.num_hidden_layer = args.num_hidden_layer
        self.obs_shape = obs_shape

        # set the initial conv layers
        self.conv_and_mlp = nn.Sequential(
            init_dense_(nn.Conv2d(in_channels=self.obs_shape[0], out_channels=num_filters, kernel_size=5, padding=2)),
            nn.LeakyReLU(negative_slope=0.2))

        ## add the N number of conv layers
        for i in range(self.num_convs_layer - 1):
            padding = 1 if i < self.num_convs_layer - 2 else 0
            self.conv_and_mlp.add_module(
                f"conv_{i}", init_dense_(nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, padding=padding)))
            self.conv_and_mlp.add_module(f"leaky_relu_{i}", nn.LeakyReLU(negative_slope=0.2))

        ## Set the layers
        self.conv_and_mlp.add_module(f"Flatten", nn.Flatten())

        ## Set the hidden dence layers
        self.fc_layers = nn.Sequential()
        num_inputs = self.num_flat_features(self.conv_and_mlp(torch.Tensor(np.zeros(self.obs_shape).reshape((1,self.obs_shape[0],self.obs_shape[1],self.obs_shape[2]))))) + self.latent_dim
        for i in range(self.num_hidden_layer-1):
            self.fc_layers.add_module(f"linear_{i}", init_dense_(nn.Linear(num_inputs, size_hidden_layers)))
            self.fc_layers.add_module(f"leaky_relu_{i}", nn.LeakyReLU(negative_slope=0.2))
            num_inputs = size_hidden_layers

        self.last_fc_layer = nn.Sequential(
            init_dense_(nn.Linear(size_hidden_layers, self.latent_dim)),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.train()

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, inputs, latent=None):
        x = self.conv_and_mlp(inputs)

        if latent != None:
            x = torch.cat((x, latent), dim=1)
        x = self.fc_layers(x)
        x = self.last_fc_layer(x)

        return x