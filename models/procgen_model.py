import math

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch

torch, nn = try_import_torch()
F = nn.functional


# MODULES
#=====================================================================

def init(module, gain=None, activation=None, param=None):
    if gain is None:
        gain = 1 if activation is None else nn.init.calculate_gain(activation, param)

    nn.init.orthogonal_(module.weight, gain=gain)
    nn.init.constant_(module.bias, 0)

    return module


def init_recursive(module, gain=None, activation=None, param=None):
    for submodule in module.children():
        if isinstance(submodule, (nn.Linear, nn.Conv2d)):
            init(submodule, gain, activation, param)
        else:
            init_recursive(submodule, gain, activation, param)


class MultiheadAttention(nn.Module):
    def __init__(self, n_features, n_heads, dropout):
        super().__init__()

        assert n_features % n_heads == 0

        self.n_features = n_features
        self.n_heads = n_heads

        self._qkv_proj = nn.Linear(n_features, 3 * n_features)
        self._out_proj = nn.Linear(n_features, n_features)
        self._dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self._qkv_proj.weight, std=0.02)
        nn.init.normal_(self._out_proj.weight, std=0.02)

    def _split_heads(self, x, is_key: bool=False):
        x = x.reshape(x.shape[0], x.shape[1], self.n_heads, self.n_features // self.n_heads)
        x = x.permute(0, 2, 3, 1) if is_key else x.permute(0, 2, 1, 3)

        return x

    def _attn(self, q, k, v):
        w = torch.matmul(q, k) / math.sqrt(self.n_features // self.n_heads)
        w = F.softmax(w, dim=-1)
        w = self._dropout(w)
        out = torch.matmul(w, v)

        return out

    def _merge_heads(self, x):
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], self.n_features)

        return x

    def forward(self, query, key, value):
        query, key, value = self._qkv_proj(query).split(self.n_features, dim=-1)

        query = self._split_heads(query)
        key = self._split_heads(key, is_key=True)
        value = self._split_heads(value)

        x = self._attn(query, key, value)
        x = self._merge_heads(x)
        x = self._out_proj(x)

        return x


class FeedForward(nn.Module):
    def __init__(self, in_features, middle_features, dropout):
        super().__init__()

        self._layer_1 = nn.Linear(in_features, middle_features)
        self._layer_2 = nn.Linear(middle_features, in_features)
        self._activation = nn.PReLU()
        self._dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self._layer_1.weight, std=0.02)
        nn.init.normal_(self._layer_2.weight, std=0.02)

    def forward(self, x):
        x = self._layer_1(x)
        x = self._activation(x)
        x = self._dropout(x)
        x = self._layer_2(x)

        return x

class TransformerLayer(nn.Module):
    def __init__(self, n_features, n_heads=1, dropout=0.0, attn_dropout=0.0, ff_dropout=0.0):
        super().__init__()

        self._attn = MultiheadAttention(n_features, n_heads, attn_dropout)
        self._attn_norm = nn.LayerNorm(n_features)
        self._attn_dropout = nn.Dropout(dropout)
        self._attn_res = nn.GRUCell(n_features, n_features)
        self._ff = FeedForward(n_features, n_features, ff_dropout)
        self._ff_norm = nn.LayerNorm(n_features)
        self._ff_dropout = nn.Dropout(dropout)
        self._ff_res = nn.GRUCell(n_features, n_features)

    def _process_attn(self, x):
        b, l, d = x.shape

        residual = x
        x = self._attn_norm(x)
        x = self._attn(x, x, x)
        x = self._attn_dropout(x)
        x = self._attn_res(x.reshape(-1, d), residual.reshape(-1, d)).reshape(b, l, d)

        return x

    def _process_ff(self, x):
        b, l, d = x.shape

        residual = x
        x = self._ff_norm(x)
        x = self._ff(x)
        x = self._ff_dropout(x)
        x = self._ff_res(x.reshape(-1, d), residual.reshape(-1, d)).reshape(b, l, d)

        return x

    def forward(self, x):
        x = self._process_attn(x)
        x = self._process_ff(x)

        return x


class ImageNorm(nn.Module):
    def __init__(self, value=255, imagenet_norm=False):
        super().__init__()

        self._value = value
        self._imagenet_norm = imagenet_norm
        if imagenet_norm:
            self.register_buffer('_mean', torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).reshape(1, -1, 1, 1))
            self.register_buffer('_std', torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).reshape(1, -1, 1, 1))
    
    def forward(self, x):
        x = x.clamp(0, self._value) / self._value
        if self._imagenet_norm:
            x.sub_(self._mean).div_(self._std)
        return x


class GeMPool2d(nn.Module):
    """Fine-tuning CNN Image Retrieval with No Human Annotation: https://arxiv.org/pdf/1711.02512.pdf"""

    def __init__(self, in_channels=None, eps=1e-9):
        super().__init__()

        p = torch.ones(1) if in_channels is None else torch.ones(in_channels).reshape(1, -1, 1, 1)
        self._power = nn.Parameter(p)
        self._avg_pool = nn.AdaptiveAvgPool2d(1)
        self._eps = eps

    def forward(self, x):
        return self._avg_pool(x.clamp(self._eps).pow(self._power)).pow(1 / self._power)


class SpatialPyramidPolling(nn.Module):
    TYPES = ('avg', 'max')

    def __init__(self, sizes=((4, 4), (2, 2), (1, 1)), pool_type='avg'):
        super().__init__()

        if pool_type not in SpatialPyramidPolling.TYPES:
            raise ValueError(f'Incorrect pooling type: expected types {SpatialPyramidPolling.TYPES}, got {pool_type}')

        self._pools = nn.ModuleList()
        for size in sizes:
            if pool_type == 'max':
                self._pools.append(nn.AdaptiveMaxPool2d(size))
            else:
                self._pools.append(nn.AdaptiveAvgPool2d(size))

        self.n_elements = sum([s[0] * s[1] for s in sizes], 0)

    def forward(self, x):
        batch_size = x.shape[0]
        pool_outs = [p(x).reshape(batch_size, -1) for p in self._pools]
        out = torch.cat(pool_outs, dim=-1)

        return out


class FrozenBatchNorm2d(nn.Module):
    @classmethod
    def replace_batchnorm(cls, module):
        for name, sub_module in module.named_children():
            if isinstance(sub_module, nn.BatchNorm2d):
                frozen_bn = cls(sub_module.num_features)
                frozen_bn.weight = sub_module.weight.reshape(1, -1, 1, 1)
                frozen_bn.bias = sub_module.bias.reshape(1, -1, 1, 1)
                frozen_bn.running_mean = sub_module.running_mean.reshape(1, -1, 1, 1)
                frozen_bn.running_var = sub_module.running_var.reshape(1, -1, 1, 1)
                frozen_bn.eps = sub_module.eps

                setattr(sub_module, name, frozen_bn)
            else:
                FrozenBatchNorm2d.replace_batchnorm(sub_module)

    def __init__(self, num_features, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()

        self.register_buffer("weight", torch.ones(num_features).reshape(1, -1, 1, 1))
        self.register_buffer("bias", torch.zeros(num_features).reshape(1, -1, 1, 1))
        self.register_buffer("running_mean", torch.zeros(num_features).reshape(1, -1, 1, 1))
        self.register_buffer("running_var", torch.ones(num_features).reshape(1, -1, 1, 1))
        self.eps = eps

    def forward(self, x):
        scale = self.weight * (self.running_var + self.eps).rsqrt()
        bias = self.bias - self.running_mean * scale
        out = x * scale + bias

        return out


class SlotAttention(nn.Module):
    """Object-Centric Learning with Slot Attention: https://arxiv.org/abs/2006.15055
    Original code: https://github.com/lucidrains/slot-attention
    """
    def __init__(self, dim, num_slots, iters=3, mlp=False, eps=1e-8, hidden_dim=128):
        super().__init__()

        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim))

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)

        self.mlp = None
        if mlp:
            hidden_dim = max(dim, hidden_dim)
            self.mlp = nn.Sequential(nn.Linear(dim, hidden_dim),
                                     nn.PReLU(),
                                     nn.Linear(hidden_dim, dim))
            self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots=None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_sigma.expand(b, n_s, -1)
        slots = torch.normal(mu, sigma)

        inputs = self.norm_input(inputs)        
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(updates.reshape(-1, d), slots_prev.reshape(-1, d)).reshape(b, -1, d)

            if self.mlp is not None:
                slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots


class BorderCoords(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self._conv = nn.Conv2d(4, in_channels, 1)

    def __call__(self, x):
        b, c, h, w = x.shape

        y_coords = torch.arange(h).unsqueeze(1).expand(h, w) / (h - 1.0)
        x_coords = torch.arange(w).unsqueeze(0).expand(h, w) / (w - 1.0)
        rev_y_coords = torch.flip(y_coords, dims=[0])
        rev_x_coords = torch.flip(x_coords, dims=[1])

        coords = torch.stack((y_coords, x_coords, rev_y_coords, rev_x_coords), dim=0)

        coords = torch.unsqueeze(coords, dim=0).repeat(b, 1, 1, 1)
        coords_emb = self._conv(coords.type_as(x))

        return x + coords_emb


class SlotAttentionPool2d(nn.Module):
    def __init__(self, num_features, num_slots, n_iters=3, n_layers=1):
        super().__init__()

        self._slot_attn = SlotAttention(num_features, num_slots, n_iters)
        self._tr = nn.Sequential(*[TransformerLayer(num_features) for _ in range(n_layers)])
        self._p = nn.Parameter(torch.randn(1, 1, num_features))
        self._out_proj = nn.Linear(num_features, num_features)


    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).permute(0, 2, 1)

        x = self._slot_attn(x)
        x = self._tr(torch.cat([x, self._p.expand(b, -1, -1)], dim=1))[:, -1, :]

        out = self._out_proj(x)

        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels, groups=1):
        super().__init__()

        self._conv0 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=groups)
        self._conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=groups)
        self._gn1 = nn.GroupNorm(1, channels)  # TODO: change it?
        self._gn2 = nn.GroupNorm(1, channels)
        self._activation = nn.PReLU()
    
    def forward(self, x):
        inputs = x
        x = self._gn1(x)
        x = self._activation(x)
        x = self._conv0(x)
        x = self._gn2(x)
        x = self._activation(x)
        x = self._conv1(x)
        x = x + inputs
        return x


class ConvSequence(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, down=True):
        super().__init__()

        self._down = down
        self._conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self._res_block0 = ResidualBlock(out_channels, groups)
        self._res_block1 = ResidualBlock(out_channels, groups)

    def forward(self, x):
        x = self._conv(x)
        if self._down:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self._res_block0(x)
        x = self._res_block1(x)
        return x


class Impala(nn.Module):
    def __init__(self, in_channels, n_channels=32, groups=1):
        super().__init__()

        self.convs = nn.Sequential(ConvSequence(in_channels, n_channels // 2, groups),
                                   ConvSequence(n_channels // 2, n_channels, groups),
                                   ConvSequence(n_channels, n_channels, groups))
        self.avgpool = nn.Identity()
        self.activation = nn.PReLU()

        self.n_channels = n_channels
        n = int(64 / 2 ** len(self.convs))
        self.fc = nn.Linear(n_channels * n * n, n_channels)

    def forward(self, x):
        assert x.shape[-2:] == (64, 64)

        x = self.convs(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.activation(x)
        x = self.fc(x)
        return x


#=====================================================================


#ENCODERS
#=====================================================================

# TODO: mobilenet

import torchvision.models as resnet_models
import resnest.torch as resnest_models


class Encoder(nn.Module):
    _allowed_models = ['impala', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnest50', 'resnest101']

    def __init__(self, embedding_size, model, in_channels=3, normalize_input=True, normalize_value=255,
                 pretrained=False, freeze_batchnorm=False, dropout=0, coords=False, n_slots=None):
        super().__init__()

        if model not in self._allowed_models:
            raise ValueError(f'Wrong model: got {model}, expected one of {self._allowed_models}')

        self._embedding_size = embedding_size

        self._norm = None
        if normalize_input:
            imagenet_norm = pretrained and in_channels == 3 and model != 'impala'
            self._norm = ImageNorm(normalize_value, imagenet_norm)

        if model == 'impala':
            self._encoder = Impala(in_channels)

            pool_channels = self._encoder.n_channels
            pool = []

            if coords:
                pool.append(BorderCoords(pool_channels))

            if n_slots is not None and n_slots > 0:
                pool.append(nn.Sequential(nn.GroupNorm(1, pool_channels),
                                        nn.Conv2d(pool_channels, pool_channels, 1),
                                        nn.PReLU(),
                                        nn.Conv2d(pool_channels, pool_channels, 1)))
                pool.append(SlotAttentionPool2d(pool_channels, num_slots=n_slots, n_iters=3))
                pool.append(nn.Dropout(dropout))
                self._encoder.fc = nn.Linear(pool_channels, embedding_size)
            else:
                self._encoder.fc = nn.Linear(self._encoder.fc.in_features, embedding_size)

            self._encoder.avgpool = nn.Sequential(*pool)
        else:
            assert False

        '''else:
            if hasattr(resnet_models, model):
                model_cls = getattr(resnet_models, model)
            else:
                model_cls = getattr(resnest_models, model)
        
            self._encoder = model_cls(pretrained=pretrained)

            if in_channels != 3:
                self._encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

            if freeze_batchnorm:
                FrozenBatchNorm2d.replace_batchnorm(self._encoder)
            
            pool_channels = self._encoder.fc.in_features
            self._encoder.avgpool = SpatialPyramidPolling()

            assert False'''

        #init_recursive(self._encoder, activation='leaky_relu', param=0.25)
        self._out = nn.Sequential(nn.LayerNorm(embedding_size), nn.Tanh())

    def forward(self, x):
        if self._norm is not None:
            x = self._norm(x)
        emb = self._encoder(x).reshape(x.shape[0], -1)
        emb = self._out(emb)
        return emb

    @property
    def embedding_size(self):
        return self._embedding_size


class UpsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, kernel_size, stride=1, padding=0):
        super().__init__()

        self._layer = nn.Sequential(nn.Conv2d(in_channels, out_channels * scale_factor ** 2, kernel_size=1),
                                    nn.PixelShuffle(scale_factor),
                                    nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                                              stride=stride, padding=padding, bias=False),
                                    nn.BatchNorm2d(out_channels),
                                    nn.PReLU())

    def forward(self, x):
        return self._layer(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dims):
        super().__init__()

        self._base_size = 8
        self._coords = BorderCoords(latent_dim)
        self._conv = nn.Sequential(nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(latent_dim),
                                   nn.PReLU())

        in_channels = [latent_dim] + list(hidden_dims[:-1])
        out_channels = list(hidden_dims)

        self._upsample_layers = nn.ModuleList([UpsampleLayer(in_c, out_c, scale_factor=2, kernel_size=3, padding=1)
                                               for in_c, out_c in zip(in_channels, out_channels)])

        self._out_conv = nn.Conv2d(out_channels[-1], output_dim, kernel_size=3, padding=1)

    def forward(self, latent):
        x = latent[..., None, None].expand(-1, -1, self._base_size, self._base_size)
        x = self._coords(x)
        x = self._conv(x)

        for upsample_layer in self._upsample_layers:
            x = upsample_layer(x)

        x = self._out_conv(x)
        x = torch.sigmoid(x)

        return x


#=====================================================================


# TRANSFORMS
#=====================================================================

def random_crop(imgs, crop_size=64, pad_size=12):
    imgs = F.pad(imgs, pad=(pad_size, pad_size, pad_size, pad_size), mode='replicate')
    n, c, h, w = imgs.shape
    h1 = torch.randint(0, h - crop_size + 1, size=(n,))
    w1 = torch.randint(0, w - crop_size + 1, size=(n,))
    cropped = torch.empty((n, c, crop_size, crop_size), dtype=imgs.dtype, device=imgs.device)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cropped[i] = img[:, h11:h11 + crop_size, w11:w11 + crop_size]

    return cropped


#=====================================================================


# MODEL
#=====================================================================

class ProcgenModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name,
                 embedding_size, backbone, pretrained=False, freeze_batchnorm=False,
                 dropout=0, coords=False, n_slots=None):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        if len(obs_space.shape) == 3:
            in_channels = obs_space.shape[-1]
        elif len(obs_space.shape) == 4:
            in_channels = obs_space.shape[0] * obs_space.shape[-1]
        else:
            assert False, obs_space.shape

        self._backbone = Encoder(in_channels=in_channels,
                                 embedding_size=embedding_size,
                                 model=backbone,
                                 pretrained=pretrained,
                                 freeze_batchnorm=freeze_batchnorm,
                                 dropout=dropout,
                                 coords=coords,
                                 n_slots=n_slots)

        self._decoder = Decoder(latent_dim=embedding_size,
                                    output_dim=in_channels,
                                    hidden_dims=[64, 64, 32])

        self._hidden_fc = nn.Sequential(init(nn.Linear(self._backbone.embedding_size, 256), activation='leaky_relu', param=0.25),
                                        nn.PReLU())
        self._logits_fc = init(nn.Linear(256, num_outputs), gain=0.01)
        self._value_fc = init(nn.Linear(256, 1))

        self._emb = None
        self._value = None
        self._logits = None

    def _preprocess_obs(self, x, transforms=False):
        if x.dim() == 4:
            x = x.permute(0, 3, 1, 2)
        elif x.dim() == 5:
            b, n, h, w, c = x.shape
            x = x.permute(0, 1, 4, 2, 3).reshape(b, -1, h, w)
        else:
            assert False

        if transforms:
            x = random_crop(x, crop_size=64, pad_size=12)
        
        return x

    def _forward(self, x):
        emb = self._backbone(x)
        out = self._hidden_fc(emb)
        logits = self._logits_fc(out)
        value = self._value_fc(out)
        
        return logits, value, emb
        
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        is_training = input_dict['is_training']

        obs = self._preprocess_obs(obs)

        if is_training:
            self.train()
            logits, value, emb = self._forward(obs)
        else:
            self.eval()
            with torch.no_grad():
               logits, value, emb = self._forward(obs)

        self._value = value
        self._logits = logits
        self._emb = emb

        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._value is not None, "must call forward() first"
        return self._value.squeeze(1)

    @override(TorchModelV2)
    def custom_loss(self, policy_loss, loss_inputs):
        assert self._value is not None and self._logits is not None

        obs = loss_inputs["obs"].float()
        obs = self._preprocess_obs(obs, transforms=True)
        logits, value, emb = self._forward(obs)
        action_probas = F.softmax(self._logits.detach(), dim=-1)
        actions = torch.multinomial(action_probas, 1).squeeze(-1)

        alpha = 0.1
        self._transform_value_loss = alpha * 0.5 * F.mse_loss(value, self._value.detach())
        self._transform_policy_loss = alpha * F.cross_entropy(logits, actions)
        self._transform_loss = self._transform_value_loss + self._transform_policy_loss

        gamma = 0.1
        reconstruction = self._decoder(self._emb)
        self._rec_loss = gamma * F.l1_loss(reconstruction, obs / 255)
        
        z_gamma = 0.03
        self._reg_z_loss = z_gamma * self._emb.pow(2).sum(dim=-1).mean()
        
        #dec_gamma = 1e-5
        #self._reg_dec_loss = 0
        #for p in self._decoder.parameters():
        #    self._reg_dec_loss += dec_gamma * p.pow(2).sum()

        self._dec_loss = self._rec_loss + self._reg_z_loss

        return [loss_ + self._transform_loss + self._dec_loss for loss_ in policy_loss]

    @override(TorchModelV2)
    def metrics(self):
        return {'transform_value_loss': self._transform_value_loss.item(),
                'transform_policy_loss': self._transform_policy_loss.item(),
                'transform_loss': self._transform_loss.item(),
                'rec_loss': self._rec_loss.item(),
                'reg_z_loss': self._reg_z_loss.item(),
                #'reg_dec_loss': self._reg_dec_loss.item(),
                'dec_loss': self._dec_loss.item()}


ModelCatalog.register_custom_model("procgen_model", ProcgenModel)

#=====================================================================
