import torch, math
from safetensors import safe_open
from diffusers import ModelMixin
from diffusers.configuration_utils import FrozenDict
from .StateDictConverter import convert_state_dict_civitai_diffusers


class Timesteps(torch.nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps):
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
        timesteps = timesteps.unsqueeze(-1)
        emb = timesteps.float() * torch.exp(exponent)
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        return emb


class GEGLU(torch.nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = torch.nn.Linear(dim_in, dim_out * 2)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * torch.nn.functional.gelu(gate)


class Attention(torch.nn.Module):

    def __init__(self, query_dim, heads, dim_head, cross_attention_dim=None):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = torch.nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = torch.nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_out = torch.nn.Linear(inner_dim, query_dim, bias=True)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
    ):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        batch_size = encoder_hidden_states.shape[0]

        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        query = query.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)

        hidden_states = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        hidden_states = hidden_states.transpose(1, 2).view(batch_size, -1, self.heads * self.dim_head)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = self.to_out(hidden_states)

        return hidden_states


class BasicTransformerBlock(torch.nn.Module):

    def __init__(self, dim, num_attention_heads, attention_head_dim, cross_attention_dim):
        super().__init__()

        # 1. Self-Attn
        self.norm1 = torch.nn.LayerNorm(dim, elementwise_affine=True)
        self.attn1 = Attention(query_dim=dim, heads=num_attention_heads, dim_head=attention_head_dim)

        # 2. Cross-Attn
        self.norm2 = torch.nn.LayerNorm(dim, elementwise_affine=True)
        self.attn2 = Attention(query_dim=dim, cross_attention_dim=cross_attention_dim, heads=num_attention_heads, dim_head=attention_head_dim)

        # 3. Feed-forward
        self.norm3 = torch.nn.LayerNorm(dim, elementwise_affine=True)
        self.act_fn = GEGLU(dim, dim * 4)
        self.ff = torch.nn.Linear(dim * 4, dim)


    def forward(self, hidden_states, encoder_hidden_states):
        # 1. Self-Attention
        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=None,)
        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        norm_hidden_states = self.norm2(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
        hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.act_fn(norm_hidden_states)
        ff_output = self.ff(ff_output)
        hidden_states = ff_output + hidden_states

        return hidden_states


class DownSampler(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, hidden_states, time_emb, text_emb, res_stack):
        hidden_states = self.conv(hidden_states)
        return hidden_states, time_emb, text_emb, res_stack


class UpSampler(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, hidden_states, time_emb, text_emb, res_stack):
        hidden_states = torch.nn.functional.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        hidden_states = self.conv(hidden_states)
        return hidden_states, time_emb, text_emb, res_stack


class ResnetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels, groups=32, eps=1e-5):
        super().__init__()
        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.time_emb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = torch.nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.nonlinearity = torch.nn.SiLU()
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, hidden_states, time_emb, text_emb, res_stack):
        x = hidden_states
        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)
        emb = self.nonlinearity(time_emb)
        emb = self.time_emb_proj(emb)[:, :, None, None]
        x = x + emb
        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)
        if self.conv_shortcut is not None:
            hidden_states = self.conv_shortcut(hidden_states)
        hidden_states = hidden_states + x
        return hidden_states, time_emb, text_emb, res_stack


class AttentionBlock(torch.nn.Module):

    def __init__(self, num_attention_heads, attention_head_dim, in_channels, num_layers=1, cross_attention_dim=None, norm_num_groups=32):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = torch.nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = torch.nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                num_attention_heads,
                attention_head_dim,
                cross_attention_dim=cross_attention_dim
            )
            for d in range(num_layers)
        ])

        self.proj_out = torch.nn.Linear(inner_dim, in_channels)

    def forward(self, hidden_states, time_emb, text_emb, res_stack):
        batch, _, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=text_emb
            )

        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        hidden_states = hidden_states + residual

        return hidden_states, time_emb, text_emb, res_stack


class PushBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, time_emb, text_emb, res_stack):
        res_stack.append(hidden_states)
        return hidden_states, time_emb, text_emb, res_stack


class PopBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, time_emb, text_emb, res_stack):
        res_hidden_states = res_stack.pop()
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        return hidden_states, time_emb, text_emb, res_stack


class BlockUNet(ModelMixin):
    def __init__(self):
        super().__init__()
        self.time_proj = Timesteps(320)
        self.time_embedding = torch.nn.Sequential(
            torch.nn.Linear(320, 1280),
            torch.nn.SiLU(),
            torch.nn.Linear(1280, 1280)
        )
        self.add_time_proj = Timesteps(256)
        self.add_time_embedding = torch.nn.Sequential(
            torch.nn.Linear(2816, 1280),
            torch.nn.SiLU(),
            torch.nn.Linear(1280, 1280)
        )
        self.conv_in = torch.nn.Conv2d(4, 320, kernel_size=3, padding=1)

        self.blocks = torch.nn.ModuleList([
            # DownBlock2D
            ResnetBlock(320, 320, 1280),
            PushBlock(),
            ResnetBlock(320, 320, 1280),
            PushBlock(),
            DownSampler(320),
            PushBlock(),
            # CrossAttnDownBlock2D
            ResnetBlock(320, 640, 1280),
            AttentionBlock(10, 64, 640, 2, 2048),
            PushBlock(),
            ResnetBlock(640, 640, 1280),
            AttentionBlock(10, 64, 640, 2, 2048),
            PushBlock(),
            DownSampler(640),
            PushBlock(),
            # CrossAttnDownBlock2D
            ResnetBlock(640, 1280, 1280),
            AttentionBlock(20, 64, 1280, 10, 2048),
            PushBlock(),
            ResnetBlock(1280, 1280, 1280),
            AttentionBlock(20, 64, 1280, 10, 2048),
            PushBlock(),
            # UNetMidBlock2DCrossAttn
            ResnetBlock(1280, 1280, 1280),
            AttentionBlock(20, 64, 1280, 10, 2048),
            ResnetBlock(1280, 1280, 1280),
            # CrossAttnUpBlock2D
            PopBlock(),
            ResnetBlock(2560, 1280, 1280),
            AttentionBlock(20, 64, 1280, 10, 2048),
            PopBlock(),
            ResnetBlock(2560, 1280, 1280),
            AttentionBlock(20, 64, 1280, 10, 2048),
            PopBlock(),
            ResnetBlock(1920, 1280, 1280),
            AttentionBlock(20, 64, 1280, 10, 2048),
            UpSampler(1280),
            # CrossAttnUpBlock2D
            PopBlock(),
            ResnetBlock(1920, 640, 1280),
            AttentionBlock(10, 64, 640, 2, 2048),
            PopBlock(),
            ResnetBlock(1280, 640, 1280),
            AttentionBlock(10, 64, 640, 2, 2048),
            PopBlock(),
            ResnetBlock(960, 640, 1280),
            AttentionBlock(10, 64, 640, 2, 2048),
            UpSampler(640),
            # UpBlock2D
            PopBlock(),
            ResnetBlock(960, 320, 1280),
            PopBlock(),
            ResnetBlock(640, 320, 1280),
            PopBlock(),
            ResnetBlock(640, 320, 1280)
        ])

        self.conv_norm_out = torch.nn.GroupNorm(num_channels=320, num_groups=32, eps=1e-5)
        self.conv_act = torch.nn.SiLU()
        self.conv_out = torch.nn.Conv2d(320, 4, kernel_size=3, padding=1)

        # For diffusers
        self.config = FrozenDict([
            ('sample_size', 128), ('in_channels', 4), ('out_channels', 4), ('center_input_sample', False), ('flip_sin_to_cos', True),
            ('freq_shift', 0), ('down_block_types', ['DownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D']),
            ('mid_block_type', 'UNetMidBlock2DCrossAttn'), ('up_block_types', ['CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'UpBlock2D']),
            ('only_cross_attention', False), ('block_out_channels', [320, 640, 1280]), ('layers_per_block', 2), ('downsample_padding', 1),
            ('mid_block_scale_factor', 1), ('act_fn', 'silu'), ('norm_num_groups', 32), ('norm_eps', 1e-05), ('cross_attention_dim', 2048),
            ('transformer_layers_per_block', [1, 2, 10]), ('encoder_hid_dim', None), ('encoder_hid_dim_type', None), ('attention_head_dim', [5, 10, 20]),
            ('num_attention_heads', None), ('dual_cross_attention', False), ('use_linear_projection', True), ('class_embed_type', None),
            ('addition_embed_type', 'text_time'), ('addition_time_embed_dim', 256), ('num_class_embeds', None), ('upcast_attention', None),
            ('resnet_time_scale_shift', 'default'), ('resnet_skip_time_act', False), ('resnet_out_scale_factor', 1.0), ('time_embedding_type', 'positional'),
            ('time_embedding_dim', None), ('time_embedding_act_fn', None), ('timestep_post_act', None), ('time_cond_proj_dim', None),
            ('conv_in_kernel', 3), ('conv_out_kernel', 3), ('projection_class_embeddings_input_dim', 2816), ('attention_type', 'default'),
            ('class_embeddings_concat', False), ('mid_block_only_cross_attention', None), ('cross_attention_norm', None),
            ('addition_embed_type_num_heads', 64), ('_class_name', 'UNet2DConditionModel'), ('_diffusers_version', '0.20.2'),
            ('_name_or_path', 'models/stabilityai/stable-diffusion-xl-base-1.0\\unet')])
        self.add_embedding = FrozenDict([("linear_1", FrozenDict([("in_features", 2816)]))])

    def from_diffusers(self, safetensor_path=None, state_dict=None):
        # Load state_dict
        if safetensor_path is not None:
            state_dict = {}
            with safe_open(safetensor_path, framework="pt", device="cpu") as f:
                for name in f.keys():
                    state_dict[name] = f.get_tensor(name)

        # Analyze the architecture
        block_types = [block.__class__.__name__ for block in self.blocks]

        # Rename each parameter
        name_list = sorted([name for name in state_dict])
        rename_dict = {}
        block_id = {"ResnetBlock": -1, "AttentionBlock": -1, "DownSampler": -1, "UpSampler": -1}
        last_block_type_with_id = {"ResnetBlock": "", "AttentionBlock": "", "DownSampler": "", "UpSampler": ""}
        for name in name_list:
            names = name.split(".")
            if names[0] in ["conv_in", "conv_norm_out", "conv_out"]:
                pass
            elif names[0] in ["time_embedding", "add_embedding"]:
                if names[0] == "add_embedding":
                    names[0] = "add_time_embedding"
                names[1] = {"linear_1": "0", "linear_2": "2"}[names[1]]
            elif names[0] in ["down_blocks", "mid_block", "up_blocks"]:
                if names[0] == "mid_block":
                    names.insert(1, "0")
                block_type = {"resnets": "ResnetBlock", "attentions": "AttentionBlock", "downsamplers": "DownSampler", "upsamplers": "UpSampler"}[names[2]]
                block_type_with_id = ".".join(names[:4])
                if block_type_with_id != last_block_type_with_id[block_type]:
                    block_id[block_type] += 1
                last_block_type_with_id[block_type] = block_type_with_id
                while block_id[block_type] < len(block_types) and block_types[block_id[block_type]] != block_type:
                    block_id[block_type] += 1
                block_type_with_id = ".".join(names[:4])
                names = ["blocks", str(block_id[block_type])] + names[4:]
                if "ff" in names:
                    ff_index = names.index("ff")
                    component = ".".join(names[ff_index:ff_index+3])
                    component = {"ff.net.0": "act_fn", "ff.net.2": "ff"}[component]
                    names = names[:ff_index] + [component] + names[ff_index+3:]
                if "to_out" in names:
                    names.pop(names.index("to_out") + 1)
            else:
                raise ValueError(f"Unknown parameters: {name}")
            rename_dict[name] = ".".join(names)

        # Convert state_dict
        state_dict_ = {}
        for name, param in state_dict.items():
            state_dict_[rename_dict[name]] = param
        self.load_state_dict(state_dict_)

    def from_civitai(self, safetensor_path=None, state_dict=None):
        # Load state_dict
        if safetensor_path is not None:
            state_dict = {}
            with safe_open(safetensor_path, framework="pt", device="cpu") as f:
                for name in f.keys():
                    state_dict[name] = f.get_tensor(name)
        
        # Convert state_dict
        state_dict = convert_state_dict_civitai_diffusers(state_dict)
        self.from_diffusers(state_dict=state_dict)
        
    def process(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        added_cond_kwargs,
        **kwargs
    ):
        # 1. time
        t_emb = self.time_proj(timestep[None]).to(sample.dtype)
        t_emb = self.time_embedding(t_emb)
        
        time_embeds = self.add_time_proj(added_cond_kwargs["time_ids"])
        time_embeds = time_embeds.reshape((time_embeds.shape[0], -1))
        add_embeds = torch.concat([added_cond_kwargs["text_embeds"], time_embeds], dim=-1)
        add_embeds = add_embeds.to(sample.dtype)
        add_embeds = self.add_time_embedding(add_embeds)

        time_emb = t_emb + add_embeds
        
        # 2. pre-process
        hidden_states = self.conv_in(sample)
        text_emb = encoder_hidden_states
        res_stack = [hidden_states]
        
        # 3. blocks
        for i, block in enumerate(self.blocks):
            hidden_states, time_emb, text_emb, res_stack = block(hidden_states, time_emb, text_emb, res_stack)

        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        added_cond_kwargs,
        **kwargs
    ):
        hidden_states = []
        for i in range(sample.shape[0]):
            added_cond_kwargs_ = {}
            added_cond_kwargs_["text_embeds"] = added_cond_kwargs["text_embeds"][i:i+1]
            added_cond_kwargs_["time_ids"] = added_cond_kwargs["time_ids"][i:i+1]
            hidden_states.append(self.process(sample[i:i+1], timestep, encoder_hidden_states[i:i+1], added_cond_kwargs_))
        hidden_states = torch.concat(hidden_states, dim=0)
        return (hidden_states,)

