import torch.nn as nn
import torch
import math
import einops

from typing import Dict

from efficient_robot_sys.models.reactive_diffusion_policy.model.vae_unet.unet1d import ConditionalUnet1D
from efficient_robot_sys.models.Tactile_Generation_Policy.tactile_generation_policy.model.action.vae_utils import DiagonalGaussianDistribution
from efficient_robot_sys.models.reactive_diffusion_policy.model.common.normalizer import LinearNormalizer

class ActionVAE(nn.Module):
    """
    ActionVAE 的 Docstring \\
    encoder和decoder的输入输出的shape都为 [B, T, A]. 将latent dim设置为action_dim(A), 方便直接接入现有的policy中 \\
    normalize操作在模型内部进行, 训练的时候需要先set_normalizer
    """
    def __init__(
        self,
        input_dim=10,
        horizon=10,
        kl_multiplier=1e-6,
        device="cuda",
        load_dir=None,
        act_scale=1.0,
    ):
        """
        __init__ 的 Docstring
        
        :param self: 说明
        :param input_dim: 说明
        :param kl_multiplier: 说明
        :param device: 说明
        :param load_dir: 说明
        :param act_scale: 说明
        """
        super().__init__()
        self.input_dim = input_dim
        self.horizon = horizon

        self.kl_multiplier = kl_multiplier
        self.device = device
        self.act_scale = act_scale

        self.encoder = ConditionalUnet1D(input_dim=input_dim)
        self.decoder = ConditionalUnet1D(input_dim=input_dim)

        # 输入输出为 (b, T*A)
        self.quant = torch.nn.Conv1d(self.input_dim, 2*self.input_dim, 1)
        self.post_quant = torch.nn.Conv1d(self.input_dim, self.input_dim, 1)

        self.normalizer = LinearNormalizer()

        self.optim_params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
        )
        self.optim_params += list(self.quant.parameters())
        self.optim_params += list(self.post_quant.parameters())


        if load_dir is not None:
            print(f"load_dir is {load_dir}")
            try:
                state_dict = torch.load(load_dir)
            except RuntimeError:
                state_dict = torch.load(load_dir, map_location=torch.device("cpu"))
            self.load_state_dict(state_dict)


    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())        
    
    def get_action_from_latent(self, latent):
        timestep = torch.Tensor([0]).to(latent.device)
        output = self.decoder(latent,timestep=timestep) * self.act_scale
        output = self.normalizer['action'].unnormalize(output)
        return output
    
    def get_latent_from_action(self, action):
        state = self.normalizer['action'].normalize(action)
        state = state / self.act_scale
        timestep = torch.Tensor([0]).to(state.device)
        state_rep = self.encoder(state.float(), timestep=timestep)
        state_vq, posterior = self.quant_state(state_rep)
        state_vq = self.postprocess_quant_state(state_vq)
        return state_vq
    
    def quant_state(self, state):
        """
        输出N T A \\
        输出N A T
        """
        state = einops.rearrange(state, "N T A -> N A T")

        moments = self.quant(state)
        posterior = DiagonalGaussianDistribution(moments)
        state_vq = posterior.sample() # N A T
        # state_vq = einops.rearrange(state_vq, "N A T -> N (T A)")

        return state_vq, posterior
    
    def postprocess_quant_state(self, state_vq):
        """
        输入 [N A T] \\
        输出 [N T A]
        """
        # state_vq = einops.rearrange(state_vq, "N (T A) -> N A T", T=self.downsampled_input_h)
        state_vq = self.post_quant(state_vq)
        state_vq = einops.rearrange(state_vq, "N A T -> N T A")

        return state_vq
    
    def calculate_loss(self, state):
        """
        输入 [N T A]
        """
        if isinstance(state, Dict):
            state = state["action"]

        state = self.normalizer['action'].normalize(state)

        timestep = torch.Tensor([0]).to(state.device)
        state = state / self.act_scale

        state_rep = self.encoder(state.float(), timestep=timestep)

        state_vq, posterior = self.quant_state(state_rep)
        
        state_vq = self.postprocess_quant_state(state_vq)

        dec_out = self.decoder(state_vq, timestep=timestep)

        encoder_loss = (state - dec_out).abs().mean()
        vae_recon_loss = torch.nn.MSELoss()(state, dec_out)
        kl_loss = posterior.kl().mean() * self.kl_multiplier
        loss = encoder_loss + kl_loss

        return loss, encoder_loss, kl_loss 
    
    def encode_then_decode(self, state):
        """
        state [N, T, A] should be torch.Tensor(device) which has been normalized
        """
        if isinstance(state, Dict):
            state = state["action"]

        latent = self.get_latent_from_action(state)
        action = self.get_action_from_latent(latent)
        return action
        # state = self.normalizer['action'].normalize(state)

        # action_dim = state.shape[-1]
        # timestep = torch.Tensor([0]).to(state.device)
        # state = state / self.act_scale

        # state_rep = self.encoder(state.float(),timestep=timestep)

        # state_vq, posterior = self.quant_state(state_rep)
        
        # state_vq = self.postprocess_quant_state(state_vq)

        # dec_out = self.decoder(state_vq, timestep = timestep)

        # dec_out = dec_out * self.act_scale
        
        # dec_out = self.normalizer['action'].unnormalize(dec_out)

        # return dec_out

if __name__ == '__main__':
    import pickle
    import sys
    sys.path.append("/home/kywang/projects/Tactile-Baseline")
    device = 'cuda'

    normalizer_path = "/home/kywang/projects/Tactile-Baseline/data/outputs/2025.12.08/05.33.25_train_diffusion_unet_image_vase_sponge_test1_dp_ddim30_60hz/normalizer.pkl"
    normalizer = pickle.load(open(normalizer_path, 'rb'))
    vae = ActionVAE(device=device)
    vae.set_normalizer(normalizer)

    B = 2
    T = 32
    A = 10
    sample = torch.randn(B,T,A).to(device)

    loss = vae.calculate_loss(sample)
    print(f"loss is {loss}")

    state = vae.encode_then_decode(sample)
    print(f"state is {state}")


