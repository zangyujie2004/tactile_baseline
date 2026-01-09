import einops
import numpy as np
import tqdm
from reactive_diffusion_policy.model.vae.model import VAE
from reactive_diffusion_policy.model.vae.block_vae_model import BlockEncodeVAE
from reactive_diffusion_policy.dataset.real_image_tactile_dataset import RealImageTactileDataset
from reactive_diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer

from reactive_diffusion_policy.dataset.real_image_tactile_dataset_reverse import RealImageTactileDatasetReverse

class RealImageTactileLatentDiffusionDataset(RealImageTactileDataset):
    def __init__(self,
                 at: VAE,
                 use_latent_action_before_vq: bool,
                 use_block_vae: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.at = at
        self.use_block_vae = use_block_vae
        self.at.eval()
        self.use_latent_action_before_vq = use_latent_action_before_vq

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = super().get_normalizer(**kwargs)

        latent_action_all = []

        for data in tqdm.tqdm(self, leave=False, desc='Calculating latent action for normalizer'):
            action = data['action'].to(self.at.device).unsqueeze(0)
            if not self.use_block_vae:
                action = normalizer['action'].normalize(action)
                latent_action = self.at.encoder(
                    self.at.preprocess(action / self.at.act_scale)
                )
                if self.at.use_vq:
                    if not self.use_latent_action_before_vq:
                        latent_action, _, _ = self.at.quant_state_with_vq(latent_action)
                else:
                    latent_action, _ = self.at.quant_state_without_vq(latent_action)
                if self.at.use_conv_encoder:
                    latent_action = einops.rearrange(latent_action, "N (T A) -> N T A", T=self.at.downsampled_input_h)
                else:
                    latent_action = einops.rearrange(latent_action, "N (T A) -> N T A", T=1)
                latent_action_all.append(latent_action[0].cpu().detach().numpy())
            else:
                self.at:BlockEncodeVAE = self.at
                batch_size, T, action_dim = action.shape # batch_size is 1
                action = action.reshape(batch_size*self.at.encode_block_num, T//self.at.encode_block_num, action_dim)
                latent_action = self.at.encode_to_latent(action,reshape_in_vae=False)
                latent_action = latent_action.reshape(batch_size, self.at.downsampled_input_h*self.at.encode_block_num, -1)
                latent_action_all.append(latent_action[0].cpu().detach().numpy())

        latent_action_all = np.concatenate(latent_action_all, axis=0)

        normalizer['latent_action'] = SingleFieldLinearNormalizer.create_fit(latent_action_all)

        return normalizer


class RealImageTactileLatentDiffusionDatasetReverse(RealImageTactileDatasetReverse):
    def __init__(self,
                 at: VAE,
                 use_latent_action_before_vq: bool,
                 **kwargs):
        super().__init__(**kwargs)
        self.at = at
        self.at.eval()
        self.use_latent_action_before_vq = use_latent_action_before_vq

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = super().get_normalizer(**kwargs)

        latent_action_all = []

        for data in tqdm.tqdm(self, leave=False, desc='Calculating latent action for normalizer'):
            action = data['action'].to(self.at.device).unsqueeze(0)
            action = normalizer['action'].normalize(action)
            latent_action = self.at.encoder(
                self.at.preprocess(action / self.at.act_scale)
            )
            if self.at.use_vq:
                if not self.use_latent_action_before_vq:
                    latent_action, _, _ = self.at.quant_state_with_vq(latent_action)
            else:
                latent_action, _ = self.at.quant_state_without_vq(latent_action)
            if self.at.use_conv_encoder:
                latent_action = einops.rearrange(latent_action, "N (T A) -> N T A", T=self.at.downsampled_input_h)
            else:
                latent_action = einops.rearrange(latent_action, "N (T A) -> N T A", T=1)
            latent_action_all.append(latent_action[0].cpu().detach().numpy())

        latent_action_all = np.concatenate(latent_action_all, axis=0)

        normalizer['latent_action'] = SingleFieldLinearNormalizer.create_fit(latent_action_all)

        return normalizer
