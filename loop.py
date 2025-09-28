if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
import copy
import random
import wandb
import numpy as np
from termcolor import cprint
from hydra.core.hydra_config import HydraConfig
from diffusion_policy_3d.policy.dp3 import DP3
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to

OmegaConf.register_new_resolver("eval", eval, replace=True)


class InferDP3Workspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, ckpt_path, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DP3 = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DP3 = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except: # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)

        # configure training state
        self.global_step = 0
        self.epoch = 0

        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)

        assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
        self.normalizer = dataset.get_normalizer()

        self.ckpt = ckpt_path

        self.optimizer = hydra.utils.instantiate(
        cfg.optimizer, params=self.model.parameters())

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.data_dict['action'],
            'agent_pos': self.data_dict['state'][...,:],
            'point_cloud': self.data_dict['point_cloud'],
            'tac_flow': self.data_dict['tac_flow_3d'],
            'tac_pos': self.data_dict['tac_pos_3d']
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def process_data(self, data_dict):
        tac_flow = data_dict['tac_flow']
        tac_pos = data_dict['tac_pos']
        point_cloud = data_dict['point_cloud']
        agent_pos = data_dict['state']

        tac_flow = torch.from_numpy(tac_flow).float()
        tac_pos = torch.from_numpy(tac_pos).float()
        point_cloud = torch.from_numpy(point_cloud).float()
        agent_pos = torch.from_numpy(agent_pos).float()

        data = {
            'obs': {
                'point_cloud': point_cloud, # T, 1024, 6
                'agent_pos': agent_pos, # T, D_pos
                'tac_flow': tac_flow,
                'tac_pos': tac_pos
            },
        }

        return data
        
    def run(self, data_dict):
        cfg = copy.deepcopy(self.cfg)
        device = torch.device(cfg.training.device)
        self.model.to(device)
        optimizer_to(self.optimizer, device)
        self.load_checkpoint(path=self.ckpt)

        # normalizer = self.get_normalizer()
        normalizer = self.normalizer
        self.model.set_normalizer(normalizer)
        
        cfg.logging.name = str(cfg.logging.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")
        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )
    
        # run validation
        policy = self.model
        with torch.no_grad():
            batch = self.process_data(data_dict)
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            obs_dict = batch['obs']
            result = policy.predict_action(obs_dict)
            pred_action = result['action']
            pred_action = pred_action.squeeze(0).cpu().numpy()
        return pred_action[:2]

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir
            

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload
    
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg):
    ckpt_path = '/data/zyp/Embodied/3D-Diffusion-Policy/3D-Diffusion-Policy/ckpt/real_blackboard/latest.ckpt'
    workspace = InferDP3Workspace(cfg, ckpt_path)
    while True:
        pts = np.ones((1, 2, 16384, 6))
        tac = np.ones((1, 2, 2, 2, 400, 3))
        state = np.ones((1, 2, 6))

        # get_input()

        data_dict = {'point_cloud': pts, 'tactile': tac, 'state': state}
        action = workspace.run(data_dict)
        print(action)

        # step()

if __name__ == "__main__":
    main()
