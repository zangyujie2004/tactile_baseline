
### 减少无用的推理
real_runner_sync.py的RealRunner类的run函数中，run policy代码块附近
```
reduce_useless_inference = True
if reduce_useless_inference:
    if (step_count % self.tcp_action_update_interval == 0) or (step_count % self.gripper_action_update_interval == 0):
        with torch.no_grad():
            if self.use_latent_action_with_rnn_decoder:
                logger.info(f"推理推理, steps_per_inference is {steps_per_inference}")
                action_dict = policy.predict_action(obs_dict,
                                                    dataset_obs_temporal_downsample_ratio=self.dataset_obs_temporal_downsample_ratio,
                                                    return_latent_action=True)
            else:
                action_dict = policy.predict_action(obs_dict)
else:
    with torch.no_grad():
        if self.use_latent_action_with_rnn_decoder:
            logger.info(f"推理推理, steps_per_inference is {steps_per_inference}")
            action_dict = policy.predict_action(obs_dict,
                                                dataset_obs_temporal_downsample_ratio=self.dataset_obs_temporal_downsample_ratio,
                                                return_latent_action=True)
        else:
            action_dict = policy.predict_action(obs_dict)
```