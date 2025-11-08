# LatentDiffusionUnetImagePolicy
## predict_action
`state_vq.shape`: (1, 64)

`action_pred.shpae`: (1, 32, 64)
+ action_pred = state_vq.unsqueeze(1).expand(-1, self.original_horizon, -1)
+ `original_horizon` is 32
+ action_pred实际是被复制得到的


`action.shape`: (1,29,64)
+ action = action_pred[:, start:end]
+ `start` is 3
+ `end` is 32

`return result`
```
result = {
    'action': action,
    'action_pred': action_pred
}
```
+ 其中，action会被实际使用，即以静态的策略手动丢掉前面几个action