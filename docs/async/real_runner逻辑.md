# init
`use_latent_action_with_rnn_decoder`, rdp.yaml中设定为true。为true时，ensemble_buffer也会被指定为true

`self.env`, 为sensors.py中的`RealRobotEnv`类。这个类负责和ros进行交互，从而实际控制机器人

`policy`不是通过self.policy来调用，在`train_diffusion_unet_real_image_workspace`中指定为`latent_diffusion_une_image_policy`

`latency_step`: 4: 从policy得到action(一个chunk，并且已经丢掉了len(obs)*downsampleratio个step的动作了)。然后再丢掉latency_step个action。
+ 这个值匹配inference_fps // control_fps
+ 由于inference_fps和control_fps是写死的，所以这个latency_step是写死的，因此：虽然policy每次能够预测32个action，但会以固定的值丢掉`过时`的action
+ 因为这个值是写死的，所以当实际的control_fps或者inference_fps和预定的值不一样的时候，就会mismatch

`gripper_latency_step`: 8
+ 和latency_step同样的道理

`control_fps`: 24

`inference_fps`: 6

`steps_per_inference` = int(self.control_fps / self.inference_fps)
+ 4. 每次推理，step_count增加4

`step_count`:
+ 初始化为0，每次主循环会 += `steps_per_inference`

`tcp_action_update_interval`: 16

`gripper_action_update_interval`: 16

# ensemble_buffer
分为tcp_ensemble_buffer和gripper_ensemble_buffer
这两个buffer，可以是latent版本，或者是非latent版本。在rdp.yaml中指定为latent版本


# run
## main循环之内
`action_all` = np_action_dict['action'].squeeze(0)
+ 从(1, 29, 74)变为(29, 74)


`self.inference_interval_time` = 1.0 / inference_fps
+ precise_sleep(max(0., self.inference_interval_time - (cur_time - start_time)))
+ 如果一次推理完成之后，时间不足interval_time，就会sleep
+ 硬编码系统

+ tcp_action和gripper_action
```
......
if action_all.shape[-1] == 4:
    tcp_action = action_all[self.latency_step:, :3]
......
if action_all.shape[-1] == 4:
    gripper_action = action_all[self.gripper_latency_step:, 3:]
```

## 概览
```
while true
    # 推理得到action_latent
    action_dict = policy.predict_action
    ......
    # 基于action_latent，朝队列中中添加action
    if step_count % self.tcp_action_update_interval == 0: #16
        self.tcp_ensemble_buffer.add_action(tcp_action, step_count)

    if step_count % self.gripper_action_update_interval == 0: #16
        self.gripper_ensemble_buffer.add_action(gripper_action, step_count)
    ......
    # 微调action，并且发送action指令给ros
    self.action_command_thread(policy, self.stop_event)

    precise_sleep(max(0., self.inference_interval_time - (cur_time - start_time)))

    # 更新step_count
    step_count += steps_per_inference #4
```
+ 即每`16/4->4`次inference，才更新队列一次。。。
+ 【实验验证】：四step推理，一step更新
+ Slow inference time: 0.1s
+ fast inference time: 0.001s
+ fast total time: 0.3s(加上执行以及传感器的时间，波动很大)
+ run time: 1.8s 运行一次主循环的时间, 波动很大
+ 每次调用slow或者fast之前，都需要get_obs(). 它本身latency不到1ms，但是从采得到policy获取的latency，有44ms。也就是说，会有44ms的observation过时
+ fast execution_action时间: 0.3s. 执行的时间长达0.3s～0.9s，是真正的bottleneck。

## 频率有关
+ control_fps之前设置为24hz. 实际fast一次需要0.3s以上，因此实际control_fps为3左右
+ inference_fps之前设置为6hz. 实际一次main需要1.8以上，因此实际inference_fps不到1
+ steps_per_inference = int(self.control_fps / self.inference_fps)。 用于预估一次slow inference之后（即一次main函数的step），control了几步。现有参数算下来，steps_per_inference是4
+ tcp_action_update_interval % tcp_action_update_interval为0时，才更新队列。由于tcp_action_update_interval默认为16，因此4次main函数step，才会真正更新一次队列。也就是7s以上才更新一次队列（profile得到的结果是6s左右）。。。

## 思考
既然bottleneck其实是执行，那么，将每次执行的步长减少，才是丝滑的前提条件。
+ 可能做精细任务的时候，布长会自动减少，从而control_fps会上去，从而inference_fps也会上去
+ 起码根据现在的实验，前面届阶段可以6s

# action_command_thread
`combined_action` = np.concatenate([tcp_step_action, gripper_step_action], axis=-1)
+ tcp_step_action和gripper_step_action平在一起
+ 这两个action分别用一个ensemble队列进行存储，不是'一个代表action，一个代表当前状态'
+ 但是latency_step和gripper_latency_step不同，他们怎么拼接在一起呢。可能因为tcp_action是一个chunk，而实际拼接的tcp_step_action是一个step的，所以它们可以拼接在一起

`control_interval_time` = 1.0 / control_fps

`tcp_step_action` = self.tcp_ensemble_buffer.get_action()
+ shape is (74, )

`gripper_step_action` = self.gripper_ensemble_buffer.get_action()
+ shape is (74, )

self.env.execute_action(step_action, use_relative_action=False, is_bimanual=is_bimanual)
+ step_action, is_bimanual = self.post_process_action(combined_action[np.newaxis, :]). step_action.shape is (16,)
+ combined_action = np.concatenate([tcp_step_action, gripper_step_action], axis=-1). combined_action.shape is 10
+ tcp_step_action = tcp_step_action[:tcp_len] shape is (9)
+ gripper_step_action = gripper_step_action[tcp_len:] shape is (1)


