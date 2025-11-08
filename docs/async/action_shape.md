1.rdp的action是什么样的，slow和fast如何存储和使用
2.rdp的异步系统的逻辑是怎么样的，我们要写的话如何复用和改善

# ensemble buffer
## get_actoin
**fast通过get_action函数读取action**

`self.timestep`为0时

`return action`中，返回的action的shape为(74,), 即只是一个action，不是chunk

`actions = self.actions[0]`中，actions是一个list，长度为：1

`self.actions`的长度为25，即此时队列中有25个action chunk

## add_action
**slow通过add_action函数存入action**

`self.timestep`为0时

`horizon` is 25

`idx` is 0

`timestep` is 0

输入`action.shape` is (25, 74)