import threading
import queue
import time
import sys
import select

def user_input_listener(input_queue):
    """后台线程，监听用户按回车"""
    while True:
        # 使用select监听是否有输入（非阻塞）
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            _ = sys.stdin.readline()  # 读取整行，但不使用内容
            if input_queue.empty():
                input_queue.put("ENTER")
        time.sleep(0.1)  # 避免占用CPU


def run():
    """主推理循环"""
    print("Start inference loop...")
    input_dict = dict()
    input_queue = queue.Queue()  # 用于接收用户输入事件

    # 启动独立线程监听键盘输入
    listener_thread = threading.Thread(target=user_input_listener, args=(input_queue,), daemon=True)
    listener_thread.start()


    # rossub_thread = threading.Thread(target=self.env.ros_thread, daemon=True)
    # rossub_thread.start()
    step_count = 0
    should_reverse = False
    # reverse_horizon = self.policy.reverse_length

    while True:
        # 🔹 检查是否有用户按下回车
        if not input_queue.empty():
            event = input_queue.get()
            if event == "ENTER":
                print("🚨 检测到用户按下回车，进入 reverse 模式!")
                should_reverse = True
                time.sleep(2)  # 暂停2秒
        time.sleep(0.3)
        print("又循环了一次")

run()