#!/bin/bash

# ================== 配置区域 ==================

PROJ_MAIN_DIR="/home/tars/projects/visual_tactile_policy/Tactile-Baseline/inference_node"
XARM_CMD="python ./gello_trajectory_pub_node.py"
CAMERA_CMD="python ./camera_pub_node.py"
TACTILE_CMD="python ./xense_pub_node.py"
GRIPPER_ACTIVATE_CMD="python ./gripper_activate.py"
GRIPPER_OPEN_CMD="python ./gripper_open.py"
GRIPPER_CLOSE_CMD="python ./gripper_close.py"
ROS_CMD="roscore"

# 终端窗口管理
MAIN_TERMINAL_TITLE="部署控制台"

# ================== 颜色定义 ==================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ================== 功能函数 ==================

# 关闭所有相关终端
close_all_terminals() {
    cd $PROJ_MAIN_DIR && python gripper_open.py && sleep 0.5 && python init_pos.py
    sleep 3
    echo -e "${YELLOW}正在关闭所有终端窗口...${NC}"
    pkill -f "gnome-terminal.*$MAIN_TERMINAL_TITLE" 2>/dev/null
    pkill -f "$CAMERA_CMD" 2>/dev/null
    pkill -f "$XARM_CMD" 2>/dev/null
    pkill -f "$TACTILE_CMD" 2>/dev/null
    pkill -f "$ROS_CMD" 2>/dev/null
    sleep 1
}


start_roscore() {
    echo -e "${YELLOW}[步骤] 启动RosCore...${NC}"
    pkill -f "gnome-terminal.*$MAIN_TERMINAL_TITLE" 2>/dev/null
    sleep 0.5
    gnome-terminal --title="$MAIN_TERMINAL_TITLE" --tab -- bash -c \
        "$ROS_CMD; exec bash"
    sleep 0.5
    return 0
}

start_inference() {
    XARM_CMD="python ./gello_trajectory_pub_node.py"
    CAMERA_CMD="python ./camera_pub_node.py"
    TACTILE_CMD="python ./xense_pub_node.py"
    
    ### STEP1: move the arm to initial position
    cd $PROJ_MAIN_DIR && python gripper_activate.py && python init_pos.py
    sleep 1 
    
    ### STEP2: Start all nodes
    # 启动触觉发布（同一终端的新标签页）
    if [ ! -d "$PROJ_MAIN_DIR" ]; then
        echo -e "${RED}错误：目录不存在 $PROJ_MAIN_DIR${NC}"
        return 1
    fi
    gnome-terminal --title="$MAIN_TERMINAL_TITLE" --tab -- bash -c \
        "cd '$PROJ_MAIN_DIR' && \
        echo -e '${YELLOW}发布触觉数据...${NC}' && \
        setsid $TACTILE_CMD & wait \$!; exec bash" &

    # 启动机械臂采发布（同一终端的新标签页）
    if [ ! -d "$PROJ_MAIN_DIR" ]; then
        echo -e "${RED}错误：目录不存在 $PROJ_MAIN_DIR${NC}"
        return 1
    fi
    gnome-terminal --title="$MAIN_TERMINAL_TITLE" --tab -- bash -c \
        "cd '$PROJ_MAIN_DIR' && echo -e '${YELLOW}发布轨迹数据...${NC}' && setsid $XARM_CMD & wait \$!; exec bash" &


    # 启动相机发布（同一终端的新标签页）
    if [ ! -d "$PROJ_MAIN_DIR" ]; then
        echo -e "${RED}错误：目录不存在 $PROJ_MAIN_DIR${NC}"
        return 1
    fi
    gnome-terminal --title="$MAIN_TERMINAL_TITLE" --tab -- bash -c \
        "cd '$PROJ_MAIN_DIR' && echo -e '${YELLOW}发布相机数据...${NC}' && $CAMERA_CMD; exec bash" &

    ### STEP3: Place the object and close the gripper (interaction)
    read -p "请将物体放置在机械臂末端，然后按任意键继续..." -n1 -s
    gnome-terminal --title="$MAIN_TERMINAL_TITLE" --tab -- bash -c \
        "cd '$PROJ_MAIN_DIR' && echo -e '${YELLOW}夹爪关闭...${NC}' && $GRIPPER_CLOSE_CMD; exec bash" &

}

# ================== 主菜单 ==================
show_menu() {
    clear
    echo -e "${GREEN}=== Gello-xArm-Dexhand 分步控制系统 ===${NC}"
    echo -e "1. 推理模式"
    echo -e "0. 退出并关闭所有终端"
    echo -n "请选择: "
}

# ================== 步骤执行 ==================
execute_step() {
    case $1 in
        1) 
            start_roscore
            start_inference ;;
        0) 
            close_all_terminals
            echo -e "${GREEN}系统已安全退出${NC}"
            exit 0 
            ;;
        *) echo -e "${RED}无效输入，请重试${NC}"; sleep 1 ;;
    esac
}

# ================== 主循环 ==================
while true; do
    show_menu
    read -r choice
    case $choice in
        [1]) execute_step "$choice" ;;
        0) execute_step 0 ;;
        *) echo -e "${RED}无效输入，请重试${NC}"; sleep 1 ;;
    esac
    echo -e "${YELLOW}按回车继续...${NC}"
    read -r
done