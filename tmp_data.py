import pickle
import os
import cv2
import imageio
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use a non-GUI backend
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# 'leftGripperCameraMarker1', 
# 'leftGripperCameraMarker2', 
# 'leftGripperCameraMarkerOffset1', 
# 'leftGripperCameraMarkerOffset2', 
# 'leftGripperCameraRGB1', 
# 'leftGripperCameraRGB2', 
# 'leftRobotGripperState', 
# 'leftRobotTCP', 
# 'leftRobotTCPVel', 
# 'leftRobotTCPWrench', 
# 'leftWristCameraPointCloud', 
# 'leftWristCameraRGB', 
# 'rightGripperCameraMarker1', 
# 'rightGripperCameraMarker2', 
# 'rightGripperCameraMarkerOffset1', 
# 'rightGripperCameraMarkerOffset2', 
# 'rightGripperCameraRGB1', 
# 'rightGripperCameraRGB2', 
# 'rightRobotGripperState', 
# 'rightRobotTCP', 
# 'rightRobotTCPVel', 
# 'rightRobotTCPWrench', 
# 'rightWristCameraPointCloud', 
# 'rightWristCameraRGB', 
# 'timestamp'

def visualize_tactile(save_path, tactile):
    xy0 = tactile[0, :, :2]
    xy1 = tactile[1, :, :2]

    fig = plt.figure(figsize=(8, 8))
    
    # 绘制 xy0 的点
    plt.scatter(xy0[:, 0], xy0[:, 1], c='blue', s=100, label='Start Points', marker='o')
    
    # 绘制箭头从 xy0 指向 xy1
    for start, end in zip(xy0, xy1):
        plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                  head_width=0.01, head_length=0.01, fc='blue', ec='blue')

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.box(True)
    plt.tight_layout(pad=2.0)
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def visualize_two_tactile(save_path, left_tactile, right_tactile):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    # 左手
    xy0_left = left_tactile[0, :, :2]
    xy1_left = left_tactile[1, :, :2]
    axs[0].scatter(xy1_left[:, 0], xy1_left[:, 1], c='red', s=30)
    axs[0].scatter(xy0_left[:, 0], xy0_left[:, 1], c='blue', s=30)
    axs[0].set_title('Left Hand Tactile')
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_xlabel('')
    axs[0].set_ylabel('')
    axs[0].set_box_aspect(1)
    axs[0].set_aspect('equal', adjustable='datalim')
    axs[0].spines['top'].set_visible(True)
    axs[0].spines['right'].set_visible(True)

    # 右手
    xy0_right = right_tactile[0, :, :2]
    xy1_right = right_tactile[1, :, :2]
    axs[1].scatter(xy1_right[:, 0], xy1_right[:, 1], c='red', s=30)
    axs[1].scatter(xy0_right[:, 0], xy0_right[:, 1], c='blue', s=30)
    axs[1].set_title('Right Hand Tactile')
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_xlabel('')
    axs[1].set_ylabel('')
    axs[1].set_box_aspect(1)
    axs[1].set_aspect('equal', adjustable='datalim')
    axs[1].spines['top'].set_visible(True)
    axs[1].spines['right'].set_visible(True)

    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=300)
    plt.close()

def reduce_and_reconstruct(data, n_components=15, normalize=True):
        '''
        pca reduction and reconstruction
        '''
        pca = PCA(n_components=n_components)
        if normalize:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
        X_pca = pca.fit_transform(data) 
        X_reconstructed = pca.inverse_transform(X_pca) 
        
        return X_pca, X_reconstructed

def get_pca_matrix(data, n_components=15):
    '''
    pca reduction
    '''
    pca = PCA(n_components=n_components)
    pca.fit(data)
    transform_matrix = pca.components_
    center_matrix = pca.mean_
    
    return transform_matrix, center_matrix

def reduce_data(data, transform_matrix, center_matrix):
    return (data - center_matrix) @ transform_matrix.T

def create_video_from_images(folder1, folder2, output_path, fps=10):
    images1 = sorted([os.path.join(folder1, img) for img in os.listdir(folder1) if img.endswith(('.png', '.jpg', '.jpeg'))])
    images2 = sorted([os.path.join(folder2, img) for img in os.listdir(folder2) if img.endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(images1) != len(images2):
        raise ValueError("两个文件夹中的图片数量不同！")

    # 读取第一张图片以获取初始尺寸
    img1 = cv2.imread(images1[0])
    img2 = cv2.imread(images2[0])
    
    # 获取初始高度和宽度
    height1, width1, _ = img1.shape
    height2, width2, _ = img2.shape
    
    # 选择最小高度和总宽度
    min_height = min(height1, height2)
    total_width = width1 + width2

    # 创建视频写入对象
    save_video_path = os.path.join(output_path, 'vis.mp4')
    writer = imageio.get_writer(save_video_path, fps=fps)

    save_video_image_path = os.path.join(output_path, 'video')
    os.makedirs(save_video_image_path, exist_ok=True)

    for img1_path, img2_path in zip(images1, images2):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        img1_resized = cv2.resize(img1, (int(width1 * min_height / height1), min_height))
        img2_resized = cv2.resize(img2, (int(width2 * min_height / height2), min_height))
        combined_img = np.hstack((img1_resized, img2_resized))

        cv2.imwrite(os.path.join(save_video_image_path, img1_path.split('/')[-1]), combined_img)
        combined_img = cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)
        writer.append_data(combined_img)

    writer.close()
     

if __name__ == '__main__':

    save_vis = False

    data_path = '/home/pc/workspace/zyh/rdp_data/peel_data/seq_01.pkl'
    save_dir = '/home/pc/workspace/zyh/rdp_data/peel_vis/01'
    
    save_rgb_path1 = os.path.join(save_dir, 'image1')
    save_rgb_path2 = os.path.join(save_dir, 'image2')
    save_tac_path1 = os.path.join(save_dir, 'tactile1')
    save_tac_path2 = os.path.join(save_dir, 'tactile2')
    save_main_tac_path1 = os.path.join(save_dir, 'main_tac1')
    save_main_tac_path2 = os.path.join(save_dir, 'main_tac2')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_rgb_path1, exist_ok=True)
    os.makedirs(save_rgb_path2, exist_ok=True)
    os.makedirs(save_tac_path1, exist_ok=True)
    os.makedirs(save_tac_path2, exist_ok=True)
    os.makedirs(save_main_tac_path1, exist_ok=True)
    os.makedirs(save_main_tac_path2, exist_ok=True)
    
    data = pickle.load(open(data_path, 'rb'))
    print('process data')

    left_tacs1 = list()
    right_tacs1 = list()
    left_tacs2 = list()
    right_tacs2 = list()
    timestamps = list()

    for frame_data in data.sensorMessages:
        left_tac_init1 = frame_data.leftGripperCameraMarker1
        left_tac_deform1 = frame_data.leftGripperCameraMarkerOffset1
        right_tac_init1 = frame_data.rightGripperCameraMarker1
        right_tac_deform1 = frame_data.rightGripperCameraMarkerOffset1

        left_tac_init2 = frame_data.leftGripperCameraMarker2
        left_tac_deform2 = frame_data.leftGripperCameraMarkerOffset2
        right_tac_init2 = frame_data.rightGripperCameraMarker2
        right_tac_deform2 = frame_data.rightGripperCameraMarkerOffset2

        left_tactile1 = np.stack((left_tac_init1, left_tac_init1 + left_tac_deform1), axis=0)
        left_tactile2 = np.stack((left_tac_init2, left_tac_init2 + left_tac_deform2), axis=0)
        right_tactile1 = np.stack((right_tac_init1, right_tac_init1 + right_tac_deform1), axis=0)
        right_tactile2 = np.stack((right_tac_init2, right_tac_init2 + right_tac_deform2), axis=0)

        timestamp = int(frame_data.timestamp * 1000)

        left_rgb = frame_data.leftWristCameraRGB
        right_rgb = frame_data.rightWristCameraRGB

        left_tacs1.append(left_tactile1)
        left_tacs2.append(left_tactile2)
        right_tacs1.append(right_tactile1)
        right_tacs2.append(right_tactile2)
        timestamps.append(timestamp)

        if save_vis:
            save_rgb_name1 = os.path.join(save_rgb_path1, str(timestamp) + '.png')
            save_rgb_name2 = os.path.join(save_rgb_path2, str(timestamp) + '.png')
            save_tac_name1 = os.path.join(save_tac_path1, str(timestamp) + '.png')
            save_tac_name2 = os.path.join(save_tac_path2, str(timestamp) + '.png')

            if left_rgb.shape[0] == 480 and left_rgb.shape[1] == 640:
                cv2.imwrite(save_rgb_name1, left_rgb)
            if right_rgb.shape[0] == 480 and right_rgb.shape[1] == 640:
                cv2.imwrite(save_rgb_name2, right_rgb)
            
            visualize_two_tactile(save_tac_name1, left_tactile1, left_tactile2)
            visualize_two_tactile(save_tac_name2, right_tactile1, right_tactile2)
    
    if save_vis:
        print('generate video')
        create_video_from_images(save_rgb_path1, save_tac_path1, save_dir)
    
    left_tacs1 = np.array(left_tacs1)
    left_tacs2 = np.array(left_tacs2)
    right_tacs1 = np.array(right_tacs1)
    right_tacs2 = np.array(right_tacs2)
    timestamps = np.array(timestamps)

    left_tac_deforms1 = left_tacs1[:,1, :,:] - left_tacs1[:,0,:,:]
    right_tac_deforms1 = right_tacs1[:,1,:,:] - right_tacs1[:,0,:,:]
    left_tac_deforms2 = left_tacs2[:,1,:,:] - left_tacs2[:,0,:,:]
    right_tac_deforms2 = right_tacs2[:,1,:,:] - right_tacs2[:,0,:,:]

    left_tac_deforms1 = left_tac_deforms1.reshape(left_tac_deforms1.shape[0], -1)
    right_tac_deforms1 = right_tac_deforms1.reshape(right_tac_deforms1.shape[0], -1)
    left_tac_deforms2 = left_tac_deforms2.reshape(left_tac_deforms2.shape[0], -1)
    right_tac_deforms2 = right_tac_deforms2.reshape(right_tac_deforms2.shape[0], -1)

    left_tac_inits1 = left_tacs1[:,0,:,:].mean(axis=0)
    right_tac_inits1 = right_tacs1[:,0,:,:].mean(axis=0)
    left_tac_inits2 = left_tacs2[:,0,:,:].mean(axis=0)
    right_tac_inits2 = right_tacs2[:,0,:,:].mean(axis=0)

    reduce, reconstruct = reduce_and_reconstruct(left_tac_deforms1, normalize=False)
    transform_matrix1, center_matrix1 = get_pca_matrix(left_tac_deforms1)
    transform_matrix2, center_matrix2 = get_pca_matrix(left_tac_deforms2)

    if save_vis:
        main_vectors1 = transform_matrix1.reshape(transform_matrix1.shape[0], -1, 2)
        for i in range(transform_matrix1.shape[0]):
            main_vector = left_tac_inits1 + 0.2 * main_vectors1[i]
            main_tac = np.stack((left_tac_inits1, main_vector), axis=0)
            save_path = os.path.join(save_main_tac_path1, str("%02d"%i) + '.png')
            visualize_tactile(save_path, main_tac)

        main_vectors2 = transform_matrix2.reshape(transform_matrix2.shape[0], -1, 2)
        for i in range(transform_matrix2.shape[0]):
            main_vector = left_tac_inits2 + 0.2 * main_vectors2[i]
            main_tac = np.stack((left_tac_inits2, main_vector), axis=0)
            save_path = os.path.join(save_main_tac_path2, str("%02d"%i) + '.png')
            visualize_tactile(save_path, main_tac)

        transform_matrix1 = np.load('/home/pc/workspace/zyh/reactive_diffusion_policy/data/PCA_Transform_GelSight/pca_transform_matrix.npy').transpose(1,0)
        transform_matrix2 = np.load('/home/pc/workspace/zyh/reactive_diffusion_policy/data/PCA_Transform_McTAC_v1/pca_transform_matrix.npy').transpose(1,0)
        for i in range(transform_matrix1.shape[0]):
            main_vector = left_tac_inits1 + 0.2 * main_vectors1[i]
            main_tac = np.stack((left_tac_inits1, main_vector), axis=0)
            save_path = os.path.join('/home/pc/workspace/zyh/rdp_data/wipe_vis/main_tac_pre1', str("%02d"%i) + '.png')
            visualize_tactile(save_path, main_tac)

        for i in range(transform_matrix2.shape[0]):
            main_vector = left_tac_inits2 + 0.2 * main_vectors2[i]
            main_tac = np.stack((left_tac_inits2, main_vector), axis=0)
            save_path = os.path.join('/home/pc/workspace/zyh/rdp_data/wipe_vis/main_tac_pre2', str("%02d"%i) + '.png')
            visualize_tactile(save_path, main_tac)

    print('end')
