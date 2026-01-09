import numpy as np
from mygrpc.client import RobotControlClient
import cv2
import time

class interact_with_grpc:
    def __init__(self):
        self.obs = None
        self.action = None
        self.grpc_client = RobotControlClient()

    def decompress_image(self, compressed_data):
        """
        解压缩图像数据，将二进制数据转换为图像数组。

        :param compressed_data: 压缩的二进制图像数据
        :return: 解压缩后的图像数组
        """
        # 将压缩的字节数据转换为 numpy 数组
        np_data = np.frombuffer(compressed_data, np.uint8)

        # 使用 OpenCV 解压缩图像
        image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("无法解压缩图像数据")

        return image

    def get_simulate_data(self, camera_name, get_origin=False):
        obs_origin = {
            'right_arm_pose': [0.014430999755859375, 0.39366498589515686, 0.5890570282936096, -0.49114400148391724,
                               0.01858000084757805, 0.033994000405073166, 0.8702160120010376],
            'right_hand_angle': [992.0, 1000.0, 994.0, 1000.0, 1000.0, 983.0],
            'left_arm_pose': [0.5743520259857178, -0.06948699802160263, 0.40091800689697266, -0.6380209922790527,
                              0.32698801159858704, -0.6356760263442993, 0.2862190008163452],
            'left_hand_angle': [996.0, 1000.0, 1000.0, 1000.0, 1000.0, 335.0]}
        if get_origin:
            return obs_origin
        else:
            obs = {
                "right_arm_pose": np.array(obs_origin['right_arm_pose']),
                "right_hand_angle": np.array(obs_origin['right_hand_angle']),
                "left_arm_pose": np.array(obs_origin['left_arm_pose']),
                "left_hand_angle": np.array(obs_origin['left_hand_angle']),
            }
            img = {}
            for name in camera_name:
                img[name] = np.zeros(
                    (720, 1280, 3), dtype=np.uint8
                )
            return img, obs

    def get_state_grpc(self, camera_name=None, simulate=False):

        if simulate:
            return self.get_simulate_data(camera_name)

        obs_origin = self.grpc_client.get_response_only()
        obs = {
            "right_arm_pose": np.array(obs_origin['right_arm_pose']),
            "right_hand_angle": np.array(obs_origin['right_hand_angle']),
            "left_arm_pose": np.array(obs_origin['left_arm_pose']),
            "left_hand_angle": np.array(obs_origin['left_hand_angle']),
        }
        img = {}
        for name in camera_name:
            if name in obs_origin:
                img[name] = self.decompress_image(obs_origin[name])
            else:
                raise ValueError(
                    f"Unsupported image name: not found {name}"
                )

        return img, obs

    def pub_action_grpc(self, action):
        obs = self.grpc_client.step(right_action=action)

    def single_step_pub_action(self, action, camera_name=None, simulate=False):
        if simulate:
            return self.get_simulate_data(camera_name, get_origin=True)
        obs_origin = self.grpc_client.step(right_action=action)
        return obs_origin

    def get_obs(self, obs_origin, camera_name):
        assert obs_origin is not None
        obs = {
            "right_arm_pose": np.array(obs_origin['right_arm_pose']),
            "right_hand_angle": np.array(obs_origin['right_hand_angle']),
            "left_arm_pose": np.array(obs_origin['left_arm_pose']),
            "left_hand_angle": np.array(obs_origin['left_hand_angle']),
        }
        img = {}
        for name in camera_name:
            if name in obs_origin:
                img[name] = self.decompress_image(obs_origin[name])
            else:
                raise ValueError(
                    f"Unsupported image name: not found {name}"
                )

        return img, obs

    def pub_action_and_get_obs(self, action, camera_name=None, simulate=False):

        if simulate:
            return self.get_simulate_data(camera_name)

        obs_origin = self.grpc_client.step(right_action=action)
        obs = {
            "right_arm_pose": np.array(obs_origin['right_arm_pose']),
            "right_hand_angle": np.array(obs_origin['right_hand_angle']),
            "left_arm_pose": np.array(obs_origin['left_arm_pose']),
            "left_hand_angle": np.array(obs_origin['left_hand_angle']),
        }
        img = {}
        for name in camera_name:
            if name in obs_origin:
                img[name] = self.decompress_image(obs_origin[name])
            else:
                raise ValueError(
                    f"Unsupported image name: not found {name}"
                )

        return img, obs

    def test_img_time(self):
        img = {}
        obs_origin = {
            'left': np.zeros((720, 1280, 3), dtype=np.uint8),
        }
        camera_name = ['left']
        for name in camera_name:
            if name in obs_origin:
                success, compressed_data = cv2.imencode('.jpg', obs_origin[name])
                res = compressed_data.tobytes()
                img[name] = self.decompress_image(res)
            else:
                raise ValueError(
                    f"Unsupported image name: not found {name}"
                )

# if __name__ == '__main__':
#     interact = interact_with_grpc()
#     for i in range(20):
#         start = time.time()
#         for j in range(20):
#             interact.test_img_time()
#         end = time.time()
#         print(f"cost time {(end - start)*1000}")