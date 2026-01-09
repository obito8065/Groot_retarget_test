from gr00t.utils.video import get_frames_by_timestamps

if __name__ == '__main__':
    data = get_frames_by_timestamps(
        video_path="/export1/vla/datasets/lerobot_oxe/roboturk_lerobot/videos/chunk-000/observation.images.front_rgb/episode_000456.mp4", # "/export1/vla/datasets/lerobot_oxe/roboturk_lerobot/videos/chunk-000/observation.images.front_rgb/episode_000559.mp4",
        timestamps=[20],
        video_backend='torchvision_av'
    )

    print(data)