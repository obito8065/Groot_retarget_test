import cv2
import os
import glob

video_dir = '/vla/users/hezeyuan/rm_lerobot/take_cup_zeyuan/videos/chunk-000/observation.images.cam_head_left'
video_files = sorted(glob.glob(os.path.join(video_dir, '*.mp4')))

if not video_files:
    print("未找到任何mp4文件")
    exit()

# --- 1. 先在第一个视频的第一帧选ROI ---
cap = cv2.VideoCapture(video_files[0])
ret, first_frame = cap.read()
cap.release()
if not ret:
    print("无法读取第一个视频的第一帧")
    exit()

drawing = False
ix, iy = -1, -1
rect = None
temp_frame = first_frame.copy()

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rect, temp_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        temp_frame = first_frame.copy()
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_frame = first_frame.copy()
            cv2.rectangle(temp_frame, (ix, iy), (x, y), (0, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect = (min(ix, x), min(iy, y), max(ix, x), max(iy, y))
        cv2.rectangle(temp_frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)

cv2.namedWindow('Select ROI')
cv2.setMouseCallback('Select ROI', draw_rectangle)
while True:
    cv2.imshow('Select ROI', temp_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 13 or key == 32:  # Enter或Space确认
        if rect is not None:
            break
    elif key == 27:  # Esc退出
        cv2.destroyAllWindows()
        exit()
cv2.destroyWindow('Select ROI')

# --- 2. 对所有视频文件依次处理 ---
for video_path in video_files[::5]:
    cap = cv2.VideoCapture(video_path)
    print(f'正在处理: {video_path}')
    count = 0
    while True:
        count += 1
        if count % 500 == 0:

            ret, frame = cap.read()
            if not ret:
                break
            # 裁剪
            crop = frame[rect[1]:rect[3], rect[0]:rect[2]]
            print(rect, rect[3]-rect[1], rect[2]-rect[0])
            # 显示
            cv2.imshow('Original Frame', frame)
            cv2.imshow('Cropped Frame', crop)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()
    cap.release()
cv2.destroyAllWindows()