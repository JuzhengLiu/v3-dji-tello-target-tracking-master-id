pip install torch torchvision torchaudio --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu121

conda activate D:\7_deepl\environment\tt_track




1.无人机测试（网络摄像头演示）
# Run with default settings (YOLOv8n, detect all classes)
python demo_webcam.py

# Use a better model
python demo_webcam.py --model yolov8s --classes person --confidence 0.6

# Track specific objects only
python demo_webcam.py --classes person ball

# Use a video file instead of webcam
python demo_webcam.py --video test_video.mp4

q - Quit  q - 退出
t - Toggle tracking on/off
t - 开关跟踪开关
d - Switch detector (YOLO/HSV)
d - 开关检测器（YOLO/HSV）
r - Reset tracker
r - 重置追踪器
s - Save screenshot
s - 保存截图
h - Toggle HUD  h - 切换 HUD
f - Toggle FPS display
f - 切换帧率显示



2.驾驶无人机飞行
# First, test with mock drone (uses webcam, no hardware)
python demo_drone.py --mock

# When ready, fly for real
python demo_drone.py

# With custom settings
python demo_drone.py --model yolov8s --confidence 0.6 --speed 50
python demo_drone.py --classes person --model yolov8n --confidence 0.6 --speed 50

python demo_drone.py --classes person --model yolov8n --confidence 0.6 --speed 80

python demo_drone_id.py --classes person --model yolov8n --confidence 0.6 --speed 70
tap->SPACE->BACKSPACE


Drone Demo Controls:  无人机演示控制：
TAB - 起飞
BACKSPACE - 降落
ESC - 紧急停止
SPACE - 切换自主跟踪
w/s/a/d - 手动控制（前/后/左/右）
↑/↓ - 手动高度控制
←/→ - 手动旋转
r - 录制视频
c - Take photo  c - 拍照
q - 退出（先落地）




drone_controller.py
lr = int(np.clip(lr, -100, 100))
        fb = int(np.clip(fb, -100, 100))
        ud = int(np.clip(ud, -100, 100))
        yaw = int(np.clip(yaw, -100, 100))


当前无人机可能会出现跟踪失效的情况，比如无人机识别到画面的情况下，人物在剧烈运动（并不只是画面中直线运动，而是左右猛转），
移除画面中的时候，可能会出现跟丢的情况，针对这一情况调整程序。

此外，调整程序，我认为目前无人机跟随时的飞行高度不够高，请你相较于当前的高度，努力实现相较于当前的高度，再维持稳定提高50cm不变高度的情况。

1.高度
2.识别到人后再识别人头等关键部位






