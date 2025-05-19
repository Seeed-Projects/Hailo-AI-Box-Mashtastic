import cv2

# RTSP 流地址，替换为你的摄像头/设备的 RTSP 地址
rtsp_url = 'rtsp://admin:12345678a@10.0.0.108:554/cam/realmonitor?channel=1&subtype=0'

# 创建视频捕获对象
cap = cv2.VideoCapture(rtsp_url)

# 检查是否成功连接
if not cap.isOpened():
    print("无法打开 RTSP 流")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取帧")
        break

    # 显示视频帧
    cv2.imshow("RTSP Stream", frame)

    # 按下 q 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
