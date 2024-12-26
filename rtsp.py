from ultralytics import YOLO
import cv2

# 加载YOLO模型
model = YOLO("yolo11n.pt")

# 设置RTSP流
rtsp_url = "rtsp://admin:ht123456@192.168.1.101:554/streaming/channels/102"  # 替换成您的RTSP URL
cap = cv2.VideoCapture(rtsp_url)  # 打开RTSP流
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 设置缓冲区大小为1，减少延迟

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("打开视频流错误")

# 处理RTSP流中的帧，添加错误处理逻辑
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # 检查frame是否为空
        if frame is not None:
            results = model.track(frame, save=False)
            if hasattr(results, 'frame'):
                cv2.imshow('YOLO11 对象检测', results.frame)
            else:
                cv2.imshow('YOLO11 对象检测', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        print("无法从视频流中读取帧")
        break
    # 使用模型对帧进行检测
    results = model(frame, save=True, show=True)

# 如果任务完成则释放所有资源
cap.release()
cv2.destroyAllWindows()