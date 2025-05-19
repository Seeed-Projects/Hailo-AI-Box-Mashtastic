import supervision as sv
import numpy as np
import cv2
import queue
import sys
import os
import threading
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict
from multiprocessing import shared_memory, Process, Queue
import meshtastic
import meshtastic.serial_interface
from pubsub import pub
import multiprocessing
import time
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import HailoAsyncInference
from typing import List, Tuple, Dict, Set

@dataclass
class Config:
    net: str
    input_video: str
    labels: str
    score_thresh: float
    point_of_line: List[Tuple[int, int]]
    send_node: int

def load_config(path: str = "config.json") -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Config(**data)

def preprocess_frame(frame: np.ndarray, model_h: int, model_w: int, video_h: int, video_w: int) -> np.ndarray:
    return cv2.resize(frame, (model_w, model_h)) if (model_h != video_h or model_w != video_w) else frame

def send_message_to_node(message: dict, interface) -> None:
    """
    Send a message to the specified node ID using Meshtastic.
    """
    if isinstance(message, dict):
        message = json.dumps(message, ensure_ascii=False)
    # Send the message
    interface.sendText(message, channelIndex=0)
    print(f"[✓] Sented")

def get_point_side(point: Tuple[float, float], line: Tuple[Tuple[float, float], Tuple[float, float]]) -> float:
    (x1, y1), (x2, y2) = line
    px, py = point
    # 向量叉积
    return (x2 - x1)*(py - y1) - (y2 - y1)*(px - x1)

def classify_people(
    line: List[Tuple[float, float]],
    information_queue: Queue,
    message_queue: Queue,
    interface: meshtastic.serial_interface.SerialInterface,
    config: Config,
) -> None:
    # 当前帧与上一帧的 tracker_id 分布
    set_A_old = set()
    set_B_old = set()

    # 累计跨区人数
    total_A_to_B = 0
    total_B_to_A = 0

    # 控制发送频率
    frame_count = 0

    while True:
        try:
            detections = json.loads(information_queue.get(timeout=0.1))
            if not detections:
                continue

            set_A = set()
            set_B = set()

            for det in detections:
                tracker_id = det["tracker_id"]
                x_min, y_min, x_max, y_max = det["bbox"]
                cx = (x_min + x_max) / 2
                cy = (y_min + y_max) / 2
                side = get_point_side((cx, cy), line)

                if side > 0:
                    set_A.add(tracker_id)
                elif side < 0:
                    set_B.add(tracker_id)

            # 跨区域判断（当前帧 vs 上一帧）
            A_to_B = len(set_A_old & set_B)  # 上一帧在 A，当前帧在 B
            B_to_A = len(set_B_old & set_A)  # 上一帧在 B，当前帧在 A

            total_A_to_B += A_to_B
            total_B_to_A += B_to_A

            # 更新历史帧
            set_A_old = set_A
            set_B_old = set_B

            frame_count += 1
            if frame_count >= 1000:
                message = {
                    "A_to_B": total_A_to_B,
                    "B_to_A": total_B_to_A,
                    "A": len(set_A),
                    "B": len(set_B),
                }

                print("[✓] Message:", message)

                send_message_to_node(
                    config.send_node,
                    json.dumps(message, ensure_ascii=False),
                    interface
                )

                frame_count = 0
                total_A_to_B = 0
                total_B_to_A = 0

        except queue.Empty:
            time.sleep(0.01)
        except Exception as e:
            logging.warning(f"[!] Error in classify_people: {e}")


def extract_detections(hailo_output: List[np.ndarray], h: int, w: int, threshold: float = 0.5, person_class_id: int = 0) -> Dict[str, np.ndarray]:
    xyxy, confidence, class_id = [], [], []
    for i, detections in enumerate(hailo_output):
        for detection in detections:
            bbox, score = detection[:4], detection[4]
            if score < threshold or i != person_class_id:
                continue
            bbox = np.clip(bbox, 0, 1)
            bbox[0], bbox[1], bbox[2], bbox[3] = (
                bbox[1] * w,
                bbox[0] * h,
                bbox[3] * w,
                bbox[2] * h,
            )
            xyxy.append(bbox)
            confidence.append(score)
            class_id.append(i)

    return {
        "xyxy": np.array(xyxy),
        "confidence": np.array(confidence),
        "class_id": np.array(class_id),
        "num_detections": len(xyxy),
    }

def frame_reader(cap: cv2.VideoCapture, frame_queue: Queue):
    """Thread function to read frames from the RTSP stream every 2 seconds."""
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of RTSP stream or error reading frame.")
            break

        frame_queue.put(frame)

        if frame_queue.qsize() > 10:
            frame_queue.get()  # 控制内存

    cap.release()

def process_and_annotate_frames(
    frame_queue: Queue,
    input_queue: Queue,
    output_queue: Queue,
    information_queue: Queue,
    model_h: int,
    model_w: int,
    video_h: int,
    video_w: int,
    config: Config,
):
    """Process function to process and annotate frames."""
    tracker = sv.ByteTrack()
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            preprocessed_frame = preprocess_frame(frame, model_h, model_w, video_h, video_w)

            # Put frame for inference
            input_queue.put([preprocessed_frame])

            # Get output from inference
            _, results = output_queue.get()

            # Deal with older hailort version outputs
            if len(results) == 1:
                results = results[0]

            detections = extract_detections(results, video_h, video_w)

            # If no detections, continue
            if detections["xyxy"].shape[0] == 0:
                continue

            # Create Detections object
            sv_dets = sv.Detections(
                xyxy=detections["xyxy"],
                confidence=detections["confidence"],
                class_id=detections["class_id"],
            )

            # Update tracker with detections
            tracked_dets = tracker.update_with_detections(sv_dets)

            # Prepare JSON data with tracker_id and bbox per tracked detection
            tracking_data = []
            for tid, bbox in zip(tracked_dets.tracker_id, tracked_dets.xyxy):
                tracking_data.append({
                    "tracker_id": int(tid),
                    "bbox": bbox.tolist()
                })
                # print(f"Tracker ID: {tid}, BBox: {bbox.tolist()}")
            

            # Put tracking data into the information queue
            information_queue.put(json.dumps(tracking_data))

# def main():
#     config = load_config()
    
#     frame_queue = Queue(maxsize=1000)
#     input_queue = Queue()
#     output_queue = Queue()
#     information_queue = Queue()
#     message_queue = Queue()


#     hailo_inference = HailoAsyncInference(
#         hef_path=config.net,
#         input_queue=input_queue,
#         output_queue=output_queue,
#     )
#     model_h, model_w, _ = hailo_inference.get_input_shape()
    

#     if config.input_video.startswith("rtsp://"):
#         cap = cv2.VideoCapture(config.input_video, cv2.CAP_FFMPEG)
#         if not cap.isOpened():
#             print("Error: Unable to open RTSP stream.")
#             return
#         video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     else:
#         print("Only RTSP streams are supported in this implementation.")
#         return

#     with open(config.labels, "r", encoding="utf-8") as f:
#         class_names = f.read().splitlines()

#     inference_thread = threading.Thread(target=hailo_inference.run)
#     inference_thread.start()
    
#     frame_reader_thread = threading.Thread(target=frame_reader, args=(cap, frame_queue))
#     frame_reader_thread.start()
    
#     process_and_annotate_frames_process = Process(
#         target=process_and_annotate_frames,
#         args=(frame_queue, input_queue, output_queue, information_queue,
#               model_h, model_w, video_h, video_w, config)
#     )
#     process_and_annotate_frames_process.start()
    
    
#     classify_thread = threading.Thread(
#         target=classify_people,
#         args=(config.point_of_line, information_queue, message_queue)
#     )
#     classify_thread.start()
    
#     seed_message_thread = threading.Thread(
#         target=send_message_to_node,
#         args=(config.send_node, message_queue)  # Ensure message_queue is passed here
#     )
#     seed_message_thread.start()
    
#     frame_reader_thread.join()
#     inference_thread.join()
#     process_and_annotate_frames_process.join()
#     classify_thread.join()
#     send_message_thread.join()

def main():
    config = load_config()
    
    frame_queue = Queue()
    input_queue = Queue()
    output_queue = Queue()
    information_queue = Queue()
    message_queue = Queue()

    hailo_inference = HailoAsyncInference(
        hef_path=config.net,
        input_queue=input_queue,
        output_queue=output_queue,
    )
    model_h, model_w, _ = hailo_inference.get_input_shape()
    interface = meshtastic.serial_interface.SerialInterface()

    if config.input_video.startswith("rtsp://"):
        cap = cv2.VideoCapture(config.input_video, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print("Error: Unable to open RTSP stream.")
            return
        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        print("Only RTSP streams are supported in this implementation.")
        return

    # Load labels
    with open(config.labels, "r", encoding="utf-8") as f:
        class_names = f.read().splitlines()

    # === Threads for I/O-bound tasks ===
    threading.Thread(target=hailo_inference.run, daemon=True).start()
    threading.Thread(target=frame_reader, args=(cap, frame_queue), daemon=True).start()
    # threading.Thread(target=send_message_to_node, args=(config.send_node, message_queue), daemon=True).start()

    # === Process for CPU-bound frame processing ===
    process_and_annotate_frames_process = Process(
        target=process_and_annotate_frames,
        args=(frame_queue, input_queue, output_queue, information_queue,
              model_h, model_w, video_h, video_w, config),
        daemon=True
    )
    process_and_annotate_frames_process.start()

    # === Process for CPU-bound classification ===
    classify_people_process = Process(
        target=classify_people,
        args=(config.point_of_line, information_queue, message_queue,interface, config),
        daemon=True
    )
    classify_people_process.start()

    # === Keep main thread alive ===
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")


if __name__ == "__main__":
    main()
