from ultralytics import YOLO
import cv2
import os

# 모델 로드
model = YOLO("yolov8n.pt")

# 경로 설정
base_dir = os.path.dirname(__file__)
video_path = os.path.join(base_dir, "../data/video.mp4")
output_path = os.path.join(base_dir, "../results/output.mp4")

# 영상 불러오기
cap = cv2.VideoCapture(video_path)

# 영상 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

# 위험 기준
DANGER_THRESHOLD = 15000

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    class_names = model.names

    danger_detected = False

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = class_names[cls_id]

        if class_name == "person":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            area = (x2 - x1) * (y2 - y1)

            # 위험 판단
            if area > DANGER_THRESHOLD:
                color = (0, 0, 255)
                label = f"DANGER {conf:.2f}"
                danger_detected = True
                thickness = 4
            else:
                color = (0, 255, 0)
                label = f"Person {conf:.2f}"
                thickness = 2

            # 박스
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # 텍스트
            cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 🚨 전체 경고 표시
    if danger_detected:
        cv2.putText(frame, "⚠️ WARNING: PERSON TOO CLOSE!",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3)

    # 화면 출력
    cv2.imshow("YOLO Detection", frame)

    # 영상 저장
    out.write(frame)

    # 종료 키
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"영상 저장 완료: {output_path}")