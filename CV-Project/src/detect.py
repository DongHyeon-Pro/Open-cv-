from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8n.pt")

base_dir = os.path.dirname(__file__)
image_path = os.path.join(base_dir, "../data/test.jpg")
output_path = os.path.join(base_dir, "../results/result_warning.jpg")

results = model(image_path)
image = cv2.imread(image_path)

class_names = model.names

# 위험 기준 (박스 크기)
DANGER_THRESHOLD = 15000  # 튜닝 가능

for box in results[0].boxes:
    cls_id = int(box.cls[0])
    class_name = class_names[cls_id]

    if class_name == "person":
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        # 박스 크기 계산
        area = (x2 - x1) * (y2 - y1)

        print(f"person | conf={conf:.2f} | area={area}")

        # 위험 판단
        if area > DANGER_THRESHOLD:
            color = (0, 0, 255)  # 빨강
            label = f"DANGER {conf:.2f}"
            danger_detected = False
        else:
            color = (0, 255, 0)  # 초록
            label = f"Person {conf:.2f}"

        # 박스
        thickness = 4 if area > DANGER_THRESHOLD else 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # 텍스트
        cv2.putText(image, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)

if danger_detected:
    cv2.putText(image, "⚠️ WARNING: PERSON TOO CLOSE!",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3)
if area > 20000:
    label = "HIGH RISK"
elif area > 10000:
    label = "MEDIUM"
else:
    label = "SAFE"

# 결과 저장
cv2.imwrite(output_path, image)

print(f"위험 감지 완료: {output_path}")