import os

label_dir = r"C:\Users\user\Desktop\Autonomous Vehicle Project\YOLO-Custom-Project\dataset\labels"

count = {0: 0, 1: 0}

for file in os.listdir(label_dir):
    if file.endswith(".txt"):
        with open(os.path.join(label_dir, file), "r") as f:
            for line in f:
                cls = int(line.split()[0])
                count[cls] += 1

print("=== 클래스 개수 ===")
print(f"person_phone (0): {count[0]}")
print(f"person_normal (1): {count[1]}")