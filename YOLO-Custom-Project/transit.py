import os

label_dir = "dataset/labels"

mapping = {
    "person_phone": 0,
    "person_normal": 1
}

for file in os.listdir(label_dir):
    if file.endswith(".txt"):
        path = os.path.join(label_dir, file)

        new_lines = []
        with open(path, "r") as f:
            for line in f:
                parts = line.split()
                if parts[0] in mapping:
                    parts[0] = str(mapping[parts[0]])
                new_lines.append(" ".join(parts))

        with open(path, "w") as f:
            f.write("\n".join(new_lines))

print("라벨 변환 완료")