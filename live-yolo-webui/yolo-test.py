from pathlib import Path

import cv2
from ultralytics import YOLOWorld

model = YOLOWorld("yolov8s-worldv2.pt")
model.set_classes([
    "chair", "table", "whiteboard",
    "door", "elevator door", "sign", "stairs"
])

result = model.predict(
    source="/home/hp17/Pictures/IMG_1090.jpeg",
    device=0,
    imgsz=640,
    conf=0.25,
    verbose=True
)[0]

image = result.orig_img.copy()
height, width = image.shape[:2]

for box in result.boxes:
    label = result.names[int(box.cls.item())]
    confidence = box.conf.item()
    x1, y1, x2, y2 = box.xyxy[0].tolist()

    center_x = ((x1 + x2) / 2) / width
    center_y = ((y1 + y2) / 2) / height

    horizontal = (
        "left" if center_x < 1 / 3
        else "right" if center_x > 2 / 3
        else "center"
    )
    vertical = (
        "top" if center_y < 1 / 3
        else "bottom" if center_y > 2 / 3
        else "middle"
    )

    text = f"{label} {confidence:.2f} | {vertical}-{horizontal}"
    print(text)

    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    color = (0, 220, 0)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # Keep the label background inside the image.
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 3.0
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, scale, thickness
    )
    left = max(0, min(x1, width - text_width - 8))
    top = max(0, y1 - text_height - baseline - 10)

    cv2.rectangle(
        image,
        (left, top),
        (left + text_width + 8, top + text_height + baseline + 8),
        color,
        -1
    )
    cv2.putText(
        image,
        text,
        (left + 4, top + text_height + 4),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )

target_name = "whiteboard"
center_tolerance = 0.05  # Center band: 45%–55% of image width

# Find detections matching the target.
matches = [
    box for box in result.boxes
    if result.names[int(box.cls.item())] == target_name
]

if not matches:
    print(f"\n{target_name.capitalize()} not detected.")
else:
    # If multiple match, select the highest-confidence detection.
    target = max(matches, key=lambda box: box.conf.item())
    x1, _, x2, _ = target.xyxy[0].tolist()

    target_center = ((x1 + x2) / 2) / width
    offset = target_center - 0.5

    if offset < -center_tolerance:
        instruction = "Pan the camera left."
    elif offset > center_tolerance:
        instruction = "Pan the camera right."
    else:
        instruction = "Target centered. Keep the camera facing this direction."

    print(f"\nTarget: {target_name}")
    print(f"Camera instruction: {instruction}")

# Draw a vertical center line on the annotated image.
center_x = width // 2

cv2.line(
    image,
    (center_x, 0),
    (center_x, height - 1),
    (0, 255, 255),  # Yellow in OpenCV's BGR format
    3
)

cv2.putText(
    image,
    "CENTER",
    (max(0, center_x - 65), 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (0, 255, 255),
    2,
    cv2.LINE_AA
)

output = Path(__file__).resolve().parent / "yolo_test_result.jpg"
if not cv2.imwrite(str(output), image):
    raise RuntimeError(f"Could not save {output}")

print("Timing in milliseconds:", result.speed)
print(f"Saved: {output}")