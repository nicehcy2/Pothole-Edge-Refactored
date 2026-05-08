from ultralytics import YOLO

model = YOLO("bestv8m.pt")
model.export(format="engine", half=True, device=0)
print("변환 완료: bestv8m.engine")
