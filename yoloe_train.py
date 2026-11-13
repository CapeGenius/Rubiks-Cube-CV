from ultralytics.data.split import autosplit
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPETrainer

# split dataset into train and val datasets
# autosplit(path="./dataset/images", weights=(0.8, 0.2, 0.0), annotated_only=True)


model = YOLOE("yoloe-26m.yaml").load("yoloe-26m-seg.pt")

results = model.train(
    data="datasets.yaml",
    epochs=10,
    patience=10,
    trainer=YOLOEPETrainer,  # <- Important: the fine-tuning trainer, not the default
)
