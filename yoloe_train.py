from ultralytics.data.split import autosplit
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPETrainer, YOLOEPESegTrainer
from ultralytics.data.converter import yolo_bbox2segment

# split dataset into train and val datasets
# autosplit(path="./dataset/images", weights=(0.8, 0.2, 0.0), annotated_only=True)


# yolo_bbox2segment(im_dir="./dataset/images", save_dir=None, sam_model="sam_b.pt")

model = YOLOE("yoloe-26s-seg.pt")

results = model.train(
    data="datasets.yaml",
    epochs=10,
    patience=10,
    trainer=YOLOEPESegTrainer,  # <- Important: the fine-tuning trainer, not the default
)
