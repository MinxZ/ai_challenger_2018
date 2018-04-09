python object_detection/train.py \
        --logtostderr \
        --pipeline_config_path=/data/zl/animals/models/ssd_mobilenet_v1/ssd_mobilenet_v1_focal_loss_coco.config \
        --train_dir=/data/zl/animals/models/ssd_mobilenet_v1/train

python object_detection/eval.py \
        --logtostderr \
        --pipeline_config_path=/data/zl/animals/models/ssd_mobilenet_v1/ssd_mobilenet_v1_focal_loss_coco.config \
        --checkpoint_dir=/data/zl/animals/models/ssd_mobilenet_v1/train \
        --eval_dir=/data/zl/animals/models/ssd_mobilenet_v1/eval
