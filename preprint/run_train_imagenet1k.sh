python train_imagenet1k.py \
--config-file=configs/rn18_88_epochs.yaml \
--data.train_data_path=/om/user/alishd/datasets/imagenet/ffcv/train_256_0.5_90.ffcv \
--data.val_data_path=/om/user/alishd/datasets/imagenet/ffcv/val_256_0.5_90.ffcv \
--data.output_dir=/om2/user/alishd/github/comparing-category-selectivity/preprint/trained_models \
--seed.global_seed=42 \
--seed.model_seed=42 \
--project.name=compcats_baseline \
--project.experiment=imagenet1k \