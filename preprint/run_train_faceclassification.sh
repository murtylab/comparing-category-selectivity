python train_faceclassification.py \
--config-file=configs/rn18_dev.yaml \
--data.celebahq_train_data_path=/om/user/alishd/datasets/celebahq/ffcv/train_500_0.50_90.ffcv \
--data.celebahq_val_data_path=/om/user/alishd/datasets/celebahq/ffcv/val_500_0.50_90.ffcv \
--data.afhqv2_train_data_path=/om/user/alishd/datasets/afhqv2/ffcv/train_500_0.50_90.ffcv \
--data.afhqv2_val_data_path=/om/user/alishd/datasets/afhqv2/ffcv/val_500_0.50_90.ffcv \
--data.lhq_train_data_path=/om/user/alishd/datasets/lhq/ffcv/train_500_0.50_90.ffcv \
--data.lhq_val_data_path=/om/user/alishd/datasets/lhq/ffcv/val_500_0.50_90.ffcv \
--data.output_dir=/om2/user/alishd/github/comparing-category-selectivity/preprint/trained_models \
--model.pretrained=1 \
--model.pretrained_model=resnet18_imagenet1kv1 \
--seed.global_seed=42 \
--seed.model_seed=42 \
--project.name=compcats_face_classification \
--project.experiment=finetune \
# --training.epochs=10 \
# --resolution.resolution_schedule_type=fixed \
# --resolution.fixed_resolution=224 \
