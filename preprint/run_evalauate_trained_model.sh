# Baseline, Imagenet1k, ResNet18, seed0, gseed42, mseed42
export rn18_imgnet1k_baseline_seed1="/om2/user/alishd/github/comparing-category-selectivity/preprint/trained_models/compcats_baseline/imagenet1k/resnet18/07b23b77-d466-4742-b831-1298411e7288"
# Baseline, Imagenet1k, ResNet18, seed1, gseed12, mseed54
export rn18_imgnet1k_baseline_seed2="/om2/user/alishd/github/comparing-category-selectivity/preprint/trained_models/compcats_baseline/imagenet1k/resnet18/3f108f6e-5475-49b0-9879-d1865e2110dc"
# Baseline, Imagenet1k, ResNet18, seed2, gseed23, mseed35
export rn18_imgnet1k_baseline_seed3="/om2/user/alishd/github/comparing-category-selectivity/preprint/trained_models/compcats_baseline/imagenet1k/resnet18/6bb52e1e-6ceb-457a-a622-d92572c430b5"
# Baseline, Imagenet1k, ResNet18, seed3, gseed2, mseed84
export rn18_imgnet1k_baseline_seed4="/om2/user/alishd/github/comparing-category-selectivity/preprint/trained_models/compcats_baseline/imagenet1k/resnet18/7973cdaa-988f-45f8-9cdf-54b38845004e"
# Baseline, Imagenet1k, ResNet18, seed4, gseed63, mseed92
export rn18_imgnet1k_baseline_seed5="/om2/user/alishd/github/comparing-category-selectivity/preprint/trained_models/compcats_baseline/imagenet1k/resnet18/45f6d793-4f37-4312-aa26-9fc6d1bb99a8"

# Face Classification, resnet18 imagenet1kv1 finetuned
export rn18_imgnet_finetuned_face_classification="/om2/user/alishd/github/comparing-category-selectivity/preprint/trained_models/compcats_face_classification/finetune/resnet18/f5b95b1f-31fd-47c4-a6fb-2e52f709349a"
# Face Classification, resnet18 scratch
export rn18_scratch_face_classification="/om2/user/alishd/github/comparing-category-selectivity/preprint/trained_models/compcats_face_classification/scratch/resnet18/f2c52f72-351f-4a13-9865-209d6edacf56"

# Face Contrastive, resnet18 imagenet1kv1 finetuned, lw0.5 tmp0.1, res 160 -> 192
export rn18_imgnet_finetuned_face_contrastive_1="/om2/user/alishd/github/comparing-category-selectivity/preprint/trained_models/compcats_face_contrastive/hopt5/resnet18/1cead616-f96e-4dbf-a4c6-4ce58bf625aa"
# Face Contrastive, resnet18 scratch, lw0.1 tmp0.1, res 224, epoch 46, cont loss = 0.871
export rn18_imgnet_finetuned_face_contrastive_2="/om2/user/alishd/github/comparing-category-selectivity/preprint/trained_models/compcats_face_contrastive/hopt5/resnet18/1116e0c4-cd57-4cb4-a551-f39b5dd02cf9"
# Face Contrastive, finetuned, lw0.5 tmp0.1, res 224, epoch 78, val cont loss = 0.649
export rn18_imgnet_finetuned_face_contrastive_3="/om2/user/alishd/github/comparing-category-selectivity/preprint/trained_models/compcats_face_contrastive/hopt6/resnet18/6a3dab82-855f-432c-803c-19fc068be407"
# Face Contrastive, finetuned, lw0.5 tmp0.1, res 160-192, epoch 144/150, val cont loss = 1.19, val top 1 = 56
export rn18_imgnet_finetuned_face_contrastive_4="/om2/user/alishd/github/comparing-category-selectivity/preprint/trained_models/compcats_face_contrastive/hopt6/resnet18/d4e566aa-ce09-493f-92dc-cbe4c52a48ea"
# Face Contrastive, scratch, lw0.5 tmp0.1, res 224, epoch 125/150, val cont loss = 1.18, val top 1 = 45
export rn18_imgnet_finetuned_face_contrastive_5="/om2/user/alishd/github/comparing-category-selectivity/preprint/trained_models/compcats_face_contrastive/hopt6/resnet18/ac2777cb-b60d-4b1a-ba79-6b9bd95ac644"
# Face Contrastive, scratch, lw0.25 tmp0.1, lr0.5, res 224, tmp epoch 113, val cont loss = 0.64, val top 1 = 56
export rn18_imgnet_finetuned_face_contrastive_6_tmp="/om2/user/alishd/github/comparing-category-selectivity/preprint/trained_models/compcats_face_contrastive/hopt7/resnet18/f8efb51a-0035-41af-87d1-d173782eb47f"

# hopt7
# Face Contrastive, finetuned, lw0.5 tmp0.1, lr1, res 224, val cont loss = 0.9, val top 1 = 53.6
export hopt7_lw0pt5_lr1="/om2/user/alishd/github/comparing-category-selectivity/preprint/trained_models/compcats_face_contrastive/hopt7/resnet18/e4af8785-636b-4db5-9de1-b552661ab3ff"
# Face Contrastive, finetuned, lw0.25 tmp0.1, lr0.5, res 224, val cont loss = 2.3, val top 1 = 58
export hopt7_lw0pt25_lr0pt5="/om2/user/alishd/github/comparing-category-selectivity/preprint/trained_models/compcats_face_contrastive/hopt7/resnet18/f8efb51a-0035-41af-87d1-d173782eb47f"
# Face Contrastive, finetuned, lw0.25 tmp0.1, lr1, res 224, val cont loss = 1.04, val top 1 = 53.2
export hopt7_lw0pt25_lr1="/om2/user/alishd/github/comparing-category-selectivity/preprint/trained_models/compcats_face_contrastive/hopt7/resnet18/c7c66073-539e-458e-8350-2dab69c7f1bb"

python evaluate_trained_model.py \
--data.model_dir=${rn18_imgnet_finetuned_face_contrastive_3} \
--data.nsd_dir=/weka/scratch/weka/sinha/alishd/datasets/nsd \
--data.localizer_dir=/net/vast-storage/scratch/vast/sinha/alishd/datasets/localizer_grillspector/fLoc_stimuli \
--data.output_dir=/om2/user/alishd/github/comparing-category-selectivity/preprint/trained_models \
# --model.epoch=113 \