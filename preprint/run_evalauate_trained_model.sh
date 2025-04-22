# Face Classification, resnet18 imagenet1kv1 finetuned
export rn18_imgnet_finetuned_face_classification="/om2/user/alishd/github/comparing-category-selectivity/preprint/trained_models/compcats_face_classification/finetune/resnet18/f5b95b1f-31fd-47c4-a6fb-2e52f709349a"
# Face Classification, resnet18 scratch
export rn18_scratch_face_classification="/om2/user/alishd/github/comparing-category-selectivity/preprint/trained_models/compcats_face_classification/scratch/resnet18/f2c52f72-351f-4a13-9865-209d6edacf56"

python evaluate_trained_model.py \
--data.model_dir=${rn18_scratch_face_classification} \
--data.nsd_dir=/weka/scratch/weka/sinha/alishd/datasets/nsd \
--data.localizer_dir=/net/vast-storage/scratch/vast/sinha/alishd/datasets/localizer_grillspector/fLoc_stimuli \
--data.output_dir=/om2/user/alishd/github/comparing-category-selectivity/preprint/trained_models \
