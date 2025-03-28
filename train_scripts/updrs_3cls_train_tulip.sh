exp_dir=train_output/hospital_updrs

mkdir -p "${exp_dir}"
python3.8 -u -m torch.distributed.run --nproc_per_node 1 \
  ./training/train.py \
    --nfold 10 \
    --type updrs \
    --data_root tulip \
    --text_prompt_classes_path "./classes/updrs_3cls_classes.txt" \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --num_steps 800 \
    --save_freq 2001 \
    --eval_freq 1 \
    --batch_size 4 \
    --backbone_path "./pretrained/clip_pretrained.pth" \
    --mean 0.48145466 0.4578275 0.40821073 \
    --std 0.26862954 0.26130258 0.27577711 \
    --lr "1e-4" \
    --num_workers 6 \
    --num_frames 70 \
    --no_mirror \
    --spatial_size 224 \
    --use_text_prompt_learning \
    --text_num_prompts 8 \
    --use_text_prompt_CSC \
    --use_summary_token \
    --use_local_prompts \
    --use_global_prompts \
    --num_global_prompts 8 \
    --text_prompt_init "cntn_split_uni_disc" \
    --knowledge_version "v1" \
    --knowledge_version "v2" \
    --knowledge_version "v3" \
    --knowledge_version "v4" \
    --knowledge_version "v5" \
    --use_support_memory \
    --memory_data_path "./data/gait/tulip_dict_basic_4f.pkl" \
    --use_focal_ordinal_loss \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"
