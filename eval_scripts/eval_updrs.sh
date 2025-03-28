for i in ./logs/updrs_mix*; do echo $i;python3.8 -u -m torch.distributed.run --nproc_per_node 1 \
  ./evaluation/evaluate.py \
    --type updrs \
    --backbone_path "./pretrained/clip_pretrained.pth" \
    --checkpoint_dir $i \
    --text_prompt_classes_path "./classes/updrs_classes_3.txt" \
    --data_root "datasets/real_3cls/test" \
    --val_list_path "./datasets/real_3cls/test/test_updrs.csv" \
    --mean 0.48145466 0.4578275 0.40821073 \
    --std 0.26862954 0.26130258 0.27577711 \
    --num_workers 6 \
    --no_mirror \
    --num_frames 70 \
    --batch_size 1 \
    --sampling_rate 1 \
    --spatial_size 224 \
    --use_text_prompt_learning \
    --text_num_prompts 8 \
    --use_text_prompt_CSC \
    --use_summary_token \
    --use_local_prompts \
    --use_global_prompts \
    --num_global_prompts 8
done