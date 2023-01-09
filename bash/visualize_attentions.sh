# KINETICS 2 NEC-DRONE

# source
CUDA_VISIBLE_DEVICES=1 python3 ../visualize_attention.py \
    --pretrained_model bash/pretrained_source_models/kinetics-nec-source-head+partial-191p572e-ep=19.ckpt \
    --dataset /data/datasets/nec-drone/cropped/test \
    --n_frames 16 \
    --frame_size 224 \
    --batch_size 1 \
    --n_clips 1 \
    --num_workers 8 \
    --replace_with_mlp \
    --mlp_n_layers 0 \
    --out_folder nec_attention_visualization_source_only

# da
CUDA_VISIBLE_DEVICES=0 python3 ../visualize_attention.py \
    --pretrained_model bash/trained_models/2th5tb5j/kinetics-nec-head+temporal-barlow-unsup-2th5tb5j-ep=19.ckpt \
    --dataset /data/datasets/nec-drone/cropped/test \
    --n_frames 16 \
    --frame_size 224 \
    --batch_size 1 \
    --n_clips 1 \
    --num_workers 8 \
    --replace_with_mlp \
    --mlp_n_layers 0 \
    --out_folder nec_attention_visualization_2th5tb5j
# -----------------------------------

# HMDB 2 UCF

# source
CUDA_VISIBLE_DEVICES=0 python3 ../visualize_attention.py \
    --pretrained_model bash/pretrained_source_models/hmdb-ucf-source-head+partial-35xc7vyr-ep=19.ckpt \
    --dataset /data/datasets/hmdb_ucf/ucf/test \
    --n_frames 16 \
    --frame_size 224 \
    --batch_size 1 \
    --n_clips 1 \
    --num_workers 8 \
    --replace_with_mlp \
    --mlp_n_layers 0 \
    --out_folder hmdb_ucf_visualization_source_only

CUDA_VISIBLE_DEVICES=1 python3 ../visualize_attention.py \
    --pretrained_model bash/trained_models/2vn14tbb/hmdb-ucf-head+temporal-barlow-unsup-2vn14tbb-ep=19.ckpt \
    --dataset /data/datasets/hmdb_ucf/ucf/test \
    --n_frames 16 \
    --frame_size 224 \
    --batch_size 1 \
    --n_clips 1 \
    --num_workers 8 \
    --replace_with_mlp \
    --mlp_n_layers 0 \
    --out_folder hmdb_ucf_attention_visualization_2vn14tbb

# -------------------------------------

# UCF 2 HMDB

CUDA_VISIBLE_DEVICES=0 python3 ../visualize_attention.py \
    --pretrained_model bash/pretrained_source_models/ucf-hmdb-source-head+partial-343fe6qo-ep=19.ckpt \
    --dataset /data/datasets/hmdb_ucf/hmdb/test \
    --n_frames 16 \
    --frame_size 224 \
    --batch_size 1 \
    --n_clips 1 \
    --num_workers 8 \
    --replace_with_mlp \
    --mlp_n_layers 0 \
    --out_folder ucf_hmdb_visualization_source_only

# da
CUDA_VISIBLE_DEVICES=1 python3 ../visualize_attention.py \
    --pretrained_model bash/trained_models/1sovt50y/ucf-hmdb-head+temporal-barlow-unsup-1sovt50y-ep=19.ckpt \
    --dataset /data/datasets/hmdb_ucf/hmdb/test \
    --n_frames 16 \
    --frame_size 224 \
    --batch_size 1 \
    --n_clips 1 \
    --num_workers 8 \
    --replace_with_mlp \
    --mlp_n_layers 0 \
    --out_folder ucf_hmdb_visualization_1sovt50y
