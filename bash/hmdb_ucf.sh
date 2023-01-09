# ! hmdb-ucf

# * source only
python3 ../main.py \
    --train_source_dataset /data/datasets/hmdb_ucf/hmdb/train \
    --val_dataset /data/datasets/hmdb_ucf/ucf/test \
    --epochs 20 \
    --optimizer sgd \
    --lr 0.001 \
    --weight_decay 1e-9 \
    --scheduler cosine \
    --batch_size 4 \
    --n_clips 1 \
    --n_frames 16 \
    --frame_size 224 \
    --num_workers 5 \
    --gpus 0 1 \
    --train head+partial \
    --mlp_hidden_dim 2048 \
    --mlp_n_layers 0 \
    --replace_with_mlp \
    --name hmdb-ucf-source-head+partial \
    --project transformer_da \
    --save_model \
    --wandb

# * ib sup - no queue
python3 ../main.py \
    --train_source_dataset /data/datasets/hmdb_ucf/hmdb/train \
    --train_target_dataset /data/datasets/hmdb_ucf/ucf/train \
    --val_dataset /data/datasets/hmdb_ucf/ucf/test \
    --epochs 20 \
    --optimizer sgd \
    --lr 0.005 \
    --weight_decay 1e-9 \
    --scheduler cosine \
    --batch_size 32 \
    --n_clips 1 \
    --n_frames 16 \
    --frame_size 224 \
    --num_workers 4 \
    --gpus 0 1 \
    --train head+temporal \
    --mlp_hidden_dim 2048 \
    --mlp_n_layers 0 \
    --replace_with_mlp \
    --name hmdb-ucf-head+temporal-ib-sup-no-queue \
    --project bmvc-2021 \
    --da ib \
    --ib_loss_weight 0.01 \
    --wandb \
    --pretrained_source_model pretrained_source_models/hmdb-ucf-source-head+partial-35xc7vyr-ep=19.ckpt

# * ib unsup - no queue
python3 ../main.py \
    --train_source_dataset /data/datasets/hmdb_ucf/hmdb/train \
    --train_target_dataset /data/datasets/hmdb_ucf/ucf/train \
    --val_dataset /data/datasets/hmdb_ucf/ucf/test \
    --epochs 20 \
    --optimizer sgd \
    --lr 0.005 \
    --weight_decay 1e-9 \
    --scheduler cosine \
    --batch_size 32 \
    --n_clips 1 \
    --n_frames 16 \
    --frame_size 224 \
    --num_workers 4 \
    --gpus 0 1 \
    --train head+temporal \
    --mlp_hidden_dim 2048 \
    --mlp_n_layers 0 \
    --replace_with_mlp \
    --name hmdb-ucf-head+temporal-ib-unsup-no-queue \
    --project bmvc-2021 \
    --da ib \
    --pseudo_labels \
    --ib_loss_weight 0.01 \
    --wandb \
    --pretrained_source_model pretrained_source_models/hmdb-ucf-source-head+partial-35xc7vyr-ep=19.ckpt

# * ib sup
python3 ../main.py \
    --train_source_dataset /data/datasets/hmdb_ucf/hmdb/train \
    --train_target_dataset /data/datasets/hmdb_ucf/ucf/train \
    --val_dataset /data/datasets/hmdb_ucf/ucf/test \
    --epochs 20 \
    --optimizer sgd \
    --lr 0.005 \
    --weight_decay 1e-9 \
    --scheduler cosine \
    --batch_size 32 \
    --n_clips 1 \
    --n_frames 16 \
    --frame_size 224 \
    --num_workers 4 \
    --gpus 0 1 \
    --train head+temporal \
    --mlp_hidden_dim 2048 \
    --mlp_n_layers 0 \
    --replace_with_mlp \
    --name hmdb-ucf-head+temporal-ib-sup \
    --project bmvc-2021 \
    --da ib \
    --use_queue \
    --queue_size 1024 \
    --ib_loss_weight 0.01 \
    --wandb \
    --pretrained_source_model pretrained_source_models/hmdb-ucf-source-head+partial-35xc7vyr-ep=19.ckpt

# * ib unsup
python3 ../main.py \
    --train_source_dataset /data/datasets/hmdb_ucf/hmdb/train \
    --train_target_dataset /data/datasets/hmdb_ucf/ucf/train \
    --val_dataset /data/datasets/hmdb_ucf/ucf/test \
    --epochs 20 \
    --optimizer sgd \
    --lr 0.005 \
    --weight_decay 1e-9 \
    --scheduler cosine \
    --batch_size 32 \
    --n_clips 1 \
    --n_frames 16 \
    --frame_size 224 \
    --num_workers 4 \
    --gpus 0 1 \
    --train head+temporal \
    --mlp_hidden_dim 2048 \
    --mlp_n_layers 0 \
    --replace_with_mlp \
    --name hmdb-ucf-head+temporal-ib-unsup \
    --project bmvc-2021 \
    --da ib \
    --pseudo_labels \
    --use_queue \
    --queue_size 1024 \
    --ib_loss_weight 0.01 \
    --wandb \
    --pretrained_source_model pretrained_source_models/hmdb-ucf-source-head+partial-35xc7vyr-ep=19.ckpt

# # #* simclr supervised
# # python3 ../main.py \
# #     --train_source_dataset /data/datasets/hmdb_ucf/hmdb/train \
# #     --train_target_dataset /data/datasets/hmdb_ucf/ucf/train \
# #     --val_dataset /data/datasets/hmdb_ucf/ucf/test \
# #     --epochs 20 \
# #     --optimizer sgd \
# #     --lr 0.005 \
# #     --weight_decay 1e-9 \
# #     --scheduler cosine \
# #     --batch_size 32 \
# #     --n_clips 1 \
# #     --n_frames 16 \
# #     --frame_size 224 \
# #     --num_workers 5 \
# #     --gpus 0 1 \
# #     --train head+temporal \
# #     --mlp_hidden_dim 2048 \
# #     --mlp_n_layers 0 \
# #     --replace_with_mlp \
# #     --name hmdb-ucf-head+temporal-simclr-sup \
# #     --project transformer_da \
# #     --da simclr \
# #     --simclr_loss_weight 5 \
# #     --save_model \
# #     --wandb \
# #     --pretrained_source_model pretrained_source_models/hmdb-ucf-source-head+partial-35xc7vyr-ep=19.ckpt

# # # * simclr unsup
# # python3 ../main.py \
# #     --train_source_dataset /data/datasets/hmdb_ucf/hmdb/train \
# #     --train_target_dataset /data/datasets/hmdb_ucf/ucf/train \
# #     --val_dataset /data/datasets/hmdb_ucf/ucf/test \
# #     --epochs 20 \
# #     --optimizer sgd \
# #     --lr 0.005 \
# #     --weight_decay 1e-9 \
# #     --scheduler cosine \
# #     --batch_size 32 \
# #     --n_clips 1 \
# #     --n_frames 16 \
# #     --frame_size 224 \
# #     --num_workers 5 \
# #     --gpus 0 1 \
# #     --train head+temporal \
# #     --mlp_hidden_dim 2048 \
# #     --mlp_n_layers 0 \
# #     --replace_with_mlp \
# #     --name hmdb-ucf-head+temporal-simclr-unsup \
# #     --project transformer_da \
# #     --da simclr \
# #     --simclr_loss_weight 1.0 \
# #     --pseudo_labels \
# #     --save_model \
# #     --wandb \
# #     --pretrained_source_model pretrained_source_models/hmdb-ucf-source-head+partial-35xc7vyr-ep=19.ckpt

# # #* vicreg supervised
# # python3 ../main.py \
# #     --train_source_dataset /data/datasets/hmdb_ucf/hmdb/train \
# #     --train_target_dataset /data/datasets/hmdb_ucf/ucf/train \
# #     --val_dataset /data/datasets/hmdb_ucf/ucf/test \
# #     --epochs 20 \
# #     --optimizer sgd \
# #     --lr 0.005 \
# #     --weight_decay 1e-9 \
# #     --scheduler cosine \
# #     --batch_size 32 \
# #     --n_clips 1 \
# #     --n_frames 16 \
# #     --frame_size 224 \
# #     --num_workers 5 \
# #     --gpus 0 1 \
# #     --train head+temporal \
# #     --mlp_hidden_dim 2048 \
# #     --mlp_n_layers 0 \
# #     --replace_with_mlp \
# #     --name hmdb-ucf-head+temporal-vicreg-sup \
# #     --project transformer_da \
# #     --da vicreg \
# #     --vicreg_loss_weight 5 \
# #     --save_model \
# #     --wandb \
# #     --pretrained_source_model pretrained_source_models/hmdb-ucf-source-head+partial-35xc7vyr-ep=19.ckpt

# # * vicreg unsup
# python3 ../main.py \
#     --train_source_dataset /data/datasets/hmdb_ucf/hmdb/train \
#     --train_target_dataset /data/datasets/hmdb_ucf/ucf/train \
#     --val_dataset /data/datasets/hmdb_ucf/ucf/test \
#     --epochs 20 \
#     --optimizer sgd \
#     --lr 0.005 \
#     --weight_decay 1e-9 \
#     --scheduler cosine \
#     --batch_size 32 \
#     --n_clips 1 \
#     --n_frames 16 \
#     --frame_size 224 \
#     --num_workers 5 \
#     --gpus 0 1 \
#     --train head+temporal \
#     --mlp_hidden_dim 2048 \
#     --mlp_n_layers 0 \
#     --replace_with_mlp \
#     --name hmdb-ucf-head+temporal-vicreg-unsup \
#     --project transformer_da \
#     --da vicreg \
#     --target_ce_selection_factor 1 \
#     --vicreg_loss_weight 2.5 \
#     --sim_loss_weight 5 \
#     --var_loss_weight 5 \
#     --cov_loss_weight 1 \
#     --pseudo_labels \
#     --save_model \
#     --wandb \
#     --pretrained_source_model pretrained_source_models/hmdb-ucf-source-head+partial-35xc7vyr-ep=19.ckpt
