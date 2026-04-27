--------- STAEformer ---------
{
    "num_nodes": 207,
    "in_steps": 12,
    "out_steps": 12,
    "train_size": 0.7,
    "val_size": 0.1,
    "time_of_day": true,
    "day_of_week": true,
    "lr": 0.001,
    "weight_decay": 0.0003,
    "milestones": [
        20,
        30
    ],
    "lr_decay_rate": 0.1,
    "batch_size": 16,
    "max_epochs": 200,
    "early_stop": 30,
    "use_cl": false,
    "cl_step_size": 2500,
    "model_args": {
        "num_nodes": 207,
        "in_steps": 12,
        "out_steps": 12,
        "steps_per_day": 288,
        "input_dim": 3,
        "output_dim": 1,
        "input_embedding_dim": 24,
        "tod_embedding_dim": 24,
        "dow_embedding_dim": 24,
        "spatial_embedding_dim": 0,
        "adaptive_embedding_dim": 80,
        "feed_forward_dim": 256,
        "num_heads": 4,
        "num_layers": 3,
        "dropout": 0.1
    }
}
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
STAEformer                               [16, 12, 207, 1]          198,720
├─Linear: 1-1                            [16, 12, 207, 24]         96
├─Embedding: 1-2                         [16, 12, 207, 24]         6,912
├─Embedding: 1-3                         [16, 12, 207, 24]         168
├─ModuleList: 1-4                        --                        --
│    └─SelfAttentionLayer: 2-1           [16, 12, 207, 152]        --
│    │    └─AttentionLayer: 3-1          [16, 207, 12, 152]        93,024
│    │    └─Dropout: 3-2                 [16, 207, 12, 152]        --
│    │    └─LayerNorm: 3-3               [16, 207, 12, 152]        304
│    │    └─Sequential: 3-4              [16, 207, 12, 152]        78,232
│    │    └─Dropout: 3-5                 [16, 207, 12, 152]        --
│    │    └─LayerNorm: 3-6               [16, 207, 12, 152]        304
│    └─SelfAttentionLayer: 2-2           [16, 12, 207, 152]        --
│    │    └─AttentionLayer: 3-7          [16, 207, 12, 152]        93,024
│    │    └─Dropout: 3-8                 [16, 207, 12, 152]        --
│    │    └─LayerNorm: 3-9               [16, 207, 12, 152]        304
│    │    └─Sequential: 3-10             [16, 207, 12, 152]        78,232
│    │    └─Dropout: 3-11                [16, 207, 12, 152]        --
│    │    └─LayerNorm: 3-12              [16, 207, 12, 152]        304
│    └─SelfAttentionLayer: 2-3           [16, 12, 207, 152]        --
│    │    └─AttentionLayer: 3-13         [16, 207, 12, 152]        93,024
│    │    └─Dropout: 3-14                [16, 207, 12, 152]        --
│    │    └─LayerNorm: 3-15              [16, 207, 12, 152]        304
│    │    └─Sequential: 3-16             [16, 207, 12, 152]        78,232
│    │    └─Dropout: 3-17                [16, 207, 12, 152]        --
│    │    └─LayerNorm: 3-18              [16, 207, 12, 152]        304
├─ModuleList: 1-5                        --                        --
│    └─SelfAttentionLayer: 2-4           [16, 12, 207, 152]        --
│    │    └─AttentionLayer: 3-19         [16, 12, 207, 152]        93,024
│    │    └─Dropout: 3-20                [16, 12, 207, 152]        --
│    │    └─LayerNorm: 3-21              [16, 12, 207, 152]        304
│    │    └─Sequential: 3-22             [16, 12, 207, 152]        78,232
│    │    └─Dropout: 3-23                [16, 12, 207, 152]        --
│    │    └─LayerNorm: 3-24              [16, 12, 207, 152]        304
│    └─SelfAttentionLayer: 2-5           [16, 12, 207, 152]        --
│    │    └─AttentionLayer: 3-25         [16, 12, 207, 152]        93,024
│    │    └─Dropout: 3-26                [16, 12, 207, 152]        --
│    │    └─LayerNorm: 3-27              [16, 12, 207, 152]        304
│    │    └─Sequential: 3-28             [16, 12, 207, 152]        78,232
│    │    └─Dropout: 3-29                [16, 12, 207, 152]        --
│    │    └─LayerNorm: 3-30              [16, 12, 207, 152]        304
│    └─SelfAttentionLayer: 2-6           [16, 12, 207, 152]        --
│    │    └─AttentionLayer: 3-31         [16, 12, 207, 152]        93,024
│    │    └─Dropout: 3-32                [16, 12, 207, 152]        --
│    │    └─LayerNorm: 3-33              [16, 12, 207, 152]        304
│    │    └─Sequential: 3-34             [16, 12, 207, 152]        78,232
│    │    └─Dropout: 3-35                [16, 12, 207, 152]        --
│    │    └─LayerNorm: 3-36              [16, 12, 207, 152]        304
├─Linear: 1-6                            [16, 207, 12]             21,900
==========================================================================================
Total params: 1,258,980
Trainable params: 1,258,980
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 16.96
==========================================================================================
Input size (MB): 0.48
Forward/backward pass size (MB): 2541.39
Params size (MB): 4.24
Estimated Total Size (MB): 2546.11
==========================================================================================

Loss: MaskedMAELoss