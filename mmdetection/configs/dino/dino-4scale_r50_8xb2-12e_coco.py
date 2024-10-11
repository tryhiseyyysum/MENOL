_base_ = [
    '../_base_/datasets/coco_detection.py', '../_base_/default_runtime.py'
]
model = dict(
    type='DINO',
    #num_queries=900,  # num_matching_queries
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4,
                               dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0))),  # 0.1 for DeformDETR
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8,
                               dropout=0.0),  # 0.1 for DeformDETR
            cross_attn_cfg=dict(embed_dims=256, num_levels=4,
                                dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0)),  # 0.1 for DeformDETR
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,  # -0.5 for DeformDETR
        temperature=20),  # 10000 for DeformDETR
    bbox_head=dict(
        type='DINOHead',
        num_classes=30,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))  # 100 for DeformDETR

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    #dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),

    ####new####
    dict(type='LoadImageFromFile'),


    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip',prob=0.5),
    dict(
        type='RandomChoice',
        transforms=
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
    ),
    #         [
    #             # dict(
    #             #     type='RandomChoiceResize',
    #             #     # The radio of all image in train dataset < 7
    #             #     # follow the original implement
    #             #     scales=[(400, 4200), (500, 4200), (600, 4200)],
    #             #     keep_ratio=True),
    #             dict(
    #                 type='RandomCrop',
    #                 crop_type='absolute_range',
    #                 crop_size=(384, 600),
    #                 allow_negative_crop=True),
    #             # dict(
    #             #     type='RandomChoiceResize',
    #             #     scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
    #             #             (608, 1333), (640, 1333), (672, 1333), (704, 1333),
    #             #             (736, 1333), (768, 1333), (800, 1333)],
    #             #     keep_ratio=True)
    #         ]
    #    ),
    dict(type='PackDetInputs')
    
]


train_dataloader = dict(
    dataset=dict(
        filter_cfg=dict(filter_empty_gt=False), pipeline=train_pipeline))

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 0.0002 for DeformDETR
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa

# learning policy
max_epochs = 12
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[11],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=2,enable=True)
# auto_scale_lr.enable=True
# auto_scale_lr.base_batch_size=8

###补充   P1
#dataset_type = 'CocoSplitDataset'
# data = dict(
#     samples_per_gpu=1,
#     workers_per_gpu=1,
#     train=dict(
#         is_class_agnostic=True,
#         train_class='voc',
#         eval_class='nonvoc',
#         type=dataset_type,
#         pipeline=train_pipeline,
#         ),
#     val=dict(
#         is_class_agnostic=True,
#         train_class='voc',
#         eval_class='nonvoc',
#         type=dataset_type,
#         pipeline=train_pipeline),
#     test=dict(
#         is_class_agnostic=True,
#         train_class='voc',
#         eval_class='nonvoc',
#         type=dataset_type,
#         pipeline=train_pipeline))


#####P2


dataset_type = 'CocoSplitDataset'
data_root = './dataset/' 
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(
#         type='AutoAugment',
#         # policies=[[
#         #     dict(
#         #         type='Resize',
#         #         img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
#         #                    (608, 1333), (640, 1333), (672, 1333), (704, 1333),
#         #                    (736, 1333), (768, 1333), (800, 1333)],
#         #         multiscale_mode='value',
#         #         keep_ratio=True)
#         # ],

#         policies=[[
#                       dict(
#                           type='Resize',
#                           #img_scale=[(720, 1280), (500, 1333), (600, 1333)],
#                           img_scale=[(1920,1080)],
#                           multiscale_mode='value',
#                           keep_ratio=True),
#                     #   dict(
#                     #       type='RandomCrop',
#                     #       crop_type='absolute_range',
#                     #       crop_size=(384, 600),
#                     #       allow_negative_crop=True),
#                     #   dict(
#                     #       type='Resize',
#                     #       img_scale=[(480, 1333), (512, 1333), (544, 1333),
#                     #                  (576, 1333), (608, 1333), (640, 1333),
#                     #                  (672, 1333), (704, 1333), (736, 1333),
#                     #                  (768, 1333), (800, 1333)],
#                     #       multiscale_mode='value',
#                     #       override=True,
#                     #       keep_ratio=True)
#                   ]]),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=1),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
# ]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        #img_scale=(1920, 1080),
        img_scale=(1280, 720),
        flip=False,

        transforms=[
            dict(type='Resize', keep_ratio=True),
            #dict(type='RandomFlip',prob=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
            dict(type='PackDetInputs')
            
        ]         
        )
        # dict(type='Resize', keep_ratio=True),
        # dict(type='RandomFlip',prob=0.5),
        # dict(type='Normalize', **img_norm_cfg),
        # dict(type='Pad', size_divisor=32),
        # dict(type='ImageToTensor', keys=['img']),
        # dict(type='Collect', keys=['img']),
        # dict(type='PackDetInputs')
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type="CocoSplitPseudoBoxDataset",
        is_class_agnostic=False,
        train_class=("pedestrian", "cyclist", "car", "truck", "tram", "tricycle", "bus",
        "bicycle", "moped", "motorcycle", "stroller","cart","construction_vehicle","dog", "barrier",
        "bollard","sentry_box","traffic_cone","traffic_island","traffic_light","traffic_sign","debris","suitcace",
        "dustbin","concrete_block","machinery","garbage","plastic_bag","stone","misc"),
        eval_class=("pedestrian", "cyclist", "car", "truck", "tram", "tricycle", "bus",
        "bicycle", "moped", "motorcycle", "stroller","cart","construction_vehicle","dog", "barrier",
        "bollard","sentry_box","traffic_cone","traffic_island","traffic_light","traffic_sign","debris","suitcace",
        "dustbin","concrete_block","machinery","garbage","plastic_bag","stone","misc"),
        pipeline=train_pipeline,
        
        # add pseudo data
        additional_ann_file=['./Final_test/pseudo_depth.json','./Final_test/pseudo_normal.json'],
        iou_thresh=0.5,
        score_thresh=None,
        top_k=1,
        random_sample_masks=False,
        merge_nms=True
        ),
    val=dict(
        is_class_agnostic=False,
        train_class=("pedestrian", "cyclist", "car", "truck", "tram", "tricycle", "bus",
        "bicycle", "moped", "motorcycle", "stroller","cart","construction_vehicle","dog", "barrier",
        "bollard","sentry_box","traffic_cone","traffic_island","traffic_light","traffic_sign","debris","suitcace",
        "dustbin","concrete_block","machinery","garbage","plastic_bag","stone","misc"),
        eval_class=("pedestrian", "cyclist", "car", "truck", "tram", "tricycle", "bus",
        "bicycle", "moped", "motorcycle", "stroller","cart","construction_vehicle","dog", "barrier",
        "bollard","sentry_box","traffic_cone","traffic_island","traffic_light","traffic_sign","debris","suitcace",
        "dustbin","concrete_block","machinery","garbage","plastic_bag","stone","misc"),
        type=dataset_type,
        pipeline=test_pipeline),
    test=dict(
        is_class_agnostic=False,
        train_class=("pedestrian", "cyclist", "car", "truck", "tram", "tricycle", "bus",
        "bicycle", "moped", "motorcycle", "stroller","cart","construction_vehicle","dog", "barrier",
        "bollard","sentry_box","traffic_cone","traffic_island","traffic_light","traffic_sign","debris","suitcace",
        "dustbin","concrete_block","machinery","garbage","plastic_bag","stone","misc"),
        eval_class=("pedestrian", "cyclist", "car", "truck", "tram", "tricycle", "bus",
        "bicycle", "moped", "motorcycle", "stroller","cart","construction_vehicle","dog", "barrier",
        "bollard","sentry_box","traffic_cone","traffic_island","traffic_light","traffic_sign","debris","suitcace",
        "dustbin","concrete_block","machinery","garbage","plastic_bag","stone","misc"),
        type=dataset_type,
        pipeline=test_pipeline))

# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[12, 14])
# runner = dict(type='EpochBasedRunner', max_epochs=35)

# checkpoint_config = dict(interval=2)
# # yapf:disable
# log_config = dict(
#     interval=10,
#     hooks=[
#         dict(type='TextLoggerHook'),
#         # dict(type='TensorboardLoggerHook')
#     ])
# # yapf:enable
# dist_params = dict(backend='nccl')
# log_level = 'INFO'
# load_from = None
# resume_from = None
# workflow = [('train', 1)]