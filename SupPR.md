# Supervised Prompt Retrieval - Training stage

## Step1: Extract (trn) image feature by a off-the-shelf model
```
python tools/featextrater_folderwise_UnsupPR.py vit_large_patch14_clip_224.laion2b features_vit-laion2b trn
```

## Step2: Calculate similarity
```
python tools/calculate_similariity.py features_vit-laion2b trn trn
```

## Step3: Evaluation
```
sh evaluate/srun_seg_evaluate.sh features_vit-laion2b_trn output_vit-laion2b-clip_trn
```

## Step3: Get positive and negative
```
python tools/get_positive_negative.py features_vit-laion2b_trn output_vit-laion2b-clip_trn
```

## Step4: Train the feature extractor for in-context learning
```
git clone https://github.com/ZhangYuanhan-AI/SupContrast.git
git checkout pre-train-vit-freeze-encoder
sh srun_train_pretrain.sh
```
The learned feature extractor would be saved at 
``
SupContrast/save/SupCon/path_models/
``


# Supervised Prompt Retrieval - Evaluation stage

## Step1: Extract image feature by learned feature extractor
### Val
```
python tools/featextrater_folderwise_SupPR.py vit_large_patch14_clip_224.laion2b features_supcon-vit-laion2b-clip-csz224-bsz64-lr0005-freeze-encoder val
```

### Train
```
python tools/featextrater_folderwise_SupPR.py vit_large_patch14_clip_224.laion2b features_supcon-vit-laion2b-clip-csz224-bsz64-lr0005-freeze-encoder trn
```

## Step2: Calculate Similarity
```
python tools/calculate_similariity.py features_supcon-vit-laion2b-clip-csz224-bsz64-lr0005-freeze-encoder val trn
```

## Step3: Evaluation
```
sh evaluate/srun_seg_evaluate.sh features_supcon-vit-laion2b-clip-csz224-bsz64-lr0005-freeze-encoder_val output_supcon-vit-laion2b-clip-csz224-bsz64-lr0005-freeze-encoder_val
```