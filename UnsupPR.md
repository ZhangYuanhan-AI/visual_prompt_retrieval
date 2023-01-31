# Unsupervised Prompt Retrieval

## Step1: Extract image feature
### Val
```
python tools/featextrater_folderwise_UnsupPR.py vit_large_patch14_clip_224.laion2b features_vit-laion2b val
```

### Train
```
python tools/featextrater_folderwise_UnsupPR.py vit_large_patch14_clip_224.laion2b features_vit-laion2b trn
```

## Step2: Calculate similarity
```
python tools/calculate_similariity.py features_vit-laion2b val trn
```

## Step3: Evaluation
```
sh evaluate/srun_seg_evaluate.sh features_vit-laion2b_val output_vit-laion2b-clip_val
```
