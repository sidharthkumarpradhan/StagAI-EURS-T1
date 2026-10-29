# Method description

Team: StagAI   Track(s): both (this repository: Track 1)   Reported testing-phase submission: 19062 (`6e2b4bf589d1`)

Sidharth Kumar Pradhan, Sidike Paheding. Related workshop paper: *TiDE-URS: A Teacher-Distilled Efficient Network for
Underwater Restoration and Segmentation* (MaCVi 2027 submission #21).
Uploaded zip for 19062: sha256 `5670a074d61e73b0ae41c499bed5285ecc5638b9843274c959dc99bd626e6d71`.

## Approach

**Segmentation masks.**
1. *Keyed colour correction.* For each (water type, depth) key, a 4x3 affine colour map is fitted by least squares on
   the train pairs of that key (20 scenes, 20k pixels per image). Each test image is corrected with the map of its own
   file-name key (`weights/key_affines.npz`, 200 maps).
2. *Fifteen teachers.* Three recipes, each trained 5 times on a 5-fold scene split (16 scenes each; together the folds
   cover all 20 train scenes). Frozen DINOv3 ViT-L/16 backbone (fp16 autocast); blocks 6/12/18/24 reassembled DPT-style
   to strides 4-32, a convolutional stem adds stride-4 detail, a semantic FPN head (256 channels) fuses the pyramid; the
   water/depth key enters through FiLM. Training at 768 x 1024: 12 epochs x 1,500 samples, AdamW lr 1e-3, wd 0.05;
   cross-entropy + 0.5 Lovász + restoration L1/SSIM; 30 % external photos.
   - `f053EL`: base recipe; the fish label of four external sources ignored, elasmobranch/other external samples boosted (x6, x2).
   - `hdEL`: `f053EL` with a 512-channel head.
   - `nbEL`: `f053EL` with external data from USIS16K and UIIS10K only, never-annotated pixels ignored.
3. *Mask.* Mean class probabilities of the 15 teachers on the corrected image, each with horizontal-flip TTA; argmax;
   9x9 median filter. If a mask is below 0.5 % foreground, that image's background probability is scaled down step by
   step (x0.8 ... x0.05) until 0.5 % is reached.

**Restored images.** Mean of three restoration networks: `rnS4all` (UNet, width 64, 4 levels), `rnT2all` (NAFNet,
width 32, 29 M parameters, with a physics head predicting a range field and gate that invert the image-formation model
at the image's key, plus a residual) and `rnT5all` (the NAFNet without physics head). Inputs: the image, its cubic colour
map for its key (fitted on train), the one-hot water type, depth / 100 and the image's 12 global colour statistics.
Training on all 20 scenes: 40k steps, AdamW lr 3e-4, 384 crops, loss log10(MSE) + 0.1 (1 - SSIM), plus 6-8 synthetic
crops per step re-degraded from the train clean images.

## Training data

The 20 scenes of `eurs_train.zip` only (4,000 degraded renderings with clean image and mask). No development or
testing-phase image was used for training, fine-tuning, pseudo-labelling, model selection or calibration; testing
images are only inputs at inference. Development-phase leaderboard scores guided which recipes we kept. Track 1 had one
testing-phase submission (19062), so nothing was chosen from testing-phase scores.

**External data and pretrained models.**
- Segmentation masks from USIS16K, UIIS10K, USIS10K, SUIM and Coralscapes, mapped to the five EURS classes and
  re-degraded synthetically with an image-formation model fitted on the EURS train pairs; near-duplicates of EURS images
  removed by perceptual hash.
- DINOv3 ViT-L/16 (LVD-1689M), `timm/vit_large_patch16_dinov3.lvd1689m` from the Hugging Face Hub (DINOv3 License),
  frozen; fetched by the root `Dockerfile` at build time (blob sha256 `45172f20...7941`), not stored in our checkpoints.

## Anything non-standard at inference

- File-name metadata ("Meta"): water type and depth are read from the trailing `_<type>_<level>` of each name; the scene
  part is never used.
- Keyed colour correction, 15-teacher ensemble with flip TTA, 9x9 median filter, background down-scaling for near-empty
  masks, 3-network restoration ensemble (all per image).
- restored/ is quantised to a colour step of 3 (as in the uploaded zip, done there to meet the 1 GB upload limit).

## Reproduction notes

`run.py` = `tools/predict_track1.py` (the script that rebuilt the submission from the hand-in) without the zip
packaging. Our run: one V100 16 GB, about 19 min per 200-image scene (about 2 h for the 1,200 testing images),
about 24 GB RAM. Weights: `weights/weights_link.yaml` (one .tar.gz, unpacked into `weights/`: `teachers/`,
`restorers/`, `key_affines.npz`). Our runs used CUDA fp16 autocast on V100; small run-to-run GPU differences are possible.
