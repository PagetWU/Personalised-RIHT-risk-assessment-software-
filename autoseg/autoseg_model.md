# Thyroid Auto-Segmentation Model

The optional thyroid auto-segmentation model was developed using nnU-Net v2. The checkpoint is hosted externally because its file size exceeds the GitHub repository limit.

## Download

[Download the model checkpoint from Google Drive](https://drive.google.com/file/d/11w32m51_bM2XIsvkfD604FJQ0GFpoJRB/view?usp=drive_link)

## Installation

After downloading, place `checkpoint_best.pth` in the following directory:

```text
autoseg_model/
└── Dataset1102_ThyroidSegmentation/
    └── nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/
        └── fold_0/
            └── checkpoint_best.pth
```

The checkpoint is optional when a thyroid mask is supplied directly to the software.
