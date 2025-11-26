## 3D Conditional Diffusion Models for Synthetic Cryo-ET Particle Subtomograms

Project layout:

    CryoDiff/
    ├─ README.md
    ├─ requirements.txt
    ├─ train.py              # trains conditional 3D DDPM (predicts x0)
    ├─ generate.py           # reverse diffusion from noise with class+coord cond
    ├─ data/
    │  └─ particles_dataset.py      # loads tomos + crops particle-centered 64^3
    ├─ models/
    │  └─ diffusion_unet3d.py       # 3D UNet with time/class/coord conditioning
    ├─ diffusion/
    │  └─ schedule.py               # betas, q(x_t|x0), and sampling step
    └─ utils/
       ├─ logger.py                 
       └─ visualization.py          # quick central-slice PNGs

Installation:
    
    cd CryoET-Diff
    python3 -m venv .venv
    source .venv/bin/activate.csh
    pip install -r requirements.txt

Make sure data layout is:
    
    dataset/
        TS_69_2.mrc
        TS_5_4.mrc
        ...
        particles.csv \

Train:
    
    python train.py \
      --train_tomo_dir Dataset \
      --csv_path particles.csv \
      --patch_size 64 \
      --epochs 100 \
      --batch_size 16 \
      --lr 1e-4 \
      --save_dir outputs

Inference:
    
    python generate.py \
      --checkpoint outputs/model_epoch100.pt \
      --class_index <your_class_index> \
