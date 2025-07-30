#!/usr/bin/env bash
set -e

pip install -e .
#pip install submodules/simple-knn submodules/fused-ssim submodules/diff-gaussian-rasterization
pip install submodules/fused-ssim

#: << 'END'
########    viewing ########
#cd ./examples && CUDA_VISIBLE_DEVICES=0 python simple_viewer.py && cd -
#cd ./examples && CUDA_VISIBLE_DEVICES=1 python simple_viewer.py --ckpt "./results/samsung_dong_mini_5_default/ckpts/ckpt_6999_rank0.pt" && cd -
#cd ./examples && CUDA_VISIBLE_DEVICES=0 python simple_viewer.py --ply "/data/iter_002000.ply" && cd -
cd ./examples && CUDA_VISIBLE_DEVICES=0 python simple_viewer.py --gs_ply "/data/point_cloud_samsung_dong_aoi_3dgs.ply" && cd -
#END

#: << 'END'
########    training  ########
#export PYTHONPATH="$PYTHONPATH:./thridparty//path/to/AA/CC"
cd ./examples && CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default && cd -
#cd ./examples && CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc && cd -
#END
