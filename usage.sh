#!/usr/bin/env bash
set -e

pip install -e .

#cd ./tests && pytest -s test_basic.py::test_proj && cd -
#cd ./tests && pytest -s test_basic.py::test_rasterize_to_pixels && cd -
#cd ./tests && pytest -s test_rasterization.py::test_rasterization && cd -
cd ./examples && CUDA_VISIBLE_DEVICES=0 python simple_viewer.py && cd -

exit 1

#: << 'END'
SRC="/data/top_samsung_dong_supersplat.png"
DST="/data/SS_Orthophoto.tif"
TAG="left : 3DGS supersplat TOP, right : Samusung dong orthophoto" 
#END



: << 'END'
SRC="/data/top_samsung_dong_supersplat.png"
DST="/data/samsung_dong_mini_21/1_TOP_SAMAH.tif"
TAG="left : 3DGS supersplat TOP, right : samah TOP" 
END

: << 'END'
SRC="./tests/data/1/cropped_2024.tif"
DST="./tests/data/1/cropped_2023.tif"
TAG="left : captured in 2024, right : captured in 2023" 
END

FEATURES=( loftr )
#FEATURES=( loftr sift orb superpoint xfeat )
#FEATURES=( superpoint )
#FEATURES=( xfeat )

for FEATURE in "${FEATURES[@]}"; do
    export CUDA_VISIBLE_DEVICES=0,1,2 
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    if [ "$FEATURE" == "loftr" ]; then
        python img_tif_alignment.py --src $SRC --dst $DST --skale 0.12 --device cuda --tag "$TAG" --write_debug --feature $FEATURE --use_kornia 
    else
        python img_tif_alignment.py --src $SRC --dst $DST --skale 0.12 --device cuda --tag "$TAG" --write_debug --feature $FEATURE
    fi     
    #python img_tif_alignment.py --src ./tests/data/1/cropped_2024.tif --dst ./tests/data/1/cropped_2023.tif --feature superpoint --skale 0.12 --device cuda --write_debug 
    #python img_tif_alignment.py --src ./tests/data/1/cropped_2024.tif --dst ./tests/data/1/cropped_2023.tif --feature orb --skale 0.12 --device cuda --write_debug 
    #python img_tif_alignment.py --src ./tests/data/1/cropped_2024.tif --dst ./tests/data/1/cropped_2023.tif --feature xfeat --write_debug
    #python img_tif_alignment.py --src ./tests/data/2/3/cropped_2024.tif --dst ./tests/data/2/3/cropped_2023.tif --feature xfeat --write_debug
    #python tif_alignment.py --src ./tests/data/1/cropped_2024.tif --dst ./tests/data/1/cropped_2023.tif --feature xfeat --write_debug
    #python tif_alignment.py --src ./tests/data/2/3/cropped_2024.tif --dst ./tests/data/2/3/cropped_2023.tif --feature xfeat --write_debug
done    


: << 'END'

#exit 1
#echo "dir_data $dir_data"
#path_dsm=${dir_data}/s2p_out/dsm.tif
path_dsm=${dir_data}/tmp/dsm_4326.tif
#path_dem=${dir_data}/s2p_out/dem_fake.tif
dir_out=$dir_data/gwarp_out
dir_cache=./cache

rm -rf $dir_cache

bash compile.sh

if [ -n "${path_dem+x}" ]; then
    echo "DEM file is provided as '${path_dem}'"
    #python orthorectify.py -folder_img $dir_data -folder_rpc $dir_data -dsm_file $path_dsm -folder_output $dir_out -cache_dir $dir_cache -nodata $val_no_data -dem_file $path_dem -uint8_out 
    python orthorectify.py -folder_img $dir_data -dsm_file $path_dsm -folder_output $dir_out -cache_dir $dir_cache -nodata $val_no_data -dem_file $path_dem -uint8_out 
else
    echo "DEM file is not provided. So it is estimated inside codes."
    python orthorectify.py -folder_img $dir_data -folder_rpc $dir_data -dsm_file $path_dsm -folder_output $dir_out -cache_dir $dir_cache -nodata $val_no_data -uint8_out 
fi
END
