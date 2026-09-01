set -e
 
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
 
cd "$BASE_DIR/sam3/sam3"
python3 sam3_patch_pipeline.py
 
cp sharpened_frame_500.png "$BASE_DIR/PET/"
 
cd "$BASE_DIR/PET"
python3 PET_inference_pipline.py
 
cd "$BASE_DIR/sam2"
python3 fish_segmentation_pipeline.py
