## RapidOCR OpenVINO GPU Python

This is a modified verison of RapidOCR (https://github.com/RapidAI/RapidOCR) to support OpenVINO GPU. Currently works only with for fixed size images (max len of 960) and also needs the image size to a multiple of 32.

## Installation:

git clone https://github.com/jaggiK/rapidocr_openvinogpu.git

python3 setup.py install

cd rapidocr_openvinogpu

## Inference

### Run inference for all the images in a given directory
python3 demo.py -d <absolute_path/to/directory>

### Infering an image:
python3 demo.py -f <absolute_path/to/image.jpg>

### To save inference results, use `-v` flag:
python3 demo.py -d <absolute_path/to/directory> -v