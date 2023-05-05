# Python RapidOCR OpenVINO with GPU (with GPU only)

This is a modified verison of RapidOCR (https://github.com/RapidAI/RapidOCR) to support OpenVINO GPU. Currently works only with for fixed size images (max len of 960) and also needs the image size to a multiple of 32.

## Installation:

`git clone https://github.com/jaggiK/rapidocr_openvinogpu.git`

`python3 setup.py install`

`cd rapidocr_openvinogpu`

## Inference:

### Run inference for all the images in a given directory
`python3 demo.py -d <absolute_path/to/directory>`

### Infering an image:
`python3 demo.py -f <absolute_path/to/image.jpg>`

### To save inference results, use `-v` flag:
`python3 demo.py -d <absolute_path/to/directory> -v` # this saves the visualization in "./inference_results/" in the current folder 

### To save inference results in a desired folder, use `-v` and `-o` flag:
`python3 demo.py -d <absolute_path/to/directory> -v -o <absolute_path/to/directory>`