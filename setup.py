# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

import setuptools
from get_pypi_latest_version import GetPyPiLatestVersion


def get_readme():
    root_dir = Path(__file__).resolve().parent
    readme_path = str(root_dir / 'README.md')
    print(readme_path)
    with open(readme_path, 'r', encoding='utf-8') as f:
        readme = f.read()
    return readme


MODULE_NAME = 'rapidocr_openvinogpu'

obtainer = GetPyPiLatestVersion()
latest_version = obtainer(MODULE_NAME)
VERSION_NUM = "1.0.0"#obtainer.version_add_one(latest_version)

# 优先提取commit message中的语义化版本号，如无，则自动加1
if len(sys.argv) > 2:
    match_str = ' '.join(sys.argv[2:])
    matched_versions = obtainer.extract_version(match_str)
    if matched_versions:
        VERSION_NUM = matched_versions
sys.argv = sys.argv[:2]

setuptools.setup(
    name=MODULE_NAME,
    version=VERSION_NUM,
    platforms="Any",
    description="A cross platform OCR Library based on OpenVINO.",
    long_description=get_readme(),
    long_description_content_type='text/markdown',
    author="SWHL",
    author_email="jagadishkmahendran@gmail.com",
    url="https://github.com/jaggiK/RapidOCR",
    license='Apache-2.0',
    include_package_data=True,
    install_requires=["pyclipper>=1.2.1", "openvino>=2022.2.0",
                      "opencv_python>=4.5.1.48", "numpy>=1.19.3",
                      "six>=1.15.0", "Shapely>=1.7.1", 'PyYAML', 'Pillow'],
    package_dir={'': MODULE_NAME},
    packages=setuptools.find_namespace_packages(where=MODULE_NAME),
    package_data={'': ['*.onnx', '*.yaml', '*.txt']},
    keywords=[
        'ocr,text_detection,text_recognition,db,onnxruntime,paddleocr,openvino,rapidocr'
    ],
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.6,<=3.10',
    entry_points={
        'console_scripts': [f'{MODULE_NAME}={MODULE_NAME}.rapid_ocr_api:main'],
    },
    # download_url="https://github.com/jaggiK/rapidocr_openvino_gpu/archive/refs/tags/1.0.5.tar.gz",
)
