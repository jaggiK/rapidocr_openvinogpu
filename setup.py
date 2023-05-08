import sys
from pathlib import Path

import setuptools


def get_readme():
    root_dir = Path(__file__).resolve().parent
    readme_path = str(root_dir / 'README.md')
    print(readme_path)
    with open(readme_path, 'r', encoding='utf-8') as f:
        readme = f.read()
    return readme


MODULE_NAME = 'rapidocr_openvinogpu'

setuptools.setup(
    name=MODULE_NAME,
    platforms="Any",
    include_package_data=True,
    # package_dir={'': MODULE_NAME},
    packages=setuptools.find_packages(),
    package_data={'': ['models/*.onnx', '*.yaml', '*.txt']},
    python_requires='>=3.6,<=3.11'
)
