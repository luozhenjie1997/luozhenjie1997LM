from setuptools import setup, find_packages

setup(
    name="ImmunoglobulinGenerate",
    version="0.0.0",
    packages=find_packages(),
    # packages=[
    #     'openfold',
    #     'pepflow',
    #     'data',
    # ],
    package_dir={
        'openfold': './openfold',
        'ImmunoglobulinGenerate': './model',
    },
)
