from setuptools import setup, find_packages

setup(
    name='cyra',
    version='1.0',
    author='Andreev S.',
    author_email='sicome.a.s@gmail.com',
    description='Large language model for text generation',
    packages=['cyra_model', 'dataset_preparing']
    # packages=find_packages(),
    # install_requires=[
    #     'package1',
    #     'package2',
    # ],
)