from setuptools import setup, find_packages

setup(
    name='anndict',
    version='0.1',
    packages=find_packages(),
    description='Conveniently process a dictionary of anndatas (adata_dict)',
    author='ggit12',
    # author_email='your.email@example.com',
    license='BSD-3-Clause',
    install_requires=[
        'numpy==1.26.4', 
        'pandas==2.2.2',
        'scikit-learn==1.5.1',
        'scanpy==1.10.2',
        'anndata==0.10.8',
        'IPython==8.26.0',
        'scipy==1.14.1',
        'seaborn==0.13.2',
        'matplotlib==3.9.2',
        'squidpy==1.6.0',
        'harmonypy==0.0.10',
        'langchain==0.2.14', 
        'langchain-community==0.2.12', 
        'langchain-openai==0.1.22',
        'langchain-anthropic==0.1.23',
        'langchain-google-genai==1.0.10', 
        'langchain-aws==0.1.17',
        'boto3==1.34.162',
        'bokeh==3.4.3',
        'holoviews==1.19.1',
        'krippendorff==0.7.0'
    ],
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.org/classifiers/
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
)
