"""Aim to distribute and install Python package of the rag system."""
import io
import os

from setuptools import find_packages, setup

# Metadata of package
NAME = 'RAG_seven_wonders'
DESCRIPTION = 'AI RAG Q&A system about the seven ancient wonders.'
URL = 'https://github.com/vladimirkanchev/haystack-train-rag/'
AUTHOR = 'Vladimir Kanchev'
EMAIL = 'kanchev.vladimir@gmail.com'
REQUIRES_PYTHON = '>=3.10'

pwd = os.path.abspath(os.path.dirname(__file__))


def list_reqs(fname='requirements.txt'):
    """List packages to be installed."""
    with io.open(os.path.join(pwd, fname), encoding='utf-8') as f:
        return f.read().splitlines()


with open('VERSION', encoding="utf-8") as version_file:
    _version = version_file.read().strip()

try:
    with io.open(os.path.join(pwd, 'README.md'), encoding='utf-8') as file:
        LONG_DESCRIPTION = '\n' + file.read()
except FileNotFoundError:
    LONG_DESCRIPTION = DESCRIPTION

setup(
    name=NAME,
    version=_version,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    include_package_data=True,
    packages=find_packages(),
    package_data={
     'rag_system': ['config.yml'],
    },
    install_requires=list_reqs(),
    extras_require={},
    license='Apache License Version 2.0',
    classifiers=[
        'Programming language :: Python :: 3',
        'Programming language :: Python :: 3.10',
        'License :: OSI Aproved :: Apache License Version 2.0',
        'Operating System :: OS I'
    ],
    entry_points={
        'console_scripts': [
            'start-rag-system=rag_system.app_fastapi:run',  # Entry point
            'start-streamlit-app=rag_system.run_app_streamlit:main'
        ]
    },
)
