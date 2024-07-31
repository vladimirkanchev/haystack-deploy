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
    install_requires=['cryptography',
'Cython==3.0.10',
'datasets==2.20.0',
'docutils==0.21.2',
'fastapi==0.111.1',
'filelock==3.13.1',
'haystack==0.42',
'hid-converge==2.0.1.1',
'HTMLParser==0.0.2',
'httpsproxy_urllib2==1.0',
'import-java==0.6',
'importlib_metadata==7.2.1',
'Jinja2==3.1.4',
'keyring==25.2.1',
'milvus-haystack==0.0.9',
'numpy==1.26.4',
'packaging==24.1',
'pillow==10.4.0',
'pip==24.2',
'PyConfigParser==1.0.5',
'pyOpenSSL==24.2.1',
'pyparsing==3.1.2',
'PySocks==1.7.1',
'python-box==7.2.0',
'python-dotenv==1.0.1',
'PyYAML==6.0.1',
'railroad==0.5.0',
'redis==5.0.8',
'setuptools==72.1.0',
'Sphinx==8.0.2',
'streamlit==1.37.0',
'railroad==0.5.0',
'torch==2.1.2+cpu',
'tornado==6.4.1',
'typing_extensions==4.12.2',
'uvicorn==0.30.3',
'wheel==0.43.0',
'python-ntlm==1.1.0',
'ignition-api'
],
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

