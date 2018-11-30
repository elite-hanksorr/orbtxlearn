import setuptools

with open('README.md') as f:
    long_description = f.read()

setuptools.setup(
    name='orbtxlearn',
    description='AI for OrbtXLearn',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='0.0.0',
    url='https://github.com/elite-hanksorr/orbtxlearn',
    packages=[
        'orbtxlearn',
    ],
    python_requires='>=3.6',
    install_requires=[
        'click',
        'mss',
        'numpy',
        'pillow',
        'pyautogui',
        'scikit-image',
        'scipy',
    ],
    extras_require={
        'cpu': [
            'tensorflow',
        ],
        'gpu': [
            'tensorflow-gpu',
        ]
    },
    classifiers=[
        'Topic :: Games/Entertainment',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Operating System :: Microsoft :: Windows',  # orbtxlearn-spy is completely untested on linux
        'Natural Language :: English',
    ],
)
