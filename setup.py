import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="WISE-core",
    version="0.0.1",
    author="Mohamed Eldesouki",
    author_email="mohamed@eldesouki.ca",
    description="A Natural Language Platform For Querying RDF-Based Graphs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CoDS-GCS/WISE",
    # When your source code is in a subdirectory under the project root, e.g.
    # `src/`, it is necessary to specify the `package_dir` argument.
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src',
                                      # include=['wise', 'wise.*'], exclude=["evaluation", "word_embedding"]
                                      ),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Knowledge Graph :: Natural Language Interface',
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',

    project_urls={
        'Documentation': 'https://github.com/CoDS-GCS/WISE/docs',
        # 'Funding': 'https://github.com/CoDS-GCS/WISE/donate/',
        'Say Thanks!': 'https://wise.eldesouki.ca/thankyou',
        'Source': 'https://github.com/CoDS-GCS/WISE/',
        'Tracker': 'https://github.com/CoDS-GCS/WISE/issues',
    },
    py_modules=["ganswer"],
    install_requires=[
        "transitions==0.8.0",
        "allennlp==0.9.0",
        "networkx==2.4",
        # "termcolor==1.1.0",
        "gensim==3.8.1",
        "requests",
        "graphviz",
    ],
    package_data={
        'sample': ['package_data.dat'],
    },
    include_package_data=True,
    data_files=[('my_data', ['data/data_file'])],
    test_suite='tests',
    tests_require=[],
)
