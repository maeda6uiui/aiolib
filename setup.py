import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="aiolib",
    version="0.0.1-snapshot",
    author="S. Maeda",
    description="AI王用ライブラリ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maeda6uiui/aiolib",
    packages=setuptools.find_packages("src"),
    package_dir={"":"src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires='>=3.6',
)
