import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="toads-ivanrychkov",
    version="0.0.15",
    author="Ivan Rychkov",
    author_email="rychyrych@yandex.ru",
    description="Data Science tools from preprocessing and visualization to statistics and ML.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ivanrychkov/toads",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
