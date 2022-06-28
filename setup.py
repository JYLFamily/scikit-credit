import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="skcredit",
    version="0.0.5",
    author="JYLFamily",
    author_email="jiangyilanf@gmail.com",
    description="scorecard",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JYLFamily/scikit-credit",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "joblib",
        "pandas",
        "scikit-learn",
        "portion",
        "statsmodels"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.7" ,
        "Programming Language :: Python :: 3.8" ,
        "Programming Language :: Python :: 3.9" ,
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
