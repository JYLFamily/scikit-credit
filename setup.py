import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="skcredit",
    version="0.0.1",
    author="JYLFamily",
    author_email="jiangyilanf@gmail.com",
    description="信贷风控评分卡建模的自动化工具",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JYLFamily/scikit-credit",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "joblib",
        "pandas",
        "sklearn",
        "portion",
        "statsmodels"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Unix",
    ],
)
