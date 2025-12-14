from setuptools import __version__, find_packages
from setuptools import setup

from hypopredict import __version__

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='hypopredict',
      version=__version__,
      description="HypoPredict Module",
      license="MIT",
      author="HypoPredict Team",
      author_email="sasha17demin@gmail.com",
      url="https://github.com/sasha17demin/hypopredict",
      install_requires=requirements,
      packages=find_packages(),
      #test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
