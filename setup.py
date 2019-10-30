import os

from setuptools import find_packages, setup


def local_file(name):
    return os.path.relpath(os.path.join(os.path.dirname(__file__), name))

SOURCE = local_file("src")
README = """
==========
Shrink Ray
==========

Shrink Ray is a new type of test-case reducer designed to be effective on
a wide range of formats. See the
[full README](https://github.com/DRMacIver/shrinkray/blob/master/README.md)
for more details.
"""


setup(
    name='shrinkray',
    version="0.0.1",
    author='David R. MacIver',
    author_email='david@drmaciver.com',
    packages=find_packages(SOURCE),
    package_dir={"": SOURCE},
    url='https://github.com/DRMacIver/shrinkray',
    license='MPL 2.0',
    description='A test case reducer',
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    install_requires=['click', 'tqdm'],
    long_description=README,
    entry_points={
        'console_scripts': [
            'shrinkray=shrinkray.__main__:reducer'
        ]
    }
)
