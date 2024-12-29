from distutils.core import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('requirements_optional.txt') as f_optional:
    optional_requirements = f_optional.read().splitlines()
    requirements.extend(optional_requirements)

setup(
    name='dust3r',
    version='1.0.0',
    packages=['dust3r', "dust3r_visloc"],
    url='',
    license='',
    author='jizong',
    author_email='',
    description='',
    install_requires=requirements
)
