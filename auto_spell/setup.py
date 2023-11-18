from setuptools import setup

with open("README", 'r') as f:
    long_description = f.read()

setup(
   name='auto_spell',
   version='1.0',
   description='Identify spelling errors',
   license="MIT",
   long_description=long_description,
   author='KP',
   author_email='',
   url="",
   packages=['auto_spell'],  #same as name
   install_requires=['torch','numpy'], #external packages as dependencies
   scripts=[

           ]
)