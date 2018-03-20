from setuptools import setup

setup(name='ml-utils',
      version='1.0',
      description='Common tools for Machine Learning',
      author='Guillaume Jeusel',
      author_email='guillaume.jeusel@gmail.com',
      packages=['ml_utils'],
      install_requires=[
          'psutil',
          'pytorch',
          'sklearn',
          'tqdm',
          'PIL',
          'pandas',
          'numpy',
          'scipy',
      ]
     )
