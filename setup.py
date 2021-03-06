from setuptools import setup

setup(name='ml-utils',
      version='1.0',
      description='Common tools for Machine Learning',
      author='Guillaume Jeusel',
      author_email='guillaume.jeusel@gmail.com',
      packages=['ml_utils'],
      install_requires=[
          'psutil',
          'sklearn',
          'tqdm',
          'Pillow',
          'pandas',
          'numpy',
          'scipy',

          # Plot purpose:
          'matplotlib',
          'colorlover',
          'plotly',
          'seaborn',
      ]
     )
