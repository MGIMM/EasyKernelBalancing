from setuptools import setup
sourcefiles = ['./EasyKernelBalancing/EasyKernelBalancing.py']
setup(
        name='EasyKernelBalancing',
        version='0.1.0',
        description='A basic kernel balancing implementation with naive GD.',
        url='https://github.com/MGIMM/EasyKernelBalancing',
        author='Qiming Du',
        author_email='qiming.du@upmc.fr',
        license='MIT',
        package_dir={"": "src"},
        packages=setuptools.find_packages(where="src"),
        install_requires=[
               'tqdm',
               'numpy',
               #'seaborn'
           ],
            python_requires='>=3.7',
    )
