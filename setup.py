import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='se2flowpipes',
    version='0.0.1',
    author='Li-Yu Lin',
    author_email='liyu8561501@gmail.com',
    description='Flow Pipes creation for SE2 Lie group',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/melodylylin/SE2_FlowPipes',
    # project_urls = {
    #     "Bug Tracker": "https://github.com/Muls/toolbox/issues"
    # },
    license='MIT',
    packages=['se2_flowpipes'],
    install_requires=['requests'],
)
