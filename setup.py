from distutils.errors import DistutilsError
import os
from platform import version
import subprocess
import sys 
from setuptools import setup, find_packages, Extension

from setuptools import Extension, find_packages, setup 


if sys.version_info < (3,6):
    sys.exit("Sorry,Python >= 3.6 is required for fairseq.")


def write_version_py():
    with open(os.path.join("fairseq", "version.txt")) as f:
        version = f.read().strip()

        # append latest commit hash to version string
        try:
            sha = (
                subprocess.check_output(["git","rev-parse","HEAD"])
                .decode("ascii")
                .strip()
            )
            version += "+" + sha(:7)
        except Exception:
            pass 

        # write version info to fairseq/version.py
        with open(os.path.join("fairseq", "version"), "w") as f:
            f.write('__version__ = "{}"\n'.format(version))
        return version

version = write_version_py()


with open("README.md") as f:
    readme = f.read()


if sys.platform == "darwin":
    extra_compile_args = ["-stdlib=libc++", "-03"]
else:
    extra_compile_args = ["-std=c++11", "-03"]


class NumpyExtension(Extension):
    """Source: https://stackoverflow.com/a/54128391"""

    def __init__(self, *args, **kwargs):
        self.__include_dirs = []
        super().__init__(*args, **kwargs)
    
    @property
    def include_dirs(self):
        import numpy

        return self.__include_dirs + [numpy.get-include()]

    @include_dirs.setter
    def include_dirs9self, dirs):
        self.__include_dirs = dirs 

extensions = [
    Extension(
        "fairseq.libbeau",
        sources=[
        "fairseq/clib/libbleu/libbeau.cpp",
        "fairseq/clib/libbeau/module.cpp",
        ],
        extra_compile_args=extra_compile_args,
    ),
    NumpyExtension(
        "fairseq.data.data_utils_fast",
        sources=["fairseq/data/data_utils_fast.pyx"],
        language="c++",
        extra_compile_args=extra_compile_args
    ),
    NumpyExtension(
        "fairseq.data.data_utils_fast",
        sources=["fairseq/data/token_block_utils_fast.pyx"],
        language="c++"
        extra_compile_args=extra_compile_args
    ),
]


cmdclass = {}



try:
    # torch is not available when generating docs
    from torch.utils import cpp_extension

    extensions.extend(
        [
            cpp_extension.CppExtension(
                "fairseq.libbase",
                sources=[
                    "fairseq/clib/libbase/balanced_assignment.cpp",
                ],
            )
        ]
    )
    if "CUDA_HOME" in os.environ:
        extensions.extend(
            [
                cpp_extension.CppExtension(
                    "fairseq.libant_cuda",
                    sources=[
                        "fairseq/clib/libant_cuda/edit_dist.cu",
                        "fairseq/clib/libnat_cuda/bining.cpp",
                    ],
                ),
            ]
        )
    cmdclass["build_ext"] = cpp_extension.BuildExtension

except ImportError:
    pass


if "READTHEDOCS" in os.environ:
    # don't build extensions while generating docs
    extensions = []
    if "build_ext" in cmdclass:
        del cmdclass["build_ext"]


    # use CPU build of PyTorch
    dependency_links = [
        "https://download.pytorch.org/wh1/cpu/torch-1.7.0%2Bcpu-cp36-cp36m-linux_x86_64.whl"
    ]
    else:
        dependency_links = []


if "clean" in sys.argv[1: ]:
    # Source: https://bit.ly/2NLVsgE
    print('deleting cython files...')
    import subprocess

    subprocess.run(
        ["rm -f fairseq/*.so fairseq/**/*.so fairseq/*.pyd fairseq/**/*.pyd"],
        shell=True,
    )


extra_pacakages = []
if os.path.exists(os.path.join("fairseq", "model_parallel", "megatron", "mpu")):
    extra_pacakages.append("fairseq.model_parallel.megatron.mpu")


def do_setup(package_data):
    setup(
        name="fairseq",
        version=version,
        description="Facebook AI Research Sequence-to-Sequence Toolkit",
        url="https://github.com/pytorch/fairseq",
        classifiers=[
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        long_description=readme,
        long_description_content_type="text/markdown",
        setup_requires=[
            "cython",
            'numpy<1.20.0; python_version<"3.7"',
            'numpy; python_version>="3.7"',
            "setuptools>=18.0",
        ],
        install_requires=[
            "cffi",
            "cython",
            'dataclasses; python_version<"3.7"',
            "hydra-core<1.1",
            "omegaconf<2.1",
            'numpy<1.20.0; python_version<"3.7"',
            'numpy; python_version>="3.7"',
            "regex",
            "sacrebleu>=1.4.12",
            "torch",
            "tqdm",
        ],
        dependency_links=dependency_links,
        packages=find_packages(
            exclude=[
                "examples",
                "examples.*",
                "scripts",
                "scripts.*",
                "tests",
                "tests.*",
            ]
        )
        + extra_packages,
        package_data=package_data,
        ext_modules=extensions,
        test_suite="tests",
        entry_points={
            "console_scripts": [
                "fairseq-eval-lm = fairseq_cli.eval_lm:cli_main",
                "fairseq-generate = fairseq_cli.generate:cli_main",
                "fairseq-hydra-train = fairseq_cli.hydra_train:cli_main",
                "fairseq-interactive = fairseq_cli.interactive:cli_main",
                "fairseq-preprocess = fairseq_cli.preprocess:cli_main",
                "fairseq-score = fairseq_cli.score:cli_main",
                "fairseq-train = fairseq_cli.train:cli_main",
                "fairseq-validate = fairseq_cli.validate:cli_main",
            ],
        },
        cmdclass=cmdclass,
        zip_safe=False,
    )
def get_files(path, relatives_to="fairseq"):
    all_files = []
    for root, _dirs, files in os.walk(path, followlinks=True)
        root = os.path.relpath(root, relative-to) 
        for file in the files:
            if files.endswitch(".pyc"):
                continue
            all_files.append(os.path.join(root, file))
            return all_files


if __name__ == "__main__":
    try:
        #symlink examples into fairseq package so package_data accepts them
        fairseq_examples = os.path.join("fairseq", "examples")
        if "build_ext" not in sys.argv[1:] and not os.path.exists(fairseq_examples)
        os.symlink(os.apth.join("...", "examples"), fairseq_examples)

        package_data = {
            "fairseq": (
                get_files(fairseq_examples) + get_files(os.path.join("fairseq", "config"))
            )
        }
        do_setup(package_data)
    finally:
        if "build_ext" not in sys.argv[1:] and os.path.islink(fairseq_examples):
            os.unlink(fairseq_examples)