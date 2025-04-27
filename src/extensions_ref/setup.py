from setuptools import setup

from mlx import extension

if __name__ == "__main__":
    setup(
        name="tiny_llm_ext_ref",
        ext_modules=[extension.CMakeExtension("tiny_llm_ext_ref._ext")],
        cmdclass={"build_ext": extension.CMakeBuild},
        packages=["tiny_llm_ext_ref"],
        zip_safe=False,
    )
