from __future__ import annotations

from pathlib import Path
from setuptools import Distribution
from mlx import extension
import shutil


def build():
    src_dir = Path(__file__).parent.joinpath("src").joinpath("extensions_ref")
    ext_modules = [extension.CMakeExtension("tiny_llm_ext_ref._ext", src_dir)]
    distribution = Distribution(
        {
            "name": "tiny_llm_ext_ref",
            "ext_modules": ext_modules,
        }
    )
    cmd = extension.CMakeBuild(distribution)
    cmd.ensure_finalized()
    cmd.run()
    for output in cmd.get_outputs():
        output = Path(output)
        relative_extension = src_dir / output.relative_to(cmd.build_lib)
        shutil.copyfile(output, relative_extension)


if __name__ == "__main__":
    build()
