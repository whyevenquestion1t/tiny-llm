from pathlib import Path
import shutil
from mlx import extension
from setuptools import Distribution

if __name__ == "__main__":
    src_dir = Path(__file__).parent
    distribution = Distribution(
        {
            "name": "tiny_llm_ext_ref",
            "ext_modules": [extension.CMakeExtension("tiny_llm_ext_ref._ext")],
        }
    )
    cmd = extension.CMakeBuild(distribution)
    cmd.initialize_options()
    cmd.build_temp = Path("build")
    cmd.build_lib = Path("build") / "lib"
    cmd.inplace = False  # we do the copy by ourselves
    cmd.ensure_finalized()
    cmd.run()
    for output in cmd.get_outputs():
        output = Path(output)
        relative_extension = src_dir / output.relative_to(cmd.build_lib)
        shutil.copyfile(output, relative_extension)
        print(f"Copied {output} to {relative_extension}")
