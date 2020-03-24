# USE_SYCL to turn on/off the Intel GPU support
import os
import pathlib
import shutil
import subprocess
from torch.utils.cpp_extension import include_paths, library_paths
from torch_ipex import include_paths as ipex_include_paths
from torch_ipex import library_paths as ipex_library_paths
from setuptools import setup, Extension, distutils, find_packages
from setuptools.command.build_ext import build_ext
from distutils.command.clean import clean


def _create_build_env():
    my_env = os.environ.copy()
    return my_env


def defines(args, **kwargs):
    "Adds definitions to a cmake argument list."
    for key, value in sorted(kwargs.items()):
        if value is not None:
            args.append('-D{}={}'.format(key, value))


class CMakeExtension(Extension):
    def __init__(self, name, cmake):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, [])
        self.cmake_file = cmake


class BuildCMakeExt(build_ext):
    """
    Builds using cmake instead of the python setuptools implicit build
    """
    def run(self):
        """
        Perform build_cmake before doing the 'normal' stuff
        """
        for extension in self.extensions:
            if isinstance(extension, CMakeExtension):
                self.build_cmake(extension)

        # super().run()

    def build_cmake(self, extension: CMakeExtension):
        """
        The steps required to build the extension
        """
        self.announce("Preparing the build environment", level=3)

        build_dir = pathlib.Path(self.build_temp)
        extension_path = pathlib.Path(self.get_ext_fullpath(extension.name))
        cmake_path = pathlib.Path(extension.cmake_file)

        build_dir.mkdir(parents=True, exist_ok=True)
        extension_path.parent.absolute().mkdir(parents=True, exist_ok=True)

        print("johnlu extensiopn path ", extension_path )

        # Now that the necessary directories are created, build
        self.announce("Configuring cmake project", level=3)

        my_env = _create_build_env()

        # Store build options that are directly stored in environment variables
        build_options = {
            # The default value cannot be easily obtained in CMakeLists.txt. We set it here.
            # 'CMAKE_PREFIX_PATH': distutils.sysconfig.get_python_lib()
        }

        for var, val in my_env.items():
            if var.startswith(('BUILD_', 'USE_', 'CMAKE_')):
                build_options[var] = val

        config = 'Debug' if self.debug else 'Release'

        def convert_cmake_dirs(paths):
            def converttostr(input_seq, seperator):
                # Join all the strings in list
                final_str = seperator.join(input_seq)
                return final_str
            try:
                return converttostr(paths, ";")
            except:
                return paths


        cmake_args = [
            '-DCMAKE_BUILD_TYPE=' + config,
            '-DCMAKE_C_COMPILER=clang',
            '-DCMAKE_CXX_COMPILER=clang++',
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + str(extension_path.parent.absolute()) + '/occl/lib',
            '-DPYTHON_INCLUDE_DIRS=' + str(distutils.sysconfig.get_python_inc()),
            '-DPYTORCH_INCLUDE_DIRS=\'' + convert_cmake_dirs(include_paths()) + "'",
            '-DPYTORCH_LIBRARY_DIRS=\'' + convert_cmake_dirs(library_paths()) + "'",
            '-DIPEX_INCLUDE_DIRS=\'' + convert_cmake_dirs(ipex_include_paths()) + "'",
            '-DIPEX_LIBRARY_DIRS=\'' + convert_cmake_dirs(ipex_library_paths()) + "'",
            '-DLIB_NAME=' + str(extension_path.name),
            ]

        defines(cmake_args, **build_options)

        cmake_cmd = ['cmake', str(cmake_path.absolute()), '-B'+self.build_temp] + cmake_args

        print("cmake cmd -> ",  " ".join(cmake_cmd))
        # self.spawn(cmake_cmd)

        # Build the target
        self.announce("Building binaries", level=3)

        build_args = [
            '--config', config,
            '--', '-j4'
        ]

        self.spawn(['cmake', '--build', self.build_temp] + build_args)

    # def build_extensions(self):
    #     pass


class Clean(clean):
    def run(self):
        import glob
        import re
        with open('.gitignore', 'r') as f:
            ignores = f.read()
            pat = re.compile(r'^#( BEGIN NOT-CLEAN-FILES )?')
            for wildcard in filter(None, ignores.split('\n')):
                match = pat.match(wildcard)
                if match:
                    if match.group(1):
                        # Marker is found and stop reading .gitignore.
                        break
                    # Ignore lines which begin with '#'.
                else:
                    for filename in glob.glob(wildcard):
                        try:
                            os.remove(filename)
                        except OSError:
                            shutil.rmtree(filename, ignore_errors=True)

        clean.run(self)

if __name__ == '__main__':
  setup(
      name='occl',
      version='0.1',
      ext_modules=[CMakeExtension(name="liboccl", cmake="./CMakeLists.txt")],
      packages=['occl'],
      cmdclass={
          'build_ext': BuildCMakeExt,
          'clean': Clean,
      }
  )
