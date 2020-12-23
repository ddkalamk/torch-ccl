# DEBUG build with debug

# USE_SYSTEM_ONECCL -  build the torch_ccl library support the sycl

import os
import pathlib
import shutil
import multiprocessing
from subprocess import check_call, check_output
from tools.env import BUILD_DIR

from torch.utils.cpp_extension import include_paths, library_paths
from setuptools import setup, Extension, distutils, find_packages
from setuptools.command.build_ext import build_ext
from distutils.command.clean import clean
from distutils.version import LooseVersion

def check_env_flag(name, default=''):
    return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']


def _get_complier(runtime):
  return "gcc", "g++"


# hotpatch environment variable 'CMAKE_BUILD_TYPE'. 'CMAKE_BUILD_TYPE' always prevails over DEBUG or REL_WITH_DEB_INFO.
if 'CMAKE_BUILD_TYPE' not in os.environ:
    if check_env_flag('DEBUG'):
        os.environ['CMAKE_BUILD_TYPE'] = 'Debug'
    elif check_env_flag('REL_WITH_DEB_INFO'):
        os.environ['CMAKE_BUILD_TYPE'] = 'RelWithDebInfo'
    else:
        os.environ['CMAKE_BUILD_TYPE'] = 'Release'

# Constant known variables used throughout this file
cwd = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(cwd, "torch_ccl", "lib")

package_name = os.getenv('OCCL_PACKAGE_NAME', 'torch-ccl')
version = open('version.txt', 'r').read().strip()
sha = 'Unknown'

try:
    sha = check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
except Exception:
    pass

if os.getenv('PYTORCH_BUILD_VERSION'):
    assert os.getenv('PYTORCH_BUILD_NUMBER') is not None
    build_number = int(os.getenv('PYTORCH_BUILD_NUMBER'))
    version = os.getenv('PYTORCH_BUILD_VERSION')
    if build_number > 1:
        version += '.post' + str(build_number)
elif sha != 'Unknown':
    version += '+' + sha[:7]
print("Building wheel {}-{}".format(package_name, version))


# all the work we need to do _before_ setup runs
def build_deps():
    print('-- Building version ' + version)

    def check_file(f):
        if not os.path.exists(f):
            print("Could not find {}".format(f))
            print("Did you run 'git submodule update --init --recursive'?")
            sys.exit(1)

    version_path = os.path.join(cwd, 'torch_ccl', 'version.py')
    with open(version_path, 'w') as f:
        f.write("__version__ = '{}'\n".format(version))
        f.write("build_type = '{}'\n".format(os.environ['CMAKE_BUILD_TYPE']))
        f.write("git_version = {}\n".format(repr(sha)))


def _create_build_env():
    my_env = os.environ.copy()
    return my_env


def which(thefile):
    path = os.environ.get("PATH", os.defpath).split(os.pathsep)
    for d in path:
        fname = os.path.join(d, thefile)
        fnames = [fname]
        for name in fnames:
            if os.access(name, os.F_OK | os.X_OK) and not os.path.isdir(name):
                return name
    return None


def defines(args, **kwargs):
    "Adds definitions to a cmake argument list."
    for key, value in sorted(kwargs.items()):
        if value is not None:
            args.append('-D{}={}'.format(key, value))


class CMakeExtension(Extension):
    """CMake extension"""


    def __init__(self, name, cmake_file):
        super().__init__(name, [])
        self.build_dir = None
        self.cmake_file = cmake_file
        self.debug = True
        self._cmake_command = CMakeExtension._get_cmake_command()
        self.runtime ='native'

    @staticmethod
    def _get_version(cmd):
        """Returns cmake version."""

        for line in check_output([cmd, '--version']).decode('utf-8').split('\n'):
            if 'version' in line:
                return LooseVersion(line.strip().split(' ')[2])
        raise RuntimeError('no version found')

    @staticmethod
    def _get_cmake_command():
        """Returns cmake command."""

        cmake_command = which('cmake')
        cmake3 = which('cmake3')
        if cmake3 is not None:
            cmake = which('cmake')
            if cmake is not None:
                bare_version = CMakeExtension._get_version(cmake)
                if (bare_version < LooseVersion("3.5.0") and
                        CMakeExtension._get_version(cmake3) > bare_version):
                    cmake_command = 'cmake3'
        return cmake_command


    @staticmethod
    def defines(args, **kwargs):
        "Adds definitions to a cmake argument list."
        for key, value in sorted(kwargs.items()):
            if value is not None:
                args.append('-D{}={}'.format(key, value))


    @property
    def _cmake_cache_file(self):
        r"""Returns the path to CMakeCache.txt.

        Returns:
          string: The path to CMakeCache.txt.
        """
        return os.path.join(self.build_dir, 'CMakeCache.txt')


    def run(self, args, env):
        """Executes cmake with arguments and an environment."""
        command = [self._cmake_command] + args
        print(' '.join(command))
        check_call(command, cwd=self.build_dir, env=env)


    def generate(self, my_env, build_dir, install_prefix):
        """Runs cmake to generate native build files."""

        def convert_cmake_dirs(paths):
            def converttostr(input_seq, seperator):
                # Join all the strings in list
                final_str = seperator.join(input_seq)
                return final_str
            try:
                return converttostr(paths, ";")
            except:
                return paths
        self.build_dir = build_dir
        cmake_args = []
        # Store build options that are directly stored in environment variables
        build_options = {
            # The default value cannot be easily obtained in CMakeLists.txt. We set it here.
            'CMAKE_INSTALL_PREFIX': install_prefix,
            'PYTHON_INCLUDE_DIRS': str(distutils.sysconfig.get_python_inc()),
            'PYTORCH_INCLUDE_DIRS': convert_cmake_dirs(include_paths()),
            'PYTORCH_LIBRARY_DIRS': convert_cmake_dirs(library_paths()),
        }

        for var, val in my_env.items():
            if var.startswith(('BUILD_', 'USE_', 'CMAKE_')):
                build_options[var] = val

        cc, cxx = _get_complier(self.runtime)
        defines(cmake_args, CMAKE_C_COMPILER=cc)
        defines(cmake_args, CMAKE_CXX_COMPILER=cxx)
        defines(cmake_args, **build_options)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cmake_args.append(base_dir)
        print(' '.join(cmake_args))
        if not os.path.exists(self._cmake_cache_file):
            # Everything's in place. Do not rerun.
            self.run(cmake_args, env=my_env)


class BuildCMakeExt(build_ext):
    """
    Builds using cmake instead of the python setuptools implicit build
    """
    def run(self):
        """
        Perform build_cmake before doing the 'normal' stuff
        """
        dpcpp_exts = [ext for ext in self.extensions if isinstance(ext, CMakeExtension)]
        for ext in dpcpp_exts:
            self.build_cmake(ext)
        self.extensions = [ext for ext in self.extensions if not isinstance(ext, CMakeExtension)]
        super(BuildCMakeExt, self).run()

    def build_cmake(self, extension: CMakeExtension):
        """
        The steps required to build the extension
        """
        # Show the build compute runtime
        self.announce("Build with runtime {}".format(extension.runtime), level=3)

        self.announce("Preparing the build environment", level=3)
        build_dir = pathlib.Path('.'.join([self.build_temp, extension.name]))

        build_dir.mkdir(parents=True, exist_ok=True)
        cwd = os.path.dirname(os.path.abspath(__file__))
        install_dir = os.path.join(cwd, "torch_ccl")

        # Now that the necessary directories are created, build
        self.announce("Configuring cmake project", level=3)

        my_env = _create_build_env()

        extension.generate(my_env, build_dir, install_dir)

        # Build the target
        self.announce("Building binaries", level=3)

        max_jobs = os.getenv('MAX_JOBS', str(multiprocessing.cpu_count()))
        build_args = ['-j', max_jobs]

        check_call(['make', 'torch_ccl'] + build_args, cwd=str(build_dir), env=my_env)
        check_call(['make', 'install'], cwd=str(build_dir), env=my_env)


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


def get_python_c_module():
    main_compile_args = []
    main_libraries = ['torch_ccl']
    main_link_args = []
    main_sources = ["torch_ccl/csrc/_C.cpp"]
    cwd = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(cwd, "torch_ccl", "lib")
    library_dirs = [lib_path]
    include_path = os.path.join(cwd, "src")
    include_dirs = include_paths()
    include_dirs.append(include_path)
    extra_link_args = []
    extra_compile_args = [
        '-Wall',
        '-Wextra',
        '-Wno-strict-overflow',
        '-Wno-unused-parameter',
        '-Wno-missing-field-initializers',
        '-Wno-write-strings',
        '-Wno-unknown-pragmas',
        # This is required for Python 2 declarations that are deprecated in 3.
        '-Wno-deprecated-declarations',
        # Python 2.6 requires -fno-strict-aliasing, see
        # http://legacy.python.org/dev/peps/pep-3123/
        # We also depend on it in our code (even Python 3).
        '-fno-strict-aliasing',
        # Clang has an unfixed bug leading to spurious missing
        # braces warnings, see
        # https://bugs.llvm.org/show_bug.cgi?id=21629
        '-Wno-missing-braces',
    ]

    def make_relative_rpath(path):
        return '-Wl,-rpath,$ORIGIN/' + path

    c_module = Extension("torch_ccl._C",
                  libraries=main_libraries,
                  sources=main_sources,
                  language='c',
                  extra_compile_args=main_compile_args + extra_compile_args,
                  include_dirs=include_dirs,
                  library_dirs=library_dirs,
                  extra_link_args=extra_link_args + main_link_args + [make_relative_rpath('lib')])

    return c_module

if __name__ == '__main__':
  build_deps()
  c_module = get_python_c_module()
  modules = [c_module, CMakeExtension("libtorch_ccl", "./CMakeLists.txt")]
  setup(
      name='torch_ccl',
      version=version,
      ext_modules=modules,
      packages=['torch_ccl'],
      package_data={
          'torch_ccl': [
              '*.py',
              '*/*.h',
              '*/*.hpp',
              'lib/*.so*',
              ]},
      cmdclass={
          'build_ext': BuildCMakeExt,
          'clean': Clean,
      }
  )
