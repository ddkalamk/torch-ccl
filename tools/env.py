import os
import platform


IS_LINUX = (platform.system() == 'Linux')

BUILD_DIR = 'build'


def _find_dpcpp_home():
    r'''Finds the DPCPP install path.'''
    dpcpp_home = os.environ.get('DPCPP_HOME') or os.environ.get('DPCPP_ROOT')
    if dpcpp_home is None:
        # Guess #2
        try:
            which = 'which'
            with open(os.devnull, 'w') as devnull:
                dpcpp = subprocess.check_output([which, 'clang'],
                                               stderr=devnull).decode().rstrip('\r\n')
                dpcpp_home = os.path.dirname(os.path.dirname(dpcpp))
        except Exception:
            # Guess #3
            dpcpp_home = '/opt/intel/dpcpp'
            if not os.path.exists(dpcpp_home):
                dpcpp_home = None
    if dpcpp_home is None:
        print("No DPCPP is found")
    return dpcpp_home