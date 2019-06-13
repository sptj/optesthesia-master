import ctypes, sys
import os
import sys


# ========================================================================================================
def request_uac_to_run(func):
    def is_admin():
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except:
            return False

    if is_admin():
        # Code of your program here
        func()
    else:
        # Re-run the program with admin rights
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, __file__, None, 1)


# ========================================================================================================
from UAC import run_as_admin
@run_as_admin
def install_deps(wheel_dir = r'J:\Deploy_Requirement\python requirements'):
    file_names = list(os.listdir(wheel_dir))
    file_names.sort()
    # print(*file_names,sep='\n')
    for file_name in file_names:
        file_full_path = os.path.join(wheel_dir, file_name)
        command = 'pip install "{}" --no-index --find-link="{}"'.format(file_full_path, wheel_dir)
        os.system(command)
        break

if __name__ == '__main__':
    install_deps()
