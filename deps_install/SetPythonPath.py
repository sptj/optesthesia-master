# coding:GB2312
from UAC import run_as_admin
import os
@run_as_admin
def exe_cmd_with_return(cmd):
    r=os.popen(cmd)
    text=r.read()
    r.close
    return text
if __name__ == '__main__':
    a=exe_cmd_with_return('path')
    print(a)




