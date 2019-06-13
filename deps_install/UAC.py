import ctypes
import sys
"""
def decorator(func):
    def wapper(*args,**kwargs):
        for frame in video_frame_gen(src_path=src_file_path):
            func(frame)
    return wapper
def process_image_soucess(src_file_path):
    def decorator(func):
        def wapper(*args,**kwargs):
            for frame in video_frame_gen(src_path=src_file_path):
                func(frame)
        return wapper
    return decorator
"""
def run_as_admin(func):
    def wapper(*args,**kwargs):
        def is_admin():
            try:
                return ctypes.windll.shell32.IsUserAnAdmin()
            except:
                return False
        if is_admin():
            return func(*args, **kwargs)
        else:
            # Re-run the program with admin rights
            ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, __file__, None, 1)
            return None
    return wapper

