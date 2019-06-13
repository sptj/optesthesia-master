import os
from glob import glob

def find_test_dataset(img_dir,video_dir):
    img_suffix='.jpg'
    vid_suffix='.mp4'
    img_filenames=glob(os.path.join(img_dir,'*'+img_suffix))
    vid_filenames=glob(os.path.join(video_dir,'*'+vid_suffix))

    img_filenames=map(lambda x:os.path.split(x)[-1],img_filenames)
    vid_filenames=map(lambda x:os.path.split(x)[-1],vid_filenames)

    img_set=set()
    vid_set=set()
    for img_filename in img_filenames:
        img_set.add(img_filename.split('_frame_')[0])
    for vid_filename in vid_filenames:
        vid_set.add(vid_filename.split('.')[0])
    test_set=vid_set-img_set
    test_full_names=list(map(lambda x:os.path.join(video_dir,x+vid_suffix),list(test_set)))
    print(img_set)
    return test_full_names
def move_file(file,target_dir):
    from shutil import move
    transed_file=os.path.join(target_dir,os.path.split(file)[-1])
    move(file,transed_file)
if __name__ == '__main__':
    test_files=find_test_dataset(r'D:\drone_image_and_annotation\JPEGImages',
                            r'D:\drone_video')
    test_video_dir=r"D:\test_drone_video"
    for test_file in test_files:
        print(test_file)
        move_file(test_file,test_video_dir)












