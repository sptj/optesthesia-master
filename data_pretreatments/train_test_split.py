import os
from glob import glob
from pathlib import Path
from shutil import move

import numpy as np


def get_shared_stems(apcs, bpcs):
    get_stem_set = lambda pcs: set(map(lambda path_cls: path_cls.stem, pcs))
    astems_set = get_stem_set(apcs)
    bstems_set = get_stem_set(bpcs)

    diff = astems_set.symmetric_difference(bstems_set)
    if len(diff) != 0:
        for item in diff:
            print(item, 'ant or img file lost')
        raise ('please check the data over and over again')
    shares = astems_set.intersection(bstems_set)
    return shares


def get_partner_files(ant_filename_tuple, img_filename_tuple):
    get_pcs = lambda filenames: map(Path, filenames)

    ant_pcs = get_pcs(ant_filename_tuple)
    img_pcs = get_pcs(img_filename_tuple)

    shared_stems=get_shared_stems(ant_pcs,img_pcs)

    path_in_shared_stems=lambda pcls:pcls.stem in shared_stems

    parted_ant=filter(path_in_shared_stems,ant_pcs)
    parted_img=filter(path_in_shared_stems,img_pcs)

    return parted_ant,parted_img


def train_test_split(src_dir, train_dir, valid_dir=None, ratio=0.2):
    if train_dir is None:
        print('train_dir is None')
        return
    img_suf = '.png'
    ant_suf = '.xml'

    glob_taget_filenames = lambda suf: glob(os.path.join(src_dir, '*' + suf))
    get_bld_filename = lambda path: os.path.splitext(os.path.basename(path))[0]
    get_bld_set = lambda filenames: set(map(get_bld_filename, filenames))

    xml_files = glob_taget_filenames(ant_suf)
    img_files = glob_taget_filenames(img_suf)

    xml_set = get_bld_set(xml_files)
    img_set = get_bld_set(img_files)

    diff = xml_set.symmetric_difference(img_set)
    if len(diff) != 0:
        for item in diff:
            print(item, 'xml or img file lost')
        raise ('please check the data over and over again')

    common_set = xml_set.intersection(img_set)
    common_bld_filenames = sorted(list(common_set))


    common_file_num = len(common_bld_filenames)
    test_file_num = int(common_file_num * ratio)
    test_bld_filenames = np.random.choice(common_bld_filenames, test_file_num)

    mkdir = lambda x: os.makedirs(x) if os.path.isdir(x) is False else print(x, 'is already exists')

    mkdir(train_dir)
    mkdir(valid_dir)

    for i, bld_name in enumerate(common_bld_filenames):
        print(i, 'of', len(common_bld_filenames), 'files', 'has been processed')
        get_src_path = lambda bld_name, suf: os.path.join(src_dir, bld_name + suf)
        src_img_path = get_src_path(bld_name, img_suf)
        src_ant_path = get_src_path(bld_name, ant_suf)

        if bld_name in test_bld_filenames:
            dst_dir = valid_dir
        else:
            dst_dir = train_dir
        if os.path.abspath(dst_dir) is os.path.abspath(src_dir):
            continue

        get_dst_path = lambda bld_name, suf: os.path.join(dst_dir, bld_name + suf)

        dst_img_path = get_dst_path(bld_name, img_suf)
        dst_ant_path = get_dst_path(bld_name, ant_suf)

        move(src_img_path, dst_img_path)
        move(src_ant_path, dst_ant_path)


class call_magic_method_atp(object):
    # atp=call_magic_method_atp()
    # atp() <- __call__ function will be toggled at this scentence
    def __call__(self, *args, **kwargs):
        print('toggle')


def pathlib_atp(url=r'G:\debug'):
    filenames = glob(url + '\*.png')
    # maped_filenames=list(map(lambda x:Path(x).stem=xml,filenames))
    a = Path(filenames[0])
    a.with_name('1')
    print(a)


if __name__ == '__main__':
    # pathlib_atp()
    train_test_split(src_dir=r'I:\final_dataset_2019_0611\200',
                     train_dir=r'I:\final_dataset_2019_0611\temp\train_dir',
                     valid_dir=r'I:\final_dataset_2019_0611\temp\valid_dir')
