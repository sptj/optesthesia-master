import numpy
from PIL import Image
import torch
import matplotlib.pyplot


def resize(img, boxes, size, max_size=1024):
    '''Resize the input PIL image to the given size.

    Args:
      img: (PIL.Image) image to be resized.
      boxes: (tensor) object boxes, sized [#ojb,4].
      size: (tuple or int)
        - if is tuple, resize image to the size.
        - if is int, resize the shorter side to the size while maintaining the aspect ratio.
      max_size: (int) when size is int, limit the image longer size to max_size.
                This is essential to limit the usage of GPU memory.
    Returns:
      img: (PIL.Image) resized image.
      boxes: (tensor) resized boxes.
    '''
    w, h = img.size
    if isinstance(size, int):
        size_min = min(w, h)
        size_max = max(w, h)
        sw = sh = float(size) / size_min
        if sw * size_max > max_size:
            sw = sh = float(max_size) / size_max
        ow = int(w * sw + 0.5)
        oh = int(h * sh + 0.5)
    else:
        ow, oh = size
        sw = float(ow) / w
        sh = float(oh) / h
    return img.resize((ow, oh), Image.BILINEAR),  boxes * torch.Tensor([sw, sh, sw, sh])
def random_crop_ext(img, box,dst_wh_tuple):
    def feasible_region_gen(cxcytwth_tuple, src_wh_tuple, dst_wh_tuple):
        cx, cy, w, h = cxcytwth_tuple
        src_w, src_h = src_wh_tuple
        dst_w, dst_h = dst_wh_tuple
        region1 = (int(dst_w / 2),
                   int(dst_h / 2),
                   src_w - int(dst_w / 2),
                   src_h - int(dst_h / 2))
        region2 = (cx - int(int(dst_w / 2) - int(w / 2)),
                   cy - int(int(dst_h / 2) - int(h / 2)),
                   cx + int(int(dst_w / 2) - int(w / 2)),
                   cy + int(int(dst_h / 2) - int(h / 2)))
        xx1 = max(region1[0], region2[0])
        yy1 = max(region1[1], region2[1])
        xx2 = min(region1[2], region2[2])
        yy2 = min(region1[3], region2[3])
        import numpy as np
        a = np.random.choice(range(int(xx1), int(xx2) + 1))
        b = np.random.choice(range(int(yy1), int(yy2) + 1))
        return (a - int(dst_w / 2), b - int(dst_h / 2), a - int(dst_w / 2) + dst_w, b - int(dst_h / 2) + dst_w)

    src_w,src_h=get_image_wh_tuple(img)

    x1, y1, x2, y2 = box
    cx, cy, w, h = (x1 + x2) // 2, (y1 + y2) // 2, x2 - x1, y2 - y1
    dst_w ,dst_h=dst_wh_tuple
    if w > src_w or h > src_h:
        # this error should move to check step in case program error at the end of runnning
        raise ('error ')
    if w < dst_w and h < dst_h:
        feasible_region=feasible_region_gen(cxcytwth_tuple=(cx, cy, w, h ), src_wh_tuple=(src_w,src_h), dst_wh_tuple=dst_wh_tuple)
        img = img.crop(feasible_region)
        box -= torch.Tensor([feasible_region[0], feasible_region[1], feasible_region[0], feasible_region[1]])
    else:
        scale = max(w / dst_w, h / dst_h)
        img,box = resize(img, box, (src_w / scale, src_h / scale))
    return img, box