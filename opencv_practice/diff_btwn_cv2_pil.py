import numpy
from PIL import Image
import cv2
def get_info_template(image):
    if isinstance(image, numpy.ndarray):
        pass
    elif isinstance(image, Image.Image):
        pass
    else:
        raise Exception('unknown image instance')

def get_image_whc(image):
    if isinstance(image, numpy.ndarray):
        h, w, c = image.shape
    elif isinstance(image, Image.Image):
        w, h = image.size
        image_mode_dict = {'1': 1, 'L': 1, 'P': 1, 'RGB': 3, 'RGBA': 4,
                           'CMYK': 4, 'YCbCr': 3, 'I': 1, 'F': 1}
        c = image_mode_dict[image.mode]
    else:
        raise Exception('unknown image instance')
    return (w, h, c)
def image_crop(image,x1y1x2y2_tuple):
    if isinstance(image, numpy.ndarray):
        pass
    elif isinstance(image, Image.Image):
        image.crop(x1y1x2y2_tuple)
    else:
        raise Exception('unknown image instance')

def get_image_whc_test():
    path = '''
    C:\Windows\Web\Wallpaper\Windows\img0.jpg
    '''.strip()
    a = Image.open(path)
    b = cv2.imread(path)
    print(get_image_whc_tuple(a))
    print(get_image_whc_tuple(b))

def get_image_info(image):
    if isinstance(image, numpy.ndarray):
        w,h,c=get_image_whc_tuple(image)
        if c==1:
            im_mode_s= '1'
        else:
            im_mode_s ='RGB'
    elif isinstance(image, Image.Image):
        im_mode_s = image.info
    else:
        raise Exception('unknown image instance')
    return im_mode_s

def reside_image(image,wh_tuple):
    if isinstance(image, numpy.ndarray):
        pass
    elif isinstance(image, Image.Image):
        pass
    else:
        raise Exception('unknown image instance')

def get_pixel(image,wh_tuple):
    w,h=wh_tuple
    if isinstance(image, numpy.ndarray):
        # caution: order is [B, G, R]
        pixel_tuple=image[h][w]
    elif isinstance(image, Image.Image):
        # caution: order is [R, G, B]
        pixel_tuple=image.getpixel((w,h))
    else:
        raise Exception('unknown image instance')
    return pixel_tuple

def channel_split(image):
    if isinstance(image, numpy.ndarray):
        b,g,r=cv2.split(image)
    elif isinstance(image, Image.Image):
        pass
    else:
        raise Exception('unknown image instance')
def channel_merge(bgr_tuple):
    b,g,r=bgr_tuple
    if isinstance(b, numpy.ndarray):
        b,g,r=cv2.merge(( b,g,r))
    elif isinstance(b, Image.Image):
        pass
    else:
        raise Exception('unknown image instance')

def test_ordinate():
    path = '''
    C:\Windows\Web\Wallpaper\Windows\img0.jpg
    '''.strip()

    a = Image.open(path)
    b = cv2.imread(path)
    print(a.getpixel((150,456)),b[456,150])
    print(a.getpixel((456,150)),b[150,456])

test_ordinate()

