# Benchmarking Neural Network Robustness to Common Corruptions and Perturbations
# [Dan Hendrycks, Thomas Dietterich, ICLR2019]
# https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_cifar_c.py

from .corr_cifar10 import *

corruption_tuple = (gaussian_noise, shot_noise, impulse_noise, defocus_blur,
                    glass_blur, motion_blur, zoom_blur, snow, frost, fog,
                    brightness, contrast, elastic_transform, pixelate,
                    jpeg_compression, speckle_noise, gaussian_blur, spatter,
                    saturate)
corruption_dict = {corr_func.__name__: corr_func for corr_func in
                   corruption_tuple}

def corrupt_image_tensor(images, severity=1, corruption_name='gaussian_noise'):
    x = images
    # print(x.shape)
    x = x.permute(0,2,3,1)
    x = x * 255
    x = np.array(x)
    image_list = []
    for image in x:
        image_list.append(corruption_dict[corruption_name](image, severity=severity))
    x =torch.tensor(image_list)
    x = x / 255
    x = x.to(torch.float32)
    x = x.permute(0, 3, 1, 2)
    return x

def corrupt_image_cifar10(images, severity=1, corruption_name='gaussian_noise'):
    x = images
    x = np.array(x).astype(np.uint8)
    x = corruption_dict[corruption_name](x, severity=severity)
    x = x.astype(np.uint8)
    x = Image.fromarray(x)
    return x

def corrupt_image_imagenet(images, severity=1, corruption_name='gaussian_noise'):
    from core.corruptions.corr import gaussian_noise, shot_noise, impulse_noise, defocus_blur,\
                    glass_blur, motion_blur, zoom_blur, snow, frost, fog,\
                    brightness, contrast, elastic_transform, pixelate,\
                    jpeg_compression, speckle_noise, gaussian_blur, spatter,\
                    saturate
    corruption_tuple = (gaussian_noise, shot_noise, impulse_noise, defocus_blur,
                        glass_blur, motion_blur, zoom_blur, snow, frost, fog,
                        brightness, contrast, elastic_transform, pixelate,
                        jpeg_compression, speckle_noise, gaussian_blur, spatter,
                        saturate)
    corruption_dict = {corr_func.__name__: corr_func for corr_func in corruption_tuple}
    x = images
    x = np.array(x).astype(np.uint8)
    x = corruption_dict[corruption_name](x, severity=severity)
    x = x.astype(np.uint8)
    x = Image.fromarray(x)
    return x

# def corrupt_image_tensor_imagenet(images, severity=1, corruption_name='gaussian_noise'):
#     x = images
#     x = x.permute(0,2,3,1)
#     x = x * 255
#     x = np.array(x)
#     image_list = []
#     for image in x:
#         image_list.append(corruption_dict[corruption_name](image, severity=severity))
#     x =torch.tensor(image_list)
#     x = x / 255
#     x = x.to(torch.float32)
#     x = x.permute(0, 3, 1, 2)
#     return x


# for corruption of a batch of images tensors (batch_size, 3, 32, 32)
# corruption_name = 'gaussian_noise'
# sev = [1, 2, 3, 4, 5][1]
# corr = corrupt_image_tensor(images, corruption_name=corruption_name, severity=sev)


def get_corruption_names(subset='common'):
    if subset == 'common':
        return [f.__name__ for f in corruption_tuple[:15]]
    elif subset == 'validation':
        return [f.__name__ for f in corruption_tuple[15:]]
    elif subset == 'all':
        return [f.__name__ for f in corruption_tuple]
    elif subset == 'noise':
        return [f.__name__ for f in corruption_tuple[0:3]]
    elif subset == 'blur':
        return [f.__name__ for f in corruption_tuple[3:7]]
    elif subset == 'weather':
        return [f.__name__ for f in corruption_tuple[7:11]]
    elif subset == 'digital':
        return [f.__name__ for f in corruption_tuple[11:15]]
    else:
        raise ValueError("subset must be one of ['common', 'validation', 'all']")



# https://github.com/bethgelab/imagecorruptions/blob/master/imagecorruptions/__init__.py

# from .corruptions import *

# # for corruptions.py
# corruption_tuple = (gaussian_noise, shot_noise, impulse_noise, defocus_blur,
#                     glass_blur, motion_blur, zoom_blur, snow, frost, fog,
#                     brightness, contrast, elastic_transform, pixelate,
#                     jpeg_compression, speckle_noise, gaussian_blur, spatter,
#                     saturate)
# corruption_dict = {corr_func.__name__: corr_func for corr_func in
#                    corruption_tuple}


# def corrupt(image, severity=1, corruption_name=None, corruption_number=-1):
#     """This function returns a corrupted version of the given image.

#     Args:
#         image (numpy.ndarray):      image to corrupt; a numpy array in [0, 255], expected datatype is np.uint8
#                                     expected shape is either (height x width x channels) or (height x width);
#                                     width and height must be at least 32 pixels;
#                                     channels must be 1 or 3;
#         severity (int):             strength with which to corrupt the image; an integer in [1, 5]
#         corruption_name (str):      specifies which corruption function to call, must be one of
#                                         'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
#                                         'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
#                                         'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
#                                         'speckle_noise', 'gaussian_blur', 'spatter', 'saturate';
#                                     the last four are validation corruptions
#         corruption_number (int):    the position of the corruption_name in the above list; an integer in [0, 18];
#                                         useful for easy looping; 15, 16, 17, 18 are validation corruption numbers
#     Returns:
#         numpy.ndarray:              the image corrupted by a corruption function at the given severity; same shape as input
#     """

#     if not isinstance(image, np.ndarray):
#         raise AttributeError('Expecting type(image) to be numpy.ndarray')
#     if not (image.dtype.type is np.uint8):
#         raise AttributeError('Expecting image.dtype.type to be numpy.uint8')

#     if not (image.ndim in [2,3]):
#         raise AttributeError('Expecting image.shape to be either (height x width) or (height x width x channels)')
#     if image.ndim == 2:
#         image = np.stack((image,)*3, axis=-1)

#     height, width, channels = image.shape

#     if (height < 32 or width < 32):
#         raise AttributeError('Image width and height must be at least 32 pixels')

#     if not (channels in [1,3]):
#         raise AttributeError('Expecting image to have either 1 or 3 channels (last dimension)')

#     if channels == 1:
#         image = np.stack((np.squeeze(image),)*3, axis=-1)

#     if not severity in [1,2,3,4,5]:
#         raise AttributeError('Severity must be an integer in [1, 5]')

#     if not (corruption_name is None):
#         image_corrupted = corruption_dict[corruption_name](Image.fromarray(image),
#                                                        severity)
#     elif corruption_number != -1:
#         image_corrupted = corruption_tuple[corruption_number](Image.fromarray(image),
#                                                           severity)
#     else:
#         raise ValueError("Either corruption_name or corruption_number must be passed")

#     return np.uint8(image_corrupted)

# def get_corruption_names(subset='common'):
#     if subset == 'common':
#         return [f.__name__ for f in corruption_tuple[:15]]
#     elif subset == 'validation':
#         return [f.__name__ for f in corruption_tuple[15:]]
#     elif subset == 'all':
#         return [f.__name__ for f in corruption_tuple]
#     elif subset == 'noise':
#         return [f.__name__ for f in corruption_tuple[0:3]]
#     elif subset == 'blur':
#         return [f.__name__ for f in corruption_tuple[3:7]]
#     elif subset == 'weather':
#         return [f.__name__ for f in corruption_tuple[7:11]]
#     elif subset == 'digital':
#         return [f.__name__ for f in corruption_tuple[11:15]]
#     else:
#         raise ValueError("subset must be one of ['common', 'validation', 'all']")
