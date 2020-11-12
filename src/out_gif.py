import numpy as np
import cv2 as cv
import imageio
import os
from skimage.transform import resize

def output_gif(input_path, output_path, gif_name):
    '''
    该函数生成gif图像便于展示
    :param input_path: path of the input inages series
    :param output_path: gif output path
    :param gif_name: name of the gif
    :return:
    '''

    outfilename = os.path.join(output_path, '{}.gif'.format(gif_name))
    frames = []
    paths = os.listdir(input_path)
    paths_sort = sorted(paths, key=lambda x: int((os.path.splitext(x))[0]))
    for path in paths_sort:
        fullpath = os.path.join(input_path, path)
        im = imageio.imread(fullpath)
        frames.append(im)
    imageio.mimsave(outfilename, frames, 'GIF', duration=0.5)


def data_prepare():
    path = r"C:\Users\Administrator\Desktop\gif\37"
    path_out = r"C:\Users\Administrator\Desktop\gif\out"
    # output_gif(path, path_out, "test")
    dires = os.listdir(path)
    n_slices = int(len(dires) / 3)
    subject = 37
    start_index = 7
    out_path_subject = os.path.join(path_out, str(subject))
    if not os.path.exists(out_path_subject):
        os.mkdir(out_path_subject)
    for i in range(start_index, n_slices + start_index):
        orig = cv.imread(os.path.join(path, "{}_{}.png".format(subject, i)))

        gt = cv.imread(os.path.join(path, "{}_{}label.png".format(subject, i)))
        pre = cv.imread(os.path.join(path, "{}_{}pre.png".format(subject, i)))
        img = resize(orig, gt.shape, order=3, mode="constant", cval=0, clip=True, preserve_range=True)
        img_cat = np.concatenate((img, gt, pre), axis=1)
        cv.imwrite(os.path.join(out_path_subject, "{}.png".format(i)), img_cat)


if __name__  == "__main__":
    path_in = r"C:\Users\Administrator\Desktop\gif\out"
    files = os.listdir(path_in)
    for file in files:
        subject_path = os.path.join(path_in, file)
        output_gif(subject_path, subject_path, "test"+file)
