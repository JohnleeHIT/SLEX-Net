import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2 as cv
import os
from torch.autograd import Variable
import colorsys
from skimage.measure import find_contours
from matplotlib.patches import Polygon
import random
import imageio


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


def mask_cross_entropy(y_true, y_pred, alpha, mask=None):
    '''
    calculate the croos-entropy loss function with mask
    :param y_true: tensor, size=N*D*H*W
    :param y_pred: tensor, size=N*class_n*D*H*W
    :param mask: tensor, size=weights on each voxel  N*D*H*W
    :return: voxel weighted cross entropy loss
    '''
    log_prob = F.log_softmax(y_pred, dim=1)
    prob = F.softmax(y_pred, dim=1)
    shape = y_pred.shape
    y_true_tensor = onehot(y_true, shape[1])
    # loss = torch.cuda.FloatTensor([0])
    loss = 0
    for i in range(shape[1]):
        y_task = y_true_tensor[:,i,...]
        y_prob = log_prob[:, i, ...]
        if torch.is_tensor(mask):
            loss += torch.mean(-y_task * y_prob * mask)*alpha[i]
        else:
            loss += torch.mean(-y_task * y_prob)*alpha[i]
    return loss


def mask_cross_entropy_3d(y_true, y_pred, alpha, mask=None):
    '''
    calculate the croos-entropy loss function with mask
    :param y_true: tensor, size=N*slice*H*W
    :param y_pred: tensor, size=N*slice*H*W
    :param mask: tensor, size=weights on each voxel  N*D*H*W
    :return: voxel weighted cross entropy loss
    '''
    y_true_0 = y_true[:,0, ...]
    y_true_1 = y_true[:, 1, ...]
    y_true_2 = y_true[:, 2, ...]

    y_pred_0 = y_pred[:,0:2, ...]
    y_pred_1 = y_pred[:, 2:4, ...]
    y_pred_2 = y_pred[:, 4:6, ...]

    loss0 = mask_cross_entropy(y_true_0, y_pred_0, alpha)
    loss1 = mask_cross_entropy(y_true_1, y_pred_1, alpha)
    loss2 = mask_cross_entropy(y_true_2, y_pred_2, alpha)

    loss = loss1+loss2+loss0

    return loss


def mask_focal_loss(y_true, y_pred, alpha, gamma=0, mask=None):
    '''
    calculate the croos-entropy loss function with mask
    :param y_true: tensor, size=N*D*H*W
    :param y_pred: tensor, size=N*class_n*D*H*W
    :param mask: tensor, size=weights on each voxel  N*D*H*W
    :return: voxel weighted cross entropy loss
    '''
    log_prob = F.log_softmax(y_pred, dim=1)
    prob = F.softmax(y_pred, dim=1)
    shape = y_pred.shape
    y_true_tensor = onehot(y_true, shape[1])
    # loss = torch.cuda.FloatTensor([0])
    loss = 0
    assert isinstance(alpha, list)
    alpha = torch.Tensor(alpha)
    # gamma = torch.Tensor(gamma)
    for i in range(shape[1]):
        y_task = y_true_tensor[:,i,...]
        y_prob = log_prob[:, i, ...]
        focal_weight = (1-prob)**gamma
        if torch.is_tensor(mask):
            loss += torch.mean(-y_task * y_prob * mask*focal_weight)*alpha[i]
        else:
            loss += torch.mean(-y_task * y_prob*focal_weight)*alpha[i]
    return loss


def onehot(input, class_n):
    '''
    onehot for pytorch
    :param input: N*H*W
    :param class_n:
    :return:N*n_class*H*W
    '''
    shape = input.shape
    onehot = torch.zeros((class_n,)+shape).cuda()
    for i in range(class_n):
        onehot[i, ...] = (input == i)
    onehot_trans = onehot.permute(1,0,2,3)
    return onehot_trans

def onehot3d(input, class_n):
    '''
    onehot for pytorch
    :param input: N*H*W*D
    :param class_n:
    :return:N*n_class*H*W*D
    '''
    shape = input.shape
    onehot = torch.zeros((class_n,)+shape).cuda()
    for i in range(class_n):
        onehot[i, ...] = (input == i)
    onehot_trans = onehot.permute(1,0,2,3,4)
    return onehot_trans


# one-hot encoding method(efficient)
def onehot_encoding(array, class_num):
    '''
    the function turn a regular array to a one-hot representation form
    :param array: input array
    :param class_num: number of classes
    :return: one-hot encoding of array
    '''
    label_one_hot = np.zeros(array.shape + (class_num,), dtype="int16")
    for k in range(class_num):
        label_one_hot[..., k] = (array == k)
    return label_one_hot



def filter3D(image, kernel):
    '''
    do 3D convolution to an input ndarray
    :param image: 3d volume data W*H*D
    :param kernel: 2d filter
    :return: convolved result
    '''
    shape = image.shape
    convled_list = []
    for i in range(shape[0]):
        convolve = cv.filter2D(image[i, ...], -1, kernel)
        convled_list.append(convolve)
    out = np.array(convled_list)
    return out


def Canny3D(image, thresh1, thresh2):
    '''
    apply canny algorithm in a 3D fashion
    :param image:
    :param thresh1:
    :param thresh2:
    :return:
    '''
    shape = image.shape
    convled_list = []
    for i in range(shape[0]):
        convolve = cv.Canny(image[i, ...], threshold1=thresh1, threshold2=thresh2)
        convled_list.append(convolve)
    out = np.array(convled_list)
    out = out/255
    return out


def normalize(data):
    max = np.max(data)
    min = np.min(data)
    normalized = (data-min)/(max-min+0.00001)
    return normalized


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class Evaluation(object):
    def __init__(self):
        pass

    # save 3d volume as slices
    def save_slice_img(self, volume_path, output_path):
        file_name = os.path.basename(volume_path)
        output_dir  = os.path.join(output_path, file_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            pass
        input_volume = nib.load(volume_path).get_data()
        # mapping to 0-1
        vol_max = np.max(input_volume)
        vol_min = np.min(input_volume)
        input_unit = (input_volume-vol_min)/(vol_max - vol_min)
        width, height, depth= input_unit.shape
        for i in range(0, depth):
            slice_path = os.path.join(output_dir, str(i)+'.png')
            img_i = input_unit[:, :, i]
            # normalize to 0-255
            img_i = (img_i*255).astype('uint8')
            # cv.imwrite(slice_path, img_i)
        return input_unit

    def save_slice_img_label(self, img_volume, pre_volume, gt_volume,
                             output_path, file_name, show_mask=False, show_gt = False):
        assert img_volume.shape == pre_volume.shape
        if show_gt:
            assert img_volume.shape == gt_volume.shape
        width, height, depth = img_volume.shape
        # gray value mapping   from MRI value to pixel value(0-255)
        volume_max = np.max(img_volume)
        volume_min = np.min(img_volume)
        volum_mapped = (img_volume-volume_min)/(volume_max-volume_min)
        volum_mapped = (255*volum_mapped).astype('uint8')
        # construct a directory for each volume to save slices
        dir_volume = os.path.join(output_path, file_name)
        if not os.path.exists(dir_volume):
            os.makedirs(dir_volume)
        else:
            pass
        for i in range(depth):
            img_slice = volum_mapped[:, :, i]
            pre_slice = pre_volume[:, :, i]
            if show_gt:
                gt_slice = gt_volume[:, :, i]
            else:
                gt_slice = []
            self.save_contour_label(img=img_slice, pre=pre_slice, gt=gt_slice,
                                    save_path=dir_volume, file_name=i,show_mask=show_mask,show_gt=show_gt)

    def apply_mask(self, image, mask, color, alpha=0.5):
        """Apply the given mask to the image.
        """
        for c in range(image.shape[-1]):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image

    def random_colors(self, N, bright=True):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

    def save_contour_label(self, img, pre, save_path='',color="red", file_name=None, show_mask=False):
        # single channel to multi-channel
        img = np.expand_dims(img, axis=-1)
        img = np.tile(img, (1, 1, 3))
        height, width = img.shape[:2]
        _, ax = plt.subplots(1, figsize=(height, width))

        # Generate random colors
        # colors = self.random_colors(4)
        # Prediction result is illustrated as red and the groundtruth is illustrated as blue
        colors = [[1.0, 0, 0], [0, 0, 1.0]]
        if color == "red":
            color_used = colors[0]
        elif color == "blue":
            color_used = colors[1]
        else:
            raise Exception("unkown color")
        # Show area outside image boundaries.

        # ax.set_ylim(height + 10, -10)
        # ax.set_xlim(-10, width + 10)
        ax.set_ylim(height + 0, 0)
        ax.set_xlim(0, width + 0)
        ax.axis('off')
        # ax.set_title("volume mask")
        masked_image = img.astype(np.uint32).copy()

        if show_mask:
            masked_image = self.apply_mask(masked_image, pre, color_used)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask_pre = np.zeros(
            (pre.shape[0] + 2, pre.shape[1] + 2), dtype=np.uint8)
        padded_mask_pre[1:-1, 1:-1] = pre
        contours = find_contours(padded_mask_pre, 0.5)
        for verts in contours:
            # reduce padding and  flipping from (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color_used, linewidth=1)
            ax.add_patch(p)

        # reduce the blank part generated by plt and keep the original resolution
        fig = plt.gcf()
        fig.set_size_inches(height/37.5, width/37.5)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        ax.imshow(masked_image.astype(np.uint8))
        # plt.show()
        fig.savefig('{}/{}.png'.format(save_path, file_name))
        # clear the image after saving
        plt.cla()
        plt.close(fig)


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]


def output_gif(input_path, output_path, gif_name):
    '''
    generate gif
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


if __name__ == "__main__":

    test_path = './data_val'
    subject_id = 9
    subject_name = 'subject-%d-' % subject_id
