import torch
import numpy as np
import torch.utils.data as dataloader
import os
from config import Config
from models.segmentor_v1 import HybridUNet_single_out_consistency
from miscellaneous.metrics import dice, rAVD, hd, brier, nll, ece
import pandas as pd
from data.datagenerator import CTData_test
from miscellaneous.utils import Evaluation
from matplotlib import pyplot as plt
import h5py
import time
import cv2 as cv
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve


def online_eval(model, dataloader, txtlog, submit_path, uncertaintys_path, save_segmentation, save_uncertainty):
    txtlog.write( "Dice_mean fg|bg|hausdorff_dist|ravd|ece|nll|sklearn_brier\n")
    my_evaluation = Evaluation()
    start_time = time.time()
    with torch.no_grad():
        dice_new_list = []
        data_dict_list = []
        hausdorff_dist_list = []
        ravd_list = []
        shape_list = []
        testset_list_pre = []
        testset_list_gt = []
        nll_list = []
        brier_list = []
        brier_sklearn_list = []
        ece_list = []
        for data_val in dataloader:
            images_val, targets_val, subject, slice, images_origin = data_val
            model.eval()
            images_val = images_val.to(device)
            targets_val = targets_val.to(device)
            outputs = model(images_val, test_config.lamda_sem)
            # final_out [i-1,i,i+1]
            outputs_val = outputs.final_out
            softmax = outputs.softmax_out
            # calculate predicted entropy as uncertainty
            softmax_1 = torch.unsqueeze(softmax[:,1,...],dim=1)
            softmax_2 = torch.unsqueeze(softmax[:, 3, ...], dim=1)
            softmax_3 = torch.unsqueeze(softmax[:, 5, ...], dim=1)
            softmax_fg = torch.cat((softmax_1, softmax_2, softmax_3), dim=1)
            softmax_fg_numpy = softmax_fg.data.cpu().numpy()
            softmax_fg_numpy = np.squeeze(softmax_fg_numpy, axis=0)
            mean_fg = np.mean(softmax_fg_numpy, axis=0)
            entropy = -mean_fg*np.log(mean_fg)

            # softmax outputs for uncertainty quantification
            softmax_final_out = softmax[:,6:8,...]
            softmax_final_out = np.squeeze(softmax_final_out.data.cpu().numpy(), axis=0)
            # 逐切片处理
            outputs_val_1 = outputs_val[:,0:2, ...]

            image_origin = images_origin.data.cpu().numpy()
            image_origin1 = np.squeeze(image_origin, axis=0)
            image_origin1 = image_origin1[:, :, 1]

            _, predicted_1 = torch.max(outputs_val_1.data, 1)

            # ----------Compute dice-----------
            predicted_val_1 = predicted_1.data.cpu().numpy()
            subject_val = subject.data.cpu().numpy()
            slice_val = slice.data.cpu().numpy()
            slice_val_1 = slice_val[0][1]
            targets_val = targets_val.data.cpu().numpy()
            targets_val_1 = targets_val[:,1, ...]

            shape_list.append(predicted_val_1.shape)
            data_dict_list.append({"subject": subject_val[0], "slice": slice_val_1, "pre": np.squeeze(predicted_val_1,axis=0),
                    "target": np.squeeze(targets_val_1, axis=0), "image": image_origin1, "uncertainty": entropy, "softmax_out":softmax_final_out})

        # test the elaps of uncertainty quantification
        end_time = time.time()
        print("elapsed:{}".format(end_time-start_time))
        # 利用pandas分组
        pd_data = pd.DataFrame(data_dict_list)
        for subject, volume_data in pd_data.groupby("subject"):
            pre = volume_data["pre"]
            tar = volume_data["target"]
            slices = volume_data["slice"]
            image = volume_data["image"]
            uncertain = volume_data["uncertainty"]
            softmax_prob = volume_data["softmax_out"]

            pre_array = pre.values
            target_array = tar.values
            image_array = image.values
            uncertain_arr = uncertain.values
            slices_arr = slices.values
            softmax_prob_arr = softmax_prob.values

            pre_temp = np.zeros((len(pre_array), pre_array[0].shape[0], pre_array[0].shape[1]), dtype="int16")
            target_temp = np.zeros((len(pre_array), target_array[0].shape[0], target_array[0].shape[1]), dtype="int16")
            # dimentions: slices*class*width*height
            softmax_probs_temp = np.zeros((len(pre_array), softmax_prob_arr[0].shape[0], softmax_prob_arr[0].shape[1],softmax_prob_arr[0].shape[2]), dtype="float32")
            for i in range(len(pre_array)):
                pre_temp[i, :, :] = pre_array[i]
                target_temp[i, :, :] = target_array[i]
                softmax_probs_temp[i,:,:,:] = softmax_prob_arr[i]
                # 保存预测结果与GT及图像
                if save_segmentation:
                    image_slice = image_array[i]
                    # save image and segmentation
                    my_evaluation.save_contour_label(image_slice.astype("int16"),
                                                  target_array[i],save_path=submit_path, color="red", file_name=str(subject)+"_"+
                                                  str(slices_arr[i])+"label",show_mask=True)
                    my_evaluation.save_contour_label(image_slice.astype("int16"),
                                                  pre_array[i], save_path=submit_path, color="blue", file_name=str(subject)+"_"+
                                                  str(slices_arr[i])+"pre", show_mask=True)

                    orig_path = os.path.join(submit_path, str(subject)+"_"+str(slices_arr[i])+'.png')
                    cv.imwrite(orig_path, image_slice.astype("uint8"))
                if save_uncertainty:
                    # Predicted error map
                    error = np.abs(pre_array[i]-target_array[i])
                    error_name = str(subject) + "_" + str(slices_arr[i]) + "error.png"
                    error_file_path = os.path.join(uncertaintys_path, error_name)
                    plt.figure()
                    plt.imshow(error, cmap=plt.cm.Reds, interpolation='nearest')
                    # Visulization of the uncertainty
                    file_name = str(subject) + "_" + str(slices_arr[i]) + ".png"
                    file_path = os.path.join(uncertaintys_path, file_name)
                    plt.colorbar()
                    plt.xticks([])
                    plt.yticks([])
                    plt.savefig(error_file_path)
                    plt.clf()
                    plt.cla()
                    plt.close()

                    plt.figure()
                    plt.imshow(uncertain_arr[i], cmap=plt.cm.rainbow, interpolation='nearest')
                    plt.colorbar()
                    plt.xticks([])
                    plt.yticks([])
                    # plt.axes('off')
                    plt.savefig(file_path)
                    plt.clf()
                    plt.cla()
                    plt.close()

            dsc_list1 = []
            if 0 == np.count_nonzero(pre_temp):
                print("zero"+"_"+str(subject))
                continue

            # calculate the dice metric
            for i in range(0, test_config.num_classes):
                dsc_i = dice(pre_temp, target_temp, i)
                dsc_list1.append(dsc_i)

            # Calculate Hausdorff Distance 以及ravd
            hausdorff_dist = hd(pre_temp, target_temp, [5, 0.42, 0.42])
            # we measure the absolute volume difference
            ravd = abs(rAVD(pre_temp, target_temp))

            # calculate the volume of ICH for GT and predictions
            volume_gt = calculate_volume(target_temp)
            volume_pre = calculate_volume(pre_temp)

            # Evaluate uncertainty qualification with nll, brier, ece
            softmax_probs_temp = softmax_probs_temp.transpose(1,0,2,3)
            brier_socre = brier(torch.from_numpy(softmax_probs_temp).float(), torch.from_numpy(target_temp).long())
            ece_subject_wise,_,_= ece(softmax_probs_temp[1,:,:,:], target_temp, 10)
            # Test sklearn
            target_onehot_temp = one_hot(target_temp, 2)

            brier_sklearn = brier_score_loss(target_onehot_temp[0, ...].flatten(), softmax_probs_temp[0, ...].flatten())+\
            brier_score_loss(target_onehot_temp[1,...].flatten(), softmax_probs_temp[1,...].flatten())

            nll_score = nll(torch.from_numpy(softmax_probs_temp).float(), torch.from_numpy(target_temp).long())
            print("nll_score:{}  brier_socre:{}".format(nll_score.data.numpy(), brier_socre.data.numpy()))
            print("dice_bg:{}  dice_fg:{}  Hausdorff_dist:{} ravd:{}".format(dsc_list1[0], dsc_list1[1],hausdorff_dist, ravd))
            txtlog.write("ID{:30} {:3f}  {:3f} {:3f} {:3f} {:3f}   {:3f}  {:3f} {:3f} {:3f} \n".format(subject, dsc_list1[0], dsc_list1[1],
                                                                            hausdorff_dist, ravd, ece_subject_wise, nll_score, brier_sklearn,volume_gt, volume_pre))
            dice_new_list.append(dsc_list1)
            hausdorff_dist_list.append(hausdorff_dist)
            ravd_list.append(ravd)

            brier_list.append(brier_socre.data.numpy())
            nll_list.append(nll_score.data.numpy())
            brier_sklearn_list.append(brier_sklearn)
            ece_list.append(ece_subject_wise)
            # store all the test data
            testset_list_pre.append(softmax_probs_temp[1,:,:,:])
            testset_list_gt.append(target_temp)

        dice_array = np.array(dice_new_list)
        dice_mean = np.mean(dice_array, axis=0)
        haus_dist_arr = np.array(hausdorff_dist_list)
        hausdorff_dist_mean = np.mean(haus_dist_arr, axis=0)
        ravd_arr = np.array(ravd_list)
        ravd_mean = np.mean(ravd_arr, axis=0)

        # uncertainty quantification
        brier_array = np.mean(np.array(brier_list),axis=0)
        nll_array = np.mean(np.array(nll_list), axis=0)
        brier_sklearn_mean = np.mean(np.array(brier_sklearn_list),axis=0)
        ece_subject_mean = np.mean(np.array(ece_list),axis=0)

        stacked_pre = merge_samples(testset_list_pre)
        stacked_gt = merge_samples(testset_list_gt)
        print("pre:{}  gt:{}".format(stacked_pre.shape, stacked_gt.shape))
        ece_score, confidence, accuracy = ece(stacked_pre,stacked_gt, 10)
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(stacked_gt.flatten(), stacked_pre.flatten(), n_bins=10)

        # Draw Reliability Diagram (binned version and curve version)
        x = np.linspace(0, 1. + 1e-8, 10)
        y3 = x
        plt.plot([0, 1], [0, 1], "k:")
        plt.bar(x, height=fraction_of_positives, color='b', width=-0.112, label='Outputs', linewidth=2, edgecolor=['black'] * len(x),
                align='edge')
        plt.bar(x, height=y3 - fraction_of_positives, color='g', bottom=fraction_of_positives, width=-0.112, label='Gap', linewidth=2,
                edgecolor=['black'] * len(x), align='edge')
        plt.xlim(0., 1.)
        plt.ylim(0., 1.)
        plt.xlabel("Confidence")
        plt.ylabel("Accuracy")
        # plt.title("Histogram polt")
        plt.legend(loc="upper left")
        plt.savefig('reliability_diagram_bined.png', dpi=400, bbox_inches='tight')

        plt.figure(figsize=(5, 5))
        ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=2)
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="calibrated_sklearn")
        ax1.set_ylabel("Fraction of positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="upper left")
        ax1.set_title('Calibration plots  (reliability curve)')
        plt.savefig('reliability_diagram_sklearn.png', dpi=400, bbox_inches='tight')

        with h5py.File("reliability_se_net.h5", "w") as f:
            f['condifence'] = confidence
            f['accuracy'] = accuracy
        txtlog.write("Dice_mean fg|bg|hausdorff_dist|ravd|ece|brier|nll|sklearn_brier|ece_sub_mea"\
                     "n:{:3f} ||{:3f}||{:3f}||{:3f}||{:3f}||{:3f}||{:3f}||{:3f} ||{:3f}\n".format(dice_mean[0],\
                    dice_mean[1], hausdorff_dist_mean, ravd_mean,ece_score,brier_array, nll_array, brier_sklearn_mean,ece_subject_mean))
        txtlog.write("Time Elapsed:  {}".format(end_time - start_time))
    return dice_mean

def mat2gray(I,limits):
    i = I.astype(np.float64)
    graymax = float(limits[1])
    graymin = float(limits[0])
    delta = 1 / (graymax - graymin)
    gray = delta * i - graymin * delta
    # 进行截断，对于大于最大值与小于最小值的部分，大于最大值设为1,小于最小值的设为0
    graycut = np.maximum(0, np.minimum(gray, 1))
    return graycut

def merge_samples(sample_list):
    '''
    merge slice-wise predictions or ground-truth to that of volume-vise
    :param sample_list:
    :return:
    '''
    sample = sample_list[0]
    for i in range(1,len(sample_list)):
        sample_stack = np.concatenate((sample, sample_list[i]), axis=0)
        sample = sample_stack
    return sample

def one_hot(input, class_n):
    '''
    onehot for pytorch
    :param input: N*H*W*D
    :param class_n:
    :return:N*n_class*H*W*D
    '''
    shape = input.shape
    onehot = np.zeros((class_n,)+shape)
    for i in range(class_n):
        onehot[i, ...] = (input == i)
    # onehot_trans = onehot.permute(1,0,2,3,4)
    onehot_trans = onehot
    return onehot_trans

def calculate_volume(binary_volume, pixel_spacing=(5,0.42,0.42)):
    '''
    calculate the volume of the hemorrhage
    :param binary_volume: D*W*H
    :param pixel_spacing: unit: mm
    :return:
    '''
    shape = binary_volume.shape
    volume = 0
    for i in range(shape[0]):
        binary_slice = binary_volume[i,:,:]
        volume_slice = np.sum(binary_slice)*pixel_spacing[0]*pixel_spacing[1]*pixel_spacing[2]
        volume += volume_slice
    # unit transfer to mL
    volume = volume/1000
    return volume



os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    test_config = Config()
    net = HybridUNet_single_out_consistency(backbone_channel=1, backbone_class=2, SEM_channels=3, SEM_class=2,
                                            drop_rate=test_config.drop_rate).to(device)
    # val_path = os.path.join(test_config.data_path, "data_val")
    val_path = os.path.join(test_config.data_path, "data_test")
    ct_data_val = CTData_test(val_path, augmentation=None)
    valloader = dataloader.DataLoader(ct_data_val, batch_size=1, shuffle=False)
    save_segmentation = test_config.save_segmentation
    save_uncertain = test_config.save_uncertainty

    dsc_mean_list = []
    if os.path.exists("test_lamda_consistency.txt"):
        os.remove("test_lamda_consistency.txt")

    with open("test_lamda_consistency.txt", "a") as txtlog:
        test_config.write_config(txtlog)
        # -----------------------Testing-------------------------------------
        # -----------------------Load the checkpoint (weights)---------------
        print ('Checkpoint: ', test_config.ckp_test)
        saved_state_dict = torch.load(test_config.ckp_test)
        net.load_state_dict(saved_state_dict)
        net.eval()
        submit_path = './submit'
        if not os.path.exists(submit_path):
            os.makedirs(submit_path)
        uncetainty_path = './uncertainty'
        if not os.path.exists(uncetainty_path):
            os.mkdir(uncetainty_path)
        online_eval(net, valloader, txtlog, submit_path, uncetainty_path, save_segmentation, save_uncertain)


