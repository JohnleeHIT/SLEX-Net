import torch.utils.data as dataloader
from data.datagenerator import CTData
import torch.optim as optim
from miscellaneous.utils import mask_cross_entropy
from imgaug import augmenters as iaa
from config import Config
from models.segmentor_v1 import HybridUNet_single_out_consistency
import os
from miscellaneous.metrics import dice
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import pandas as pd
from skimage.transform import resize
import time


# on line evaluate the result
def online_eval(model, dataloader):
    with torch.no_grad():
        dice_new_list = []
        data_dict_list = []
        last_pre = np.zeros((2,2))
        last_tar = np.zeros((2, 2))
        shape_list = []
        Loss_epoch_val = 0
        for data_val in dataloader:
            images_val, targets_val, subject, slice = data_val
            model.eval()
            images_val = images_val.to(device)
            targets_val = targets_val.to(device)
            outputs_val = model(images_val, config.lamda_sem)
            outputs_val_out = outputs_val.final_out

            # calculate validation loss
            Loss_backbone = mask_cross_entropy(targets_val[:, 1, ...], outputs_val.final_out, alpha=config.alpha)
            Loss_SEM = mask_cross_entropy(targets_val[:, 1, ...], outputs_val.out1_sem_2, alpha=config.alpha) + \
                       mask_cross_entropy(targets_val[:, 1, ...], outputs_val.out1_sem_0, alpha=config.alpha)
            Loss_seg = Loss_backbone + Loss_SEM
            Loss_epoch_val += Loss_seg

            # slice-wise prediction
            outputs_val_1 = outputs_val_out[:,0:2, ...]
            _, predicted_1 = torch.max(outputs_val_1.data, 1)
            predicted_val_1 = predicted_1.data.cpu().numpy()

            subject_val = subject.data.cpu().numpy()
            slice_val = slice.data.cpu().numpy()
            slice_val_1 = slice_val[0][1]

            targets_val = targets_val.data.cpu().numpy()
            targets_val_1 = targets_val[:,1, ...]
            shape_list.append(predicted_val_1.shape)

            '''
                    append to a dict list, 
                    "subject": ID of the Patient.
                    "slice": Slice index in the volume of a Patient.
                    "Pre": Prediction of a slice.
                    "target": Label of a slice.
            '''
            data_dict_list.append({"subject": subject_val[0], "slice": slice_val_1, "pre": np.squeeze(predicted_val_1,axis=0),
                                   "target": np.squeeze(targets_val_1, axis=0)})

        # Turn slice-wise predictions to subject-wise predictions
        pd_data = pd.DataFrame(data_dict_list)
        for name, volume_data in pd_data.groupby("subject"):
            pre = volume_data["pre"]
            tar = volume_data["target"]
            pre_array = pre.values
            target_array = tar.values
            pre_temp = np.zeros((len(pre_array), pre_array[0].shape[0], pre_array[0].shape[1]), dtype="int16")
            target_temp = np.zeros((len(pre_array), target_array[0].shape[0], target_array[0].shape[1]), dtype="int16")
            for i in range(len(pre_array)):
                pre_temp[i, :, :] = pre_array[i]

            for i in range(len(target_array)):
                target_temp[i, :, :] = target_array[i]

            dsc_list1 = []

            # calculate dice coefficient
            for i in range(0, config.num_classes):
                dsc_i = dice(pre_temp, target_temp, i)
                dsc_list1.append(dsc_i)
            dice_new_list.append(dsc_list1)

        dice_array = np.array(dice_new_list)
        dice_mean = np.mean(dice_array, axis=0)

    return dice_mean, Loss_epoch_val


os.environ["CUDA_VISIBLE_DEVICES"] = '3'  # '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # Select the Nvidia card
    start_time = time.time()
    torch.set_num_threads(2)

    #  ********************training and On-line Validation***************************
    config = Config()
    config.display()

    train_path = os.path.join(config.data_path, "data_train")
    val_path = os.path.join(config.data_path, "data_val")

    if os.path.exists("hybrid_consistency.txt"):
        os.remove("hybrid_consistency.txt")

    with open("hybrid_consistency.txt", "a") as mylog:

        config.write_config(mylog)

        # define a SummaryWriter for show loss curve in the Tensorboard
        best_dice_train = 0
        best_dice_val = 0
        with SummaryWriter("./result") as writer:
            # define the main model
            model_S = HybridUNet_single_out_consistency(backbone_channel=1, backbone_class=2,
                                 drop_rate=config.drop_rate, SEM_channels=3, SEM_class=2).to(device)
            # loss function
            weight_tensor = torch.from_numpy(np.array(config.alpha)).float()
            criterion_S = nn.CrossEntropyLoss(weight_tensor).cuda()

            # whether to pretrain from the backbone network and SEM
            if config.pre_trained:
                print("start pre train from backbone and SEM")
                checkpoint_bb = torch.load(config.pre_trained_model)
                model_S.backbone.load_state_dict(checkpoint_bb)
                checkpoint_sem = torch.load(config.pre_trained_SEM_model)
                model_S.SEM.load_state_dict(checkpoint_sem)
                # # Test freezing parameters of SEM
                # for p in model_S.SEM.parameters():
                #     p.requires_grad = False
                # for name, value in model_S.SEM.named_parameters():
                #     print('name: {},\t grad: {}'.format(name, value.requires_grad))
                # # Test freezing parameters of backbone
                # for p in model_S.backbone.parameters():
                #     p.requires_grad = False
                # for name, value in model_S.backbone.named_parameters():
                #     print('name: {},\t grad: {}'.format(name, value.requires_grad))
                print("Pre train succeed SEM and backbone!")
            # whether load old models
            if config.load_old_model:
                print("start load checkpoint: {}".format(config.note))
                checkpoint = torch.load(config.checkpoint)
                model_S.load_state_dict(checkpoint)
                print("load checkpoint succeed!")

            # set optimizer
            optimizer_S = optim.Adam(model_S.parameters(), lr=config.lr_S, weight_decay=6e-4, betas=(0.97, 0.999))
            # Test freezing parameters of SEM
            # filtered_para = filter(lambda p: p.requires_grad, model_S.parameters())
            # optimizer_S = optim.Adam(filtered_para, lr=config.lr_S,
            #                          weight_decay=6e-4, betas=(0.97, 0.999))
            scheduler_S = optim.lr_scheduler.StepLR(optimizer_S, step_size=config.step_size_S, gamma=0.1)

            # add data augmentation
            augmentation = iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.OneOf([iaa.Affine(rotate=30),
                           iaa.Affine(rotate=60),
                           iaa.Affine(rotate=330)]),
                iaa.GaussianBlur(sigma=(0.0, 3.0))
            ])
            # data augmentation
            if config.augment:
                ct_data_train = CTData(train_path, augmentation=augmentation)
            else:
                ct_data_train = CTData(train_path, augmentation=None)

            trainloader = dataloader.DataLoader(ct_data_train, batch_size=config.batch_train, shuffle=True)

            ct_data_val = CTData(val_path, augmentation = None)

            valloader = dataloader.DataLoader(ct_data_val, batch_size=1, shuffle=False)
            valloader_train = dataloader.DataLoader(ct_data_train, batch_size=1, shuffle=False)

            test = 0
            print('Rate     | epoch  | Loss seg|DSC_train(bg)| DSC_train(fg)| DSC_val(bg)| DSC_val(fg) ')
            mylog.close()

            for epoch in range (config.num_epoch):
                start_time = time.time()
                loss_epoch = 0
                # loss_epoch_val = 0
                mylog  = open("hybrid_consistencys.txt", "a")
                scheduler_S.step(epoch)
                # zero the parameter gradients
                model_S.train()
                for i, data in enumerate(trainloader):
                    images, targets, _, _ = data
                    # Set mode cuda if it is enable, otherwise mode CPU
                    images = images.to(device)
                    targets = targets.to(device)
                    # mask = mask.to(device)
                    optimizer_S.zero_grad()
                    outputs = model_S(images, config.lamda_sem)

                    loss_backbone = mask_cross_entropy(targets[:,1,...], outputs.final_out, alpha=config.alpha)
                    loss_SEM = mask_cross_entropy(targets[:,1,...], outputs.out1_sem_2, alpha=config.alpha)+\
                               mask_cross_entropy(targets[:,1,...], outputs.out1_sem_0, alpha=config.alpha)
                    loss_seg = loss_backbone+loss_SEM

                    loss_epoch += loss_seg
                    loss_seg.backward()
                    optimizer_S.step()
                    test = i

                end_time = time.time()
                elapsed = end_time-start_time

                # -----------------------Validation------------------------------------
                dice_val, loss_epoch_val = online_eval(model_S, valloader)
                dice_train, _ = online_eval(model_S, valloader_train)
                #-------------------Debug-------------------------
                for param_group in optimizer_S.param_groups:
                    print('%0.6f | %6d | %0.5f | %0.5f| %0.5f| %0.5f| %0.5f ' % (
                            param_group['lr'], epoch,
                            loss_epoch.data.cpu().numpy(),
                            dice_train[0], dice_train[1],
                            dice_val[0], dice_val[1]))

                mylog.write("epoch_{:30} train_loss:  {:3f}  val_loss:  {:3f} elapse:  {:3f} train_dice: {:3f}  {:3f}  val_dice: {:3f}  "
                            "{:3f} \n".format(str(epoch), loss_epoch, loss_epoch_val, elapsed, dice_train[0], dice_train[1],
                                                                                                 dice_val[0], dice_val[1]))
                mylog.close()
                # save tensorboard files
                writer.add_scalar("train_loss", loss_epoch, epoch)
                writer.add_scalar("val_loss", loss_epoch_val, epoch)

                # save best checkpoint
                if dice_val[1]>best_dice_val:
                    torch.save(model_S.state_dict(), "./checkpoints/best_checkpoint.pth")
                    best_dice_val = dice_val[1]
                if dice_train[1]>best_dice_train:
                    torch.save(model_S.state_dict(), "./checkpoints/best_checkpoint_train.pth")
                    best_dice_train = dice_train[1]

            # export the loss for external use
            writer.export_scalars_to_json("./result/all_scalar.json")
            writer.close()