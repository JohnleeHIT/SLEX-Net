import torch
import torch.backends.cudnn as cudnn


class Config(object):
    drop_rate = 0
    lamda_sem = 0.2
    alpha = [0.1, 1]
    gamma = 2
    data_path = "/home/lixiangyu/Dataset/new_hemorrhage_cut_slices_count"
    # Network setting
    pre_trained = True
    load_old_model = False
    augment = False
    save_segmentation = True
    save_uncertainty = False
    # Optimization
    num_epoch = 100
    lr_S = 1e-5
    lr_D = 0.0005
    momentum_S=0.9
    momentum_D=0.9
    step_size_S = 20
    step_size_D = 5
    beta1=0.9
    beta2=0.999
    batch_train = 2
    # CUDNN
    cudnn.enabled = True
    cudnn.benchmark=True
    num_classes= 2
    # Note
    checkpoint_name= 'model_2d_unet'

    num_checkpoint = '00020'
    note= str(num_checkpoint) +'_' + checkpoint_name

    # pretrain
    pre_trained_model = "checkpoints/backbone.pth"
    pre_trained_SEM_model = "checkpoints/SEM.pth"

    # load old model
    checkpoint='./checkpoints/ah_3d_temp.pth'

    # Testing
    num_ckp_test = '00040'

    # ckp_test = '/home/lixiangyu/myprojects/TMI/result/test_other_methods/AH-Net/resize 224X224 crop_size 224X224X16 5倍输入 augmentation/checkpoints/best_checkpoint.pth'
    # ckp_test = '/home/lixiangyu/myprojects/TMI/result/test_other_methods/AH-Net/resize 224X224/checkpoints/best_checkpoint.pth'
    # ckp_test = '/home/lixiangyu/myprojects/TMI/result/test_other_methods/Asconv/224X224/checkpoints/best_checkpoint.pth'
    ckp_test = '/home/lixiangyu/myprojects/TMI/result/test_weight_consistency/测试consistency/lamda=0.2情况/best_checkpoint.pth'


    remarks = " test best result"

    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()

    def __init__(self):
        pass

    def display(self):
        '''
        display configurations
        :return:
        '''
        print("configurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

    def write_config(self, log):
        log.write("configurations: \n")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                log.write("{:30} {} \n".format(a, getattr(self, a)))

