import h5py
import torch
import torch.utils.data as data
import glob
import os
import numpy as np
from skimage.transform import resize


class CTData(data.Dataset):

    def __init__(self, root_path, augmentation = None):
        self.file_list = [x for x in glob.glob(os.path.join(root_path, "*.h5"))]
        self.augmentation = augmentation

    def __getitem__(self, index):
        # read h5 files
        h5file = h5py.File(self.file_list[index])
        self.data = h5file.get("data")
        self.label = h5file.get("label")
        self.subject = h5file.get("subject")
        self.slice = h5file.get("slice")
        self.data1 = self.data.value
        self.label1 = self.label.value
        self.subject1 = self.subject.value
        self.slice = self.slice.value

        self.data1 = resize(self.data1, (224, 224), order=3, mode="constant", cval=0, clip=True, preserve_range=True)
        self.label1 = resize(self.label1, (224, 224), order=0, mode="edge", cval=0, clip=True, preserve_range=True)

        # data augmentation
        if self.augmentation!=None:
            data_aug, mask_aug = data_augment(self.data1, self.label1, self.augmentation)
        else:
            data_aug = self.data1
            mask_aug = self.label1

        data1_norm = (data_aug - data_aug.mean()) / data_aug.std()

        image1_out = data1_norm
        label1_out = mask_aug.copy()
        label1_out_cat = label1_out.astype("float32")
        label1_out_cat = label1_out_cat / 255

        image1_out = image1_out.transpose((2,0,1))
        label1_out_cat = label1_out_cat.transpose((2,0,1))

        subject_out = np.array(self.subject1)
        slice_out = np.array(self.slice)
        return (torch.from_numpy(image1_out).float(),
                torch.from_numpy(label1_out_cat).long(),
                torch.from_numpy(subject_out).long(),
                torch.from_numpy(slice_out).long())

    def __len__(self):
        return len(self.file_list)


class CTData_test(data.Dataset):

    def __init__(self, root_path, augmentation = None):
        self.file_list = [x for x in glob.glob(os.path.join(root_path, "*.h5"))]
        self.augmentation = augmentation

    def __getitem__(self, index):
        # read h5 files
        h5file = h5py.File(self.file_list[index])
        self.data = h5file.get("data")
        self.label = h5file.get("label")
        self.subject = h5file.get("subject")
        self.slice = h5file.get("slice")
        self.data1 = self.data.value
        self.label1 = self.label.value
        self.subject1 = self.subject.value
        self.slice = self.slice.value

        self.data1 = resize(self.data1, (224, 224), order=3, mode="constant", cval=0, clip=True, preserve_range=True)
        self.label1 = resize(self.label1, (224, 224), order=0, mode="edge", cval=0, clip=True, preserve_range=True)

        # data augmentation
        if self.augmentation!=None:
            data_aug, mask_aug = data_augment(self.data1, self.label1, self.augmentation)
        else:
            data_aug = self.data1
            mask_aug = self.label1

        data1_norm = (data_aug - data_aug.mean()) / data_aug.std()

        image1_out = data1_norm
        label1_out = mask_aug.copy()
        label1_out_cat = label1_out.astype("float32")
        label1_out_cat = label1_out_cat / 255

        image1_out = image1_out.transpose((2,0,1))
        label1_out_cat = label1_out_cat.transpose((2,0,1))
        subject_out = np.array(self.subject1)
        slice_out = np.array(self.slice)
        return (torch.from_numpy(image1_out).float(),
                torch.from_numpy(label1_out_cat).long(),
                torch.from_numpy(subject_out).long(),
                torch.from_numpy(slice_out).long(),
                torch.from_numpy(data_aug).float())

    def __len__(self):
        return len(self.file_list)


# data augmentation
def data_augment_volume(datalist, augmentation):

        # first get the volume data from the data list
        image1, mask1 = datalist
        # Augmentation
        # This requires the imgaug lib (https://github.com/aleju/imgaug)
        if augmentation:
            import imgaug
            # Augmenters that are safe to apply to masks
            # Some, such as Affine, have settings that make them unsafe, so always
            # test your augmentation on masks
            MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                               "Fliplr", "Flipud", "CropAndPad",
                               "Affine", "PiecewiseAffine"]

            def hook(images, augmenter, parents, default):
                """Determines which augmenters to apply to masks."""
                return augmenter.__class__.__name__ in MASK_AUGMENTERS

            # Store shapes before augmentation to compare
            image1_shape = image1.shape
            mask1_shape = mask1.shape
            # Make augmenters deterministic to apply similarly to images and masks
            det = augmentation.to_deterministic()
            # image should be uint8!!
            image1 = det.augment_image(image1)

            # Change mask to np.uint8 because imgaug doesn't support np.bool
            mask1 = det.augment_image(mask1.astype(np.uint8),
                                      hooks=imgaug.HooksImages(activator=hook))
            # Verify that shapes didn't change
            assert image1.shape == image1_shape, "Augmentation shouldn't change image size"
            assert mask1.shape == mask1_shape, "Augmentation shouldn't change mask size"

            # Change mask back to bool
            # masks = masks.astype(np.bool)
        return image1,  mask1


def data_augment(image, mask, augmentation):
    # Augmentation
    # This requires the imgaug lib (https://github.com/aleju/imgaug)
    if augmentation:
        import imgaug
        # Augmenters that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always
        # test your augmentation on masks
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        # Store shapes before augmentation to compare
        image_shape = image.shape
        mask_shape = mask.shape
        # Make augmenters deterministic to apply similarly to images and masks
        det = augmentation.to_deterministic()
        # image should be uint8!!
        images = det.augment_image(image)
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        masks = det.augment_image(mask.astype(np.uint8),
                                   hooks=imgaug.HooksImages(activator=hook))
        # Verify that shapes didn't change
        assert images.shape == image_shape, "Augmentation shouldn't change image size"
        assert masks.shape == mask_shape, "Augmentation shouldn't change mask size"
        # Change mask back to bool
        # masks = masks.astype(np.bool)
    return images, masks


if __name__ == "__main__":
    path = "./data_train"
    data_generator = CTData(root_path=path)
    trainloader = data.DataLoader(data_generator, batch_size=1, shuffle=False)
    for i, data in enumerate(trainloader):
        img1 = data[0].numpy()
        img2 = data[1].numpy()
        imgs = np.concatenate((img1, img1, img1), axis=0).transpose(1,2,0)
        # cv.waitKey(500)
        a = 1


