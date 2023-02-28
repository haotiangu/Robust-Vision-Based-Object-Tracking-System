# :computer: How to Train/Finetune FPRN

- [Train FPRN](#train-fprn)
  - [Overview](#overview)
  - [Dataset Preparation](#dataset-preparation)
  - [Train SRNet](#Train-SRNet)
  - [Train Autoencoder](#Train-Autoencoder)
- [Finetune FPRN on your own dataset](#Finetune-FPRN-on-your-own-dataset)
  - [Generate degraded images on the fly](#Generate-degraded-images-on-the-fly)
  - [Use paired training data](#use-your-own-paired-data)



## Train FPRN

### Overview

The training has been divided into two stages. These two stages have the same data synthesis process and training pipeline, except for the loss functions. Specifically,

1. We first train SRNet with L1 loss from the pre-trained model SRNet.
1. We then use the trained SRNet model as an initialization of the generator, and train the Autoencoder with a combination of L1 loss, perceptual loss and GAN loss.

### Dataset Preparation

We use DF2K datasets for our training. Only HR images are required. <br>
You can download from :

1. DIV2K: http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip

Here are steps for data preparation.

#### Step 1: [Optional] Generate multi-scale images

For the DF2K dataset, we use a multi-scale strategy, *i.e.*, we downsample HR images to obtain several Ground-Truth images with different scales. <br>
You can use the [scripts/generate_multiscale_DF2K.py](scripts/generate_multiscale_DF2K.py) script to generate multi-scale images. <br>
Note that this step can be omitted if you just want to have a fast try.

```bash
python scripts/generate_multiscale_DF2K.py --input datasets/DF2K/DF2K_HR --output datasets/DF2K/DF2K_multiscale
```

#### Step 2: [Optional] Crop to sub-images

We then crop DF2K images into sub-images for faster IO and processing.<br>
This step is optional if your IO is enough or your disk space is limited.

You can use the [scripts/extract_subimages.py](scripts/extract_subimages.py) script. Here is the example:

```bash
 python scripts/extract_subimages.py --input datasets/DF2K/DF2K_multiscale --output datasets/DF2K/DF2K_multiscale_sub --crop_size 400 --step 200
```

#### Step 3: Prepare a txt for meta information

You need to prepare a txt file containing the image paths. The following are some examples in `meta_info_DF2Kmultiscale.txt` (As different users may have different sub-images partitions, this file is not suitable for your purpose and you need to prepare your own txt file):

```txt
DF2K_HR_sub/000001_s001.png
DF2K_HR_sub/000001_s002.png
DF2K_HR_sub/000001_s003.png
...
```

You can use the [scripts/generate_meta_info.py](scripts/generate_meta_info.py) script to generate the txt file. <br>
You can merge several folders into one meta_info txt. Here is the example:

```bash
 python scripts/generate_meta_info.py --input datasets/DF2K/DF2K_HR datasets/DF2K/DF2K_multiscale --root datasets/DF2K datasets/DF2K --meta_info datasets/DF2K/meta_info/meta_info_DF2Kmultiscale.txt
```

### Train SRNet

1. Download pre-trained model [SRNet](https://github.com/haotiangu/FPRN/releases/download/FPRN/SRNET_SRx4_DF2K.pth) into `experiments/pretrained_models`.
    ```bash
    wget https://github.com/haotiangu/FPRN/releases/download/FPRN/SRNET_SRx4_DF2K.pth -P experiments/pretrained_models
    ```
1. Modify the content in the option file `options/train_SRNet_x4plus.yml` accordingly:
    ```yml
    train:
        name: DF2K
        type: SRDataset
        dataroot_gt: datasets/DF2K  # modify to the root path of your folder
        meta_info: datasets/DF2K/meta_info/meta_info_DF2Kmultiscale.txt  # modify to your own generate meta info txt
        io_backend:
            type: disk
    ```
1. If you want to perform validation during training, uncomment those lines and modify accordingly:
    ```yml
      # Uncomment these for validation
      # val:
      #   name: validation
      #   type: PairedImageDataset
      #   dataroot_gt: path_to_gt
      #   dataroot_lq: path_to_lq
      #   io_backend:
      #     type: disk

    ...

      # Uncomment these for validation
      # validation settings
      # val:
      #   val_freq: !!float 5e3
      #   save_img: True

      #   metrics:
      #     psnr: # metric name, can be arbitrary
      #       type: calculate_psnr
      #       crop_border: 4
      #       test_y_channel: false
    ```
1. Before the formal training, you may run in the `--debug` mode to see whether everything is OK. We use four GPUs for training:
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 FPRN/train.py -opt options/train_SRNet_x4plus.yml --launcher pytorch --debug
    ```

    Train with **a single GPU** in the *debug* mode:
    ```bash
    python FPRN/train.py -opt options/train_SRNet_x4plus.yml --debug
    ```
1. The formal training. We use four GPUs for training. We use the `--auto_resume` argument to automatically resume the training if necessary.
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 FPRN/train.py -opt options/train_SRNet_x4plus.yml --launcher pytorch --auto_resume
    ```

    Train with **a single GPU**:
    ```bash
    python FPRN/train.py -opt options/train_SRNet_x4plus.yml --auto_resume
    ```

### Train Autoencoder

1. After the training of SRNet, you now have the file `experiments/train_RealESRNetx4plus_1000k_B12G4_fromESRGAN/model/net_g_1000000.pth`. If you need to specify the pre-trained path to other files, modify the `pretrain_network_g` value in the option file `train_fprn_x4plus.yml`.
1. Modify the option file `train_fprn_x4plus.yml` accordingly. Most modifications are similar to those listed above.
1. Before the formal training, you may run in the `--debug` mode to see whether everything is OK. Before running training script, check the GPU configuration no matter server or workstation. We use four GPUs for training:
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 FPRN/train.py -opt options/train_fprn_x4plus.yml --launcher pytorch --debug
    ```

    Train with **a single GPU** in the *debug* mode:
    ```bash
    python FPRN/train.py -opt options/train_fprn_x4plus.yml --debug
    ```
1. The formal training. We use four GPUs for training. We use the `--auto_resume` argument to automatically resume the training if necessary.
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 FPRN/train.py -opt options/train_fprn_x4plus.yml --launcher pytorch --auto_resume
    ```

    Train with **a single GPU**:
    ```bash
    python FPRN/train.py -opt options/train_fprn_x4plus.yml --auto_resume
    ```

## Finetune FPRN on your own dataset

You can finetune FPRN on your own dataset. Typically, the fine-tuning process can be divided into two cases:

1. [Generate degraded images on the fly](#Generate-degraded-images-on-the-fly)
1. [Use your own **paired** data](#Use-your-own-paired-data)

### Generate degraded images on the fly

Only high-resolution images are required. The low-quality images are generated with the two degradation process described in FPRN during training.

**1. Prepare dataset**

See [this section](#dataset-preparation) for more details.

**2. Download pre-trained models**

Download pre-trained models into `experiments/pretrained_models`.

- *FPRN_x4plus.pth*:
    ```bash
    wget https://github.com/haotiangu/FPRN/releases/download/FPRN/FPRN_x4plus.pth -P experiments/pretrained_models
    ```

- *FPRN_x4plus_netD.pth*:
    ```bash
    wget https://github.com/haotiangu/FPRN/releases/download/FPRN/FPRN_x4plus_netD.pth -P experiments/pretrained_models
    ```

**3. Finetune**

Modify [options/finetune_fprn_x4plus.yml](options/finetune_fprn_x4plus.yml) accordingly, especially the `datasets` part:

```yml
train:
    name: DF2K
    type: FPRNDataset
    dataroot_gt: datasets/DF2K  # modify to the root path of your folder
    meta_info: datasets/DF2K/meta_info/meta_info_DF2Kmultiscale.txt  # modify to your own generate meta info txt
    io_backend:
        type: disk
```

We use four GPUs for training. We use the `--auto_resume` argument to automatically resume the training if necessary.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 FPRN/train.py -opt options/finetune_fprn_x4plus.yml --launcher pytorch --auto_resume
```

Finetune with **a single GPU**:
```bash
python FPRN/train.py -opt options/finetune_fprn_x4plus.yml --auto_resume
```

### Use your own paired data

You can also finetune FPRN with your own paired data.

**1. Prepare dataset**

Assume that you already have two folders:

- **gt folder** (Ground-truth, high-resolution images): *datasets/DF2K/DIV2K_train_HR_sub*
- **lq folder** (Low quality, low-resolution images): *datasets/DF2K/DIV2K_train_LR_bicubic_X4_sub*

Then, you can prepare the meta_info txt file using the script [scripts/generate_meta_info_pairdata.py](scripts/generate_meta_info_pairdata.py):

```bash
python scripts/generate_meta_info_pairdata.py --input datasets/DF2K/DIV2K_train_HR_sub datasets/DF2K/DIV2K_train_LR_bicubic_X4_sub --meta_info datasets/DF2K/meta_info/meta_info_DIV2K_sub_pair.txt
```

**2. Download pre-trained models**

Download pre-trained models into `experiments/pretrained_models`.

- *FPRN_x4plus.pth*
    ```bash
    wget https://github.com/haotiangu/FPRN/releases/download/FPRN/FPRN_x4plus.pth -P experiments/pretrained_models
    ```

- *FPRN_x4plus_netD.pth*
    ```bash
    wget https://github.com/haotiangu/FPRN/releases/download/FPRN/FPRN_x4plus_netD.pth -P experiments/pretrained_models
    ```

**3. Finetune**

Modify [options/finetune_fprn_x4plus_pairdata.yml](options/finetune_fprn_x4plus_pairdata.yml) accordingly, especially the `datasets` part:

```yml
train:
    name: DIV2K
    type: FPRNPairedDataset
    dataroot_gt: datasets/DF2K  # modify to the root path of your folder
    dataroot_lq: datasets/DF2K  # modify to the root path of your folder
    meta_info: datasets/DF2K/meta_info/meta_info_DIV2K_sub_pair.txt  # modify to your own generate meta info txt
    io_backend:
        type: disk
```

We use four GPUs for training. We use the `--auto_resume` argument to automatically resume the training if necessary.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 FPRN/train.py -opt options/finetune_fprn_x4plus_pairdata.yml --launcher pytorch --auto_resume
```

Finetune with **a single GPU**:
```bash
python FPRN/train.py -opt options/finetune_fprn_x4plus_pairdata.yml --auto_resume
```