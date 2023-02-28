import numpy as np


from FPRN.utils import FPRNer

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in SR block in SRNet.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x

class SR(nn.Module):
    """Residual in Residual Dense Block.
    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(SR, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x



@ARCH_REGISTRY.register()
class SRNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in for improving the image resolution.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(SRNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(SR, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out





def test_fprner():
    # initialize with default model
    restorer = FPRNer(
        scale=4,
        model_path='experiments/pretrained_models/RealESRGAN_x4plus.pth',
        model=None,
        tile=10,
        tile_pad=10,
        pre_pad=2,
        half=False)
    assert isinstance(restorer.model, SRNet)
    assert restorer.half is False
    # initialize with user-defined model
    model = SRNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    restorer = FPRNer(
        scale=4,
        model_path='experiments/pretrained_models/RealESRGAN_x4plus_anime_6B.pth',
        model=model,
        tile=10,
        tile_pad=10,
        pre_pad=2,
        half=True)
    # test attribute
    assert isinstance(restorer.model, SRNet)
    assert restorer.half is True

    # ------------------ test pre_process ---------------- #
    img = np.random.random((12, 12, 3)).astype(np.float32)
    restorer.pre_process(img)
    assert restorer.img.shape == (1, 3, 14, 14)
    # with modcrop
    restorer.scale = 1
    restorer.pre_process(img)
    assert restorer.img.shape == (1, 3, 16, 16)

    # ------------------ test process ---------------- #
    restorer.process()
    assert restorer.output.shape == (1, 3, 64, 64)

    # ------------------ test post_process ---------------- #
    restorer.mod_scale = 4
    output = restorer.post_process()
    assert output.shape == (1, 3, 60, 60)

    # ------------------ test tile_process ---------------- #
    restorer.scale = 4
    img = np.random.random((12, 12, 3)).astype(np.float32)
    restorer.pre_process(img)
    restorer.tile_process()
    assert restorer.output.shape == (1, 3, 64, 64)

    # ------------------ test enhance ---------------- #
    img = np.random.random((12, 12, 3)).astype(np.float32)
    result = restorer.enhance(img, outscale=2)
    assert result[0].shape == (24, 24, 3)
    assert result[1] == 'RGB'

    # ------------------ test enhance with 16-bit image---------------- #
    img = np.random.random((4, 4, 3)).astype(np.uint16) + 512
    result = restorer.enhance(img, outscale=2)
    assert result[0].shape == (8, 8, 3)
    assert result[1] == 'RGB'

    # ------------------ test enhance with gray image---------------- #
    img = np.random.random((4, 4)).astype(np.float32)
    result = restorer.enhance(img, outscale=2)
    assert result[0].shape == (8, 8)
    assert result[1] == 'L'

    # ------------------ test enhance with RGBA---------------- #
    img = np.random.random((4, 4, 4)).astype(np.float32)
    result = restorer.enhance(img, outscale=2)
    assert result[0].shape == (8, 8, 4)
    assert result[1] == 'RGBA'

    # ------------------ test enhance with RGBA, alpha_upsampler---------------- #
    restorer.tile_size = 0
    img = np.random.random((4, 4, 4)).astype(np.float32)
    result = restorer.enhance(img, outscale=2, alpha_upsampler=None)
    assert result[0].shape == (8, 8, 4)
    assert result[1] == 'RGBA'
