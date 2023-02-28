import torch

from FPRN.archs.autoencoder_arch import Autoencoder


def test_autoencoder():
    """Test arch: Autoencoder."""

    # model init and forward (cpu)
    net = Autoencoder(num_in_ch=3, num_feat=4, skip_connection=True)
    img = torch.rand((1, 3, 32, 32), dtype=torch.float32)
    output = net(img)
    assert output.shape == (1, 1, 32, 32)

    # model init and forward (gpu)
    if torch.cuda.is_available():
        net.cuda()
        output = net(img.cuda())
        assert output.shape == (1, 1, 32, 32)
