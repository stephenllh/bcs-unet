import torch
from benchmark.reconnet.net import ReconNet
from benchmark.scsnet.net import SCSNetInit, SCSNetDeep


def test_reconnet(sr, img_dim=32):
    num_measurements = int(sr * img_dim ** 2)
    net = ReconNet(num_measurements=num_measurements, img_dim=img_dim)
    batch_size = 2

    # CCS
    y_input = torch.randn((batch_size, 1, num_measurements))
    image_recon = net(y_input)
    assert image_recon.shape == (batch_size, 1, img_dim, img_dim)

    # BCS
    block_size = 4
    y_input = torch.randn(
        (batch_size, num_measurements // block_size ** 2, block_size, block_size)
    )
    image_recon = net(y_input)
    print(image_recon.min(), image_recon.max())
    assert image_recon.shape == (batch_size, 1, img_dim, img_dim)
    print("ReconNet: Passed all tests.")


def test_scsnet(sr, img_dim=32, block_size=4):
    """
    Test:
    1. Whether of not the permute function in `SCSNetInit` is correct.
    1. The output shape of the `SCSNetDeep`
    """

    if img_dim % block_size != 0:
        raise Exception("Image dimension is not divisible by block size.")

    measurements = int(sr * img_dim ** 2)
    num_blocks = int(img_dim // block_size)
    in_channels = int(measurements / num_blocks ** 2)

    net1 = SCSNetInit(in_channels=in_channels, block_size=block_size)
    net2 = SCSNetDeep()

    batch_size = 2

    # Test 1 for y_input: Check shape
    y_input = torch.randn((batch_size, in_channels, num_blocks, num_blocks))
    out1 = net1(y_input)
    print(out1.shape)
    assert out1.shape == (batch_size, 1, img_dim, img_dim)

    # Test 2 for y_input: Correct arrangement
    img_dim2 = 8
    num_blocks = int(img_dim2 // block_size)
    n = img_dim2 ** 2
    y_input2 = (
        torch.arange(1, n + 1)
        .view(num_blocks, num_blocks, block_size ** 2)
        .permute(2, 0, 1)
    )

    out2 = net1._permute(y_input2.unsqueeze(0))
    print(
        f"If {img_dim2}x{img_dim2} image and {block_size}x{block_size} block, then the output image = "
    )
    print(out2)

    # Check the shape of reconstructed image
    image_recon = net2(out1)
    if image_recon.shape != (batch_size, 1, img_dim, img_dim):
        raise Exception(
            "The shape of reconstructed image is not as equal as intended `img_dim`."
        )
    else:
        print("The shape of reconstructed image - Passed.")

    print("Passed all tests.")


def test_all():
    # test_reconnet(0.125)
    test_scsnet(sr=0.125, img_dim=96)


if __name__ == "__main__":
    test_all()
