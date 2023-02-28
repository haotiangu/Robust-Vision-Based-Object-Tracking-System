# :european_castle: Model Zoo

- [For General Images](#for-general-images)

---

## For General Images

| Models                                                                                                                          | Scale | Description                                  |
| ------------------------------------------------------------------------------------------------------------------------------- | :---- | :------------------------------------------- |
| [FPRN_x4plus](https://github.com/haotiangu/FPRN/releases/download/FPRN/FPRN_x4plus.pth)                      | X4    | X4 model for general images                  |              |
| [FPRN_x2plus](https://github.com/haotiangu/FPRN/releases/download/FPRN/FPRN_x2plus.pth)                      | X2    | X2 model for general images                  |
| [SRNet_x4plus](https://github.com/haotiangu/FPRN/releases/download/FPRN/SRNet_x4plus.pth)                      | X4    | X4 model with MSE loss (over-smooth effects) |
| [official SRNET_x4](https://github.com/haotiangu/FPRN/releases/download/FPRN/SRNET_SRx4_DF2K.pth) | X4    | ppretrained srnet model                        |
| [fprn-general-x4v3](https://github.com/haotiangu/FPRN/releases/download/FPRN/fprn-general-x4v3.pth) | X4 (can also be used for X1, X2, X3) | A tiny small model (consume much fewer GPU memory and time); not too strong deblur and denoise capacity |

The following models are **discriminators**, which are usually used for fine-tuning.

| Models                                                                                                                 | Corresponding model |
| ---------------------------------------------------------------------------------------------------------------------- | :------------------ |
| [FPRN_x4plus_netD](https://github.com/haotiangu/FPRN/releases/download/FPRN/FPRN_x4plus_netD.pth) | FPRN_x4plus   |
| [FPRN_x2plus_netD](https://github.com/haotiangu/FPRN/releases/download/FPRN/FPRN_x2plus_netD.pth) | FPRN_x2plus   |


Note: <br>
<sup>1</sup> This model can also be used for X1, X2, X3.

The following models are **discriminators**, which are usually used for fine-tuning.

