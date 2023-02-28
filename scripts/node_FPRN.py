#!/usr/bin/env python3
import numpy as np
import rospy, cv2
import torch
import os, sys

import argparse
import time
import torch.nn as nn

import signal

from std_msgs.msg import Float32MultiArray        # See https://gist.github.com/jarvisschultz/7a886ed2714fac9f5226
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist

from agents.image_attack_agent import ImageAttacker
from setting_params import FREQ_MID_LEVEL, SETTING

import glob

from basicsr.utils.download_util import load_file_from_url


from FPRN import FPRNer
from FPRN.archs.srvgg_arch import SRVGGNetCompact

from FPRN.archs.srnet_arch import SRNet



IMAGE_CLEAN_RECEIVED = None
def fnc_clean_img_callback(msg):
    global IMAGE_CLEAN_RECEIVED
    IMAGE_CLEAN_RECEIVED = msg

IMAGE_RECEIVED = None
def fnc_img_callback(msg):
    global IMAGE_RECEIVED
    IMAGE_RECEIVED = msg

ADV_IMAGE_RECEIVED = None
def fnc_adv_img_callback(msg):
    global ADV_IMAGE_RECEIVED
    ADV_IMAGE_RECEIVED = msg


def get_args():
    """ Get arguments for individual tb3 deployment. """
    parser = argparse.ArgumentParser(
        description="Denoise a sequence with FPRN"
    )

    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='fprn-general-x4v3',
        help=('Model names: FPRN_x4plus | SRNet_x4plus | FPRN_x2plus | fprn-general-x4v3'))

    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument(
        '-dn',
        '--denoise_strength',
        type=float,
        default=0.5,
        help=('Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. '
              'Only used for the fprn-general-x4v3 model'))
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument(
        '--model_path', type=str, default=None, help='[Option] Model path. Usually, you do not need to specify it')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument(
        '--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='fprn',
        help='The upsampler for the alpha channels. Options: fprn | bicubic')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    parser.add_argument(
        '-g', '--gpu-id', type=int, default=None, help='gpu device to use (default=None) can be 0,1,2 for multi-gpu')

    return parser.parse_known_args(sys.argv)


if __name__ == '__main__':

    # rosnode node initialization
    rospy.init_node('FPRN_node')
    print('FPRN_node is initialized at', os.getcwd())
    start_time = time.time()
    args, unknown = get_args()

    # determine models according to model names
    args.model_name = args.model_name.split('.')[0]


    if args.model_name == 'FPRN_x4plus':#'RealESRGAN_x4plus':  # x4 SRNet model
        model = SRNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/haotiangu/FPRN/releases/download/FPRN/FPRN_x4plus.pth']
    elif args.model_name == 'SRNet_x4plus':#'RealESRNet_x4plus':  # x4 SRNet model
        model = SRNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/haotiangu/FPRN/releases/download/FPRN/SRNet_x4plus.pth']
    elif args.model_name == 'FPRN_x2plus':#'RealESRGAN_x2plus':  # x2 SRNet model
        model = SRNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/haotiangu/FPRN/releases/download/FPRN/FPRN_x2plus.pth']
    elif args.model_name == 'fprn-general-x4v3':#'realesr-general-x4v3':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/haotiangu/FPRN/releases/download/FPRN/fprn-general-wdn-x4v3.pth',
            'https://github.com/haotiangu/FPRN/releases/download/FPRN/fprn-general-x4v3.pth'
        ]


    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
		      print('{}: {}'.format(p, v))

        # determine model paths
    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = os.path.join('weights', args.model_name + '.pth')
        if not os.path.isfile(model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            for url in file_url:
                # model_path will be updated
                model_path = load_file_from_url(
                    url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)
        # restorer
    dni_weight = None
    if args.model_name == 'fprn-general-x4v3' and args.denoise_strength != 1:
        wdn_model_path = model_path.replace('fprn-general-x4v3', 'fprn-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]
    upsampler = FPRNer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        gpu_id=args.gpu_id)

    if args.face_enhance:  # Use GFPGAN for face enhancement
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/haotiangu/FPRN/releases/download/FPRN/GFPGANv1.3.pth',
            upscale=args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)
    os.makedirs(args.output, exist_ok=True)

    mid_time = time.time()
    # subscriber init.
    sub_clean_image = rospy.Subscriber('/airsim_node/camera_frame', Image, fnc_clean_img_callback)
    sub_attacked_image  = rospy.Subscriber('/attack_generator_node/attacked_image', Image, fnc_img_callback)
    sub_adv_image = rospy.Subscriber('/attack_generator_node/perturbation_image', Image, fnc_adv_img_callback)
    # publishers init.
    pub_clean_image = rospy.Publisher('/fastdvdnet_node/clean_image', Image, queue_size=10)

    # Running rate
    rate=rospy.Rate(FREQ_MID_LEVEL)


    # a bridge from cv2 image to ROS image
    mybridge = CvBridge()

    error_count = 0
    n_iteration = 0

    seq_list = []

    ##############################
    ### Instructions in a loop ###
    ##############################

    while not rospy.is_shutdown():

        n_iteration += 1
        # Load the saved Model every 10 iteration
        # Image generation
        #if IMAGE_RECEIVED is not None and ADV_IMAGE_RECEIVED is not None and IMAGE_CLEAN_RECEIVED is not None:
        # TRY THE REAL_TIME PERFORMANCE
        if IMAGE_CLEAN_RECEIVED is not None:
            with torch.no_grad():
                # Get camera image
                np_clean_im = np.frombuffer(IMAGE_CLEAN_RECEIVED.data, dtype=np.uint8).reshape(IMAGE_CLEAN_RECEIVED.height, IMAGE_CLEAN_RECEIVED.width, -1)
                np_clean_im = np.array(np_clean_im)
                # print(np_clean_im.shape) 448*448*3
                # Get attacked image
                np_im = np.frombuffer(IMAGE_RECEIVED.data, dtype=np.uint8).reshape(IMAGE_RECEIVED.height, IMAGE_RECEIVED.width, -1)
                np_im = np.array(np_im)

                # Get visualized noise image
                np_adv_im = np.frombuffer(ADV_IMAGE_RECEIVED.data, dtype=np.uint8).reshape(ADV_IMAGE_RECEIVED.height, ADV_IMAGE_RECEIVED.width, -1)
                np_adv_im = np.array(np_adv_im)
                #print(np_adv_im.shape)#(448,448,3)

                if args.face_enhance:
                    _, _, output = face_enhancer.enhance(np_im, has_aligned=False, only_center_face=False, paste_back=True)
                    print(output.shape)
                else:
                    output, _ = upsampler.enhance(np_im, outscale=args.outscale)
                    print(output.shape)

                adv_frame = mybridge.cv2_to_imgmsg(output)
                pub_clean_image.publish(adv_frame)
                mid_start_time = time.time()

                seq_time = time.time()


            stop_time = time.time()
            runtime = (stop_time - seq_time)

        try:
            experiment_done_done = rospy.get_param('experiment_done')
        except:
            experiment_done_done = False
        if experiment_done_done and n_iteration > FREQ_MID_LEVEL*3:
            rospy.signal_shutdown('Finished 100 Episodes!')

        rate.sleep()
