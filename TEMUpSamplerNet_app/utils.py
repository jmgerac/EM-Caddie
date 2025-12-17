import torch
from torchvision.transforms import ToTensor
import numpy as np
import torchvision
import importlib
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from model import Net
import torch.nn as nn
import os
"""Model 0 : irdm_pth"""
"""Model 1 : /home/student/Documents/u-net_pytorch/epochs200_layer3_ori_256/"""
"""Model 2 : /home/student/Documents/u-net-pytorch-original/lr001_weightdecay00001/"""
"""Model 3 : /home/student/Documents/u-net_denoising/dataset_small_mask/"""
"""Model 4 : /home/student/Documents/Atom Segmentation APP/AtomSegGUI/atomseg_bupt_new_10/"""
"""Model 5 : /home/student/Documents/Atom Segmentation APP/AtomSegGUI/atomseg_bupt_new_100/"""
"""Model 6 : /home/student/Documents/Atom Segmentation APP/AtomSegGUI/atom_seg_gaussian_mask/"""


def load_model(model_path, data, scale_factor, cuda):

	if os.path.basename(model_path) == "upinterpolation_superresolution_gen1.pth":
		from unet_standard import NestedUNet
		net = NestedUNet()
	elif os.path.basename(model_path) == "Gen1-noNoiseNoBackgroundUpinterpolation2x.pth" or \
			os.path.basename(model_path) == "Gen1-noNoiseUpinterpolation2x.pth" or \
		os.path.basename(model_path) == "Gen1-noNoiseNoBackgroundSuperresolution.pth":
		from mypackage.model.unet_standard2x import NestedUNet
		net = NestedUNet()
	else:
		net = Net(upscale_factor = scale_factor)

	if cuda:
		net = net.cuda()


	# if not cuda:
	# 	net.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
	# else:
	# 	net.load_state_dict(torch.load(model_path))
	if os.path.basename(model_path) == "upinterpolation_superresolution_gen1.pth" or \
			os.path.basename(model_path) == "Gen1-noNoiseNoBackgroundUpinterpolation2x.pth" or \
			os.path.basename(model_path) == "Gen1-noNoiseUpinterpolation2x.pth" or \
			os.path.basename(model_path) == "Gen1-noNoiseNoBackgroundSuperresolution.pth":
		if cuda:
			net = nn.DataParallel(net)
			net.load_state_dict(torch.load(model_path))
		else:
			net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path).items()})
	else:
		if os.name == 'posix':
			if cuda:
				net = torch.load(model_path)
			else:
				net = torch.load(model_path, map_location=torch.device('cpu'))
		else:
			from functools import partial
			import pickle
			pickle.load = partial(pickle.load, encoding="latin1")
			pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
			net = torch.load(model_path, map_location=lambda storage, loc: storage, pickle_module=pickle)

	transform = ToTensor()
	ori_tensor = transform(data)
	ori_tensor = torch.unsqueeze(ori_tensor,0)

	padding_left = 0
	padding_right = 0
	padding_top = 0
	padding_bottom = 0
	ori_height = ori_tensor.size()[2]
	ori_width = ori_tensor.size()[3]
	use_padding = False
	if ori_width>=ori_height:
		padsize = ori_width
	else:
		padsize = ori_height
	if np.log2(padsize) >= 7.0:
		if (np.log2(padsize)-7) % 1 > 0:
			padsize = 2**(np.log2(padsize)//1+1)
	else:
		padsize = 2**7
	if ori_height<padsize:
		padding_top = int(padsize - ori_height) // 2
		padding_bottom =int(padsize - ori_height - padding_top)
		use_padding = True
	if ori_width < padsize:
		padding_left = int(padsize - ori_width) // 2
		padding_right = int(padsize - ori_width - padding_left)
		use_padding = True
	if use_padding:
		padding_transform = torch.nn.ConstantPad2d((padding_left, \
												   padding_right, \
												   padding_top, \
												   padding_bottom), 0)
		ori_tensor = padding_transform(ori_tensor)


	if cuda:
		ori_tensor = ori_tensor.cuda()
	output = net(ori_tensor)

	if cuda:
		result = (output.data).cpu().numpy()
	else:
		result = (output.data).numpy()


	padsize = int(padsize)
	result = result[0,0,2*padding_top:2*(padsize-padding_bottom),2*padding_left:2*(padsize-padding_right)]
	return result


def PIL2Pixmap(im):
    """Convert PIL image to QImage """
    if im.mode == "RGB":
        pass
    elif im.mode == "L":
        im = im.convert("RGBA")
    data = im.convert("RGBA").tobytes("raw", "RGBA")
    qim = QtGui.QImage(data, im.size[0], im.size[1], QtGui.QImage.Format_ARGB32)
    pixmap = QtGui.QPixmap.fromImage(qim)
    return pixmap

def map01(mat):
    return (mat - mat.min())/(mat.max() - mat.min())
