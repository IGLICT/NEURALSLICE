import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class resnet_block(nn.Module):
	def __init__(self, dim_in, dim_out):
		super(resnet_block, self).__init__()
		self.dim_in = dim_in
		self.dim_out = dim_out
		if self.dim_in == self.dim_out:
			self.conv_1 = nn.Conv2d(self.dim_in, self.dim_out, 3, stride=1, padding=1, bias=False)
			self.conv_2 = nn.Conv2d(self.dim_out, self.dim_out, 3, stride=1, padding=1, bias=False)
			nn.init.xavier_uniform_(self.conv_1.weight)
			nn.init.xavier_uniform_(self.conv_2.weight)
		else:
			self.conv_1 = nn.Conv2d(self.dim_in, self.dim_out, 3, stride=2, padding=1, bias=False)
			self.conv_2 = nn.Conv2d(self.dim_out, self.dim_out, 3, stride=1, padding=1, bias=False)
			self.conv_s = nn.Conv2d(self.dim_in, self.dim_out, 1, stride=2, padding=0, bias=False)
			nn.init.xavier_uniform_(self.conv_1.weight)
			nn.init.xavier_uniform_(self.conv_2.weight)
			nn.init.xavier_uniform_(self.conv_s.weight)

	def forward(self, input, is_training=False):
		if self.dim_in == self.dim_out:
			output = self.conv_1(input)
			output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
			output = self.conv_2(output)
			output = output+input
			output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
		else:
			output = self.conv_1(input)
			output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
			output = self.conv_2(output)
			input_ = self.conv_s(input)
			output = output+input_
			output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
		return output


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.img_ef_dim = 64
        self.z_dim = 1024
        self.img_ef_dim = 64
        self.z_dim = 1024
        self.conv_0 = nn.Conv2d(3, self.img_ef_dim, 7, stride=2, padding=3, bias=False)
        self.res_1 = resnet_block(self.img_ef_dim, self.img_ef_dim)
        self.res_2 = resnet_block(self.img_ef_dim, self.img_ef_dim)
        self.res_3 = resnet_block(self.img_ef_dim, self.img_ef_dim )
        self.res_4 = resnet_block(self.img_ef_dim , self.img_ef_dim)
        self.res_5 = resnet_block(self.img_ef_dim , self.img_ef_dim )
        self.res_6 = resnet_block(self.img_ef_dim, self.img_ef_dim )
        self.res_7 = resnet_block(self.img_ef_dim, self.img_ef_dim )
        self.conv_7 = nn.Conv2d(self.img_ef_dim , self.img_ef_dim , 4, stride=1, padding=1, bias=True)
        self.conv_8 = nn.Conv2d(self.img_ef_dim , self.img_ef_dim , 4, stride=1, padding=1, bias=True)
        self.plane1 = nn.Conv2d(self.img_ef_dim , self.img_ef_dim //2, 4, stride=1, padding=0, bias=True)
        self.plane2 = nn.Conv2d(self.img_ef_dim , self.img_ef_dim //2, 4, stride=1, padding=0, bias=True)
        self.plane3 = nn.Conv2d(self.img_ef_dim , self.img_ef_dim//2 , 4, stride=1, padding=0, bias=True)
        nn.init.xavier_uniform_(self.conv_0.weight)
        nn.init.xavier_uniform_(self.conv_7.weight)
        nn.init.constant_(self.conv_7.bias, 0)
        nn.init.xavier_uniform_(self.conv_8.weight)
        nn.init.constant_(self.conv_8.bias, 0)
        nn.init.xavier_uniform_(self.plane1.weight)
        nn.init.constant_(self.plane1.bias, 0)
        nn.init.xavier_uniform_(self.plane2.weight)
        nn.init.constant_(self.plane2.bias, 0)        
        nn.init.xavier_uniform_(self.plane3.weight)
        nn.init.constant_(self.plane3.bias, 0)
        
    def forward(self, view, is_training=True):
    
        layer_0 = self.conv_0( view)
        layer_0 = F.leaky_relu(layer_0, negative_slope=0.01, inplace=True)

        layer_1 = self.res_1(layer_0, is_training=is_training)
        layer_2 = self.res_2(layer_1, is_training=is_training)

        layer_3 = self.res_3(layer_2, is_training=is_training)
        layer_4 = self.res_4(layer_3, is_training=is_training)

        layer_5 = self.res_5(layer_4, is_training=is_training)
        layer_6 = self.res_6(layer_5, is_training=is_training)
        layer_6 = self.res_7(layer_6, is_training=is_training)

        layer_7 = self.conv_7(layer_6)
        layer_7 = F.leaky_relu(layer_7, negative_slope=0.01, inplace=True)

        layer_8 = self.conv_8(layer_7)
        layer_8 = F.leaky_relu(layer_8, negative_slope=0.01, inplace=True)

        plane1 = self.plane1(layer_8)
        plane2 = self.plane2(layer_8)
        plane3 = self.plane3(layer_8)

        return plane1, plane2, plane3