# -*- coding: utf-8 -*-
"""
Spyder Editor

This is Unet Architecture


"""
# Importing the dependency library first

import torch
import torch.nn as nn  # https://pytorch.org/docs/stable/nn.functional.html
 
# the above link will provide all the required information likes of convolution 
 
# Lets build the class for unet archietcture :)



def conv_struct(in_channels, out_channels):
    
     
   # Since it is sequential way operation layer can be treated as follow
    
    conv_layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3)
                               ,nn.ReLU(inplace=True)
                               ,nn.Conv2d(out_channels, out_channels, kernel_size=3)
                               ,nn.ReLU(inplace=True)
                               )
                               
                               
    #conv_layer.add_module(nn.Conv2d(in_channels, out_channels, kernel_size=3))
   # conv_layer.add_module(nn.ReLU(inplace=True)) # Each cov layer is followed by Relu unit
    
   # *****  https://discuss.pytorch.org/t/whats-the-difference-between-nn-relu-and-nn-relu-inplace-true/948
  
    #conv_layer.add_module(nn.Conv2d(in_channels, out_channels, kernel_size=3))
    #conv_layer.add_module(nn.ReLU(inplace=True))
    
    return conv_layer

def crop_image(input_image,output_image): # two size of tensor is given 
    
    target_image = output_image.size()[3]  # Target size is checked 
    #print(target_image)
    input_image1  = input_image.size()[3]  # Input size is checked 
   # print(input_image1)
    difference = input_image1-target_image # Difference is calculated 
    
    difference = difference//2  # This is to get acucrate divison 
   # print(difference)
    input_image = input_image[:,:,difference:input_image1-difference,difference:input_image1-difference]
    
    # The above will one size, one channel, and width, height of input image after croppping which
    # will be used for cocatnation
    
    return input_image




class unet(nn.Module):    # Importing the Module from pytorch
    
    def __init__(self):
        
        super(unet,self).__init__()
        
        # Lets see how it can be made, in the unet architecture we have
        # Convolution operation ----->activation function----> Maxpooling 
        # Offocure strides             
        
        self.max_pool = nn.MaxPool2d(kernel_size=2,stride=2)       
    
        # Lets define the convolution operation network where we can give how many 
        #time convolution is done and with what size and what padding is used for it
        # as we now there are two convolution layer each followed by our great 
        # activation fucntion called Relu 
        
        # Foolowing a Unet architecture 
        self.Dsample_conv_1 = conv_struct(1, 64) 
        # Dsample means as we see unet architecture it follows top to down and down to top approach
        # so Dsample can be treated as Downsampling option and following the unet architecture we have
        self.Dsample_conv_2 = conv_struct(64, 128)
        self.Dsample_conv_3 = conv_struct(128, 256)
        self.Dsample_conv_4 = conv_struct(256, 512)
        self.Dsample_conv_5 = conv_struct(512, 1024)
        
        ############################################################################
        
        self.upsample_conv_1 = nn.ConvTranspose2d(in_channels=1024, out_channels= 512, kernel_size=2,stride=2)     
        self.up_sample_conv_1 = conv_struct(1024, 512)
        
        
        self.upsample_conv_3 = nn.ConvTranspose2d(in_channels=512, out_channels= 256, kernel_size=2,stride=2)     
        self.up_sample_conv_3 = conv_struct(512,256)
        
        self.upsample_conv_4 = nn.ConvTranspose2d(in_channels=256, out_channels= 128, kernel_size=2,stride=2)     
        self.up_sample_conv_4 = conv_struct(256, 128)
        
        self.upsample_conv_5 = nn.ConvTranspose2d(in_channels=128, out_channels= 64, kernel_size=2,stride=2)     
        self.up_sample_conv_5 = conv_struct(128, 64)
        
    
        
        self.out = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)
        
        
        
        
        
# Lets now define the encoder block which will help in extracting the required feature from the input

    
    def forward(self,image):
        
        # Encoder
        
        conv_input_1 = self.Dsample_conv_1(image) # First input is sent in  
        # This top layer is connected to Right Leg of U as in paper mentioned 
        # It can be called as a skip connection where in feature from one network is 
        # added to othe rlayer of the network
        # in a similar way some downsample architecture network is connected to Upsample network
        # In the below whereever ** required is mentioned that can be used for skip connection
        
        conv_Maxpool_1 = self.max_pool(conv_input_1)   # Now appply max pooling
        
        conv_input_2   = self.Dsample_conv_2(conv_Maxpool_1)  #  **Required
        conv_Maxpool_2 = self.max_pool(conv_input_2)
        
        conv_input_3   = self.Dsample_conv_3(conv_Maxpool_2)   # **Required
        conv_Maxpool_3 = self.max_pool(conv_input_3)
        
        conv_input_4  = self.Dsample_conv_4(conv_Maxpool_3)   # **Required
        conv_Maxpool_4 = self.max_pool(conv_input_4)
        
        conv_input_5  = self.Dsample_conv_5(conv_Maxpool_4)
        #conv_Maxpool_5= self.max_pool(conv_input_5)
        
        
        
        
        
        # Decoder Network
##############################################################################################
        conv_back_prop = self.upsample_conv_1(conv_input_5)  # Here we start with decoder part
        croped_image = crop_image(conv_input_4,conv_back_prop)
        
        # Croppping of the data is done that is image and then finally concatenated to get require shape
        # of the image, same process is followed over othe rlayer of architecture
        
        conv_back_prop = self.up_sample_conv_1(torch.cat([conv_back_prop,croped_image],1))
       
 ##############################################################################################
       
        conv_back_prop_1 = self.upsample_conv_3(conv_back_prop)  # Here we start with decoder part
        croped_image_1 = crop_image(conv_input_3,conv_back_prop_1)
        
        conv_back_prop_1 = self.up_sample_conv_3(torch.cat([conv_back_prop_1,croped_image_1],1))
        
###############################################################################################
        conv_back_prop_2 = self.upsample_conv_4(conv_back_prop_1)  # Here we start with decoder part
        croped_image_2 = crop_image(conv_input_2,conv_back_prop_2)
               
        conv_back_prop_2 = self.up_sample_conv_4(torch.cat([conv_back_prop_2,croped_image_2],1))
###############################################################################################
      
        conv_back_prop_3 = self.upsample_conv_5(conv_back_prop_2)  # Here we start with decoder part
        croped_image_3 = crop_image(conv_input_1,conv_back_prop_3)
             
        conv_back_prop_3 = self.up_sample_conv_5(torch.cat([conv_back_prop_3,croped_image_3],1))
        
        conv_back_prop_4 = self.out(conv_back_prop_3)
        print(conv_back_prop_4.size())
        return conv_back_prop_4
        
        
###############################################################################################     
        
    
    
# Lets check the function 

if __name__=='__main__':
    
    image = torch.randn((1, 1, 572, 572))    
    model = unet()
    print(model(image))
    
    
    

    



    
    


