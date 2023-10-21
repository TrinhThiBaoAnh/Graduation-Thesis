import streamlit as st
import sys
import cv2
import tempfile
sys.path.append("/home/baoanh/baoanh/DATN/ESFPNet")
import pandas as pd
from io import StringIO
# loading in and transforming data
import os
import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, jaccard_score
from PIL import Image
from Encoder import mit
from Decoder import mlp
import matplotlib.pyplot as plt
from mmcv.cnn import ConvModule
class ESFPNetStructure(nn.Module):

    def __init__(self, embedding_dim = 160):
        super(ESFPNetStructure, self).__init__()
        
        # Backbone
        if model_type == 'B0':
            self.backbone = mit.mit_b0()
        if model_type == 'B1':
            self.backbone = mit.mit_b1()
        if model_type == 'B2':
            self.backbone = mit.mit_b2()
        if model_type == 'B3':
            self.backbone = mit.mit_b3()
        if model_type == 'B4':
            self.backbone = mit.mit_b4()
        if model_type == 'B5':
            self.backbone = mit.mit_b5()
        
        self._init_weights()  # load pretrain
        
        # LP Header
        self.LP_1 = mlp.LP(input_dim = self.backbone.embed_dims[0], embed_dim = self.backbone.embed_dims[0])
        self.LP_2 = mlp.LP(input_dim = self.backbone.embed_dims[1], embed_dim = self.backbone.embed_dims[1])
        self.LP_3 = mlp.LP(input_dim = self.backbone.embed_dims[2], embed_dim = self.backbone.embed_dims[2])
        self.LP_4 = mlp.LP(input_dim = self.backbone.embed_dims[3], embed_dim = self.backbone.embed_dims[3])
        
        # Linear Fuse
        self.linear_fuse34 = ConvModule(in_channels=(self.backbone.embed_dims[2] + self.backbone.embed_dims[3]), out_channels=self.backbone.embed_dims[2], kernel_size=1,norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse23 = ConvModule(in_channels=(self.backbone.embed_dims[1] + self.backbone.embed_dims[2]), out_channels=self.backbone.embed_dims[1], kernel_size=1,norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse12 = ConvModule(in_channels=(self.backbone.embed_dims[0] + self.backbone.embed_dims[1]), out_channels=self.backbone.embed_dims[0], kernel_size=1,norm_cfg=dict(type='BN', requires_grad=True))
        
        # Fused LP Header
        self.LP_12 = mlp.LP(input_dim = self.backbone.embed_dims[0], embed_dim = self.backbone.embed_dims[0])
        self.LP_23 = mlp.LP(input_dim = self.backbone.embed_dims[1], embed_dim = self.backbone.embed_dims[1])
        self.LP_34 = mlp.LP(input_dim = self.backbone.embed_dims[2], embed_dim = self.backbone.embed_dims[2])
        
        # Final Linear Prediction
        self.linear_pred = nn.Conv2d((self.backbone.embed_dims[0] + self.backbone.embed_dims[1] + self.backbone.embed_dims[2] + self.backbone.embed_dims[3]), 1, kernel_size=1)
        
    def _init_weights(self):
        
        if model_type == 'B0':
            pretrained_dict = torch.load('./Pretrained/mit_b0.pth')
        if model_type == 'B1':
            pretrained_dict = torch.load('./Pretrained/mit_b1.pth')
        if model_type == 'B2':
            pretrained_dict = torch.load('./Pretrained/mit_b2.pth')
        if model_type == 'B3':
            pretrained_dict = torch.load('./Pretrained/mit_b3.pth')
        if model_type == 'B4':
            pretrained_dict = torch.load('./Pretrained/mit_b4.pth')
        if model_type == 'B5':
            pretrained_dict = torch.load('./Pretrained/mit_b5.pth')
            
            
        model_dict = self.backbone.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict)
        print("successfully loaded!!!!")
        
        
    def forward(self, x):
        
        ##################  Go through backbone ###################
        
        B = x.shape[0]
        
        #stage 1
        out_1, H, W = self.backbone.patch_embed1(x)
        for i, blk in enumerate(self.backbone.block1):
            out_1 = blk(out_1, H, W)
        out_1 = self.backbone.norm1(out_1)
        out_1 = out_1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  #(Batch_Size, self.backbone.embed_dims[0], 88, 88)
        
        # stage 2
        out_2, H, W = self.backbone.patch_embed2(out_1)
        for i, blk in enumerate(self.backbone.block2):
            out_2 = blk(out_2, H, W)
        out_2 = self.backbone.norm2(out_2)
        out_2 = out_2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  #(Batch_Size, self.backbone.embed_dims[1], 44, 44)
        
        # stage 3
        out_3, H, W = self.backbone.patch_embed3(out_2)
        for i, blk in enumerate(self.backbone.block3):
            out_3 = blk(out_3, H, W)
        out_3 = self.backbone.norm3(out_3)
        out_3 = out_3.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  #(Batch_Size, self.backbone.embed_dims[2], 22, 22)
        
        # stage 4
        out_4, H, W = self.backbone.patch_embed4(out_3)
        for i, blk in enumerate(self.backbone.block4):
            out_4 = blk(out_4, H, W)
        out_4 = self.backbone.norm4(out_4)
        out_4 = out_4.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  #(Batch_Size, self.backbone.embed_dims[3], 11, 11)
        
        # go through LP Header
        lp_1 = self.LP_1(out_1)
        lp_2 = self.LP_2(out_2)  
        lp_3 = self.LP_3(out_3)  
        lp_4 = self.LP_4(out_4)
        
        # linear fuse and go pass LP Header
        lp_34 = self.LP_34(self.linear_fuse34(torch.cat([lp_3, F.interpolate(lp_4,scale_factor=2,mode='bilinear', align_corners=False)], dim=1)))
        lp_23 = self.LP_23(self.linear_fuse23(torch.cat([lp_2, F.interpolate(lp_34,scale_factor=2,mode='bilinear', align_corners=False)], dim=1)))
        lp_12 = self.LP_12(self.linear_fuse12(torch.cat([lp_1, F.interpolate(lp_23,scale_factor=2,mode='bilinear', align_corners=False)], dim=1)))
        
        # get the final output
        lp4_resized = F.interpolate(lp_4,scale_factor=8,mode='bilinear', align_corners=False)
        lp3_resized = F.interpolate(lp_34,scale_factor=4,mode='bilinear', align_corners=False)
        lp2_resized = F.interpolate(lp_23,scale_factor=2,mode='bilinear', align_corners=False)
        lp1_resized = lp_12
        
        out = self.linear_pred(torch.cat([lp1_resized, lp2_resized, lp3_resized, lp4_resized], dim=1))
        
        return out
def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def predict(type_dataset, image_path, testsize):
    # Step 1:Preprocessing
    transform = transforms.Compose([
            transforms.Resize((testsize, testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
    # gt_transform = transforms.ToTensor()
    image = rgb_loader(image_path)
    w,h = image.size 
    image = transform(image).unsqueeze(0)
    # print(image.shape)
    # gt = binary_loader(mask_path)
    # gt = gt_transform(gt)
    # gt = np.asarray(gt, np.float32)
    # gt /= (gt.max() + 1e-8)
    image = image.cuda()

    # Step 2:Prediction
    pred = ESFPNetBest(image)
    pred = F.upsample(pred, size=(h,w), mode='bilinear', align_corners=False)
    # if type_dataset=="LOETHTT":
    #     pred = F.upsample(pred, size=(1024, 1280), mode='bilinear', align_corners=False)
    # elif type_dataset=="HPDUONG":
    #     pred = F.upsample(pred, size=(994, 1280), mode='bilinear', align_corners=False)
    # else:
    #     pred = F.upsample(pred, size=(995, 1280), mode='bilinear', align_corners=False)
    pred = pred.sigmoid()
    threshold = torch.tensor([0.5]).to(device)
    pred = (pred > threshold).float() * 1
    pred = pred.data.cpu().numpy().squeeze()
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    # gt_image = Image.fromarray((gt * 255).astype(np.uint8))
    # pred_image = Image.fromarray((pred * 255).astype(np.uint8))
    return (pred * 255).astype(np.uint8)

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp + 1e-6)

def calculateMetrics(mask_file, predicted_file):
    mask = np.array(mask_file)
    pred = np.array(predicted_file)
    # print(mask)
    # print(pred)
    true_values = mask.flatten()
    predicted_values = pred.flatten()
    true_values = np.array(true_values) > 0.5
    predicted_values = np.array(predicted_values) > 0.5
    # Calculate metrics
    precision = precision_score(true_values, predicted_values)
    recall = recall_score(true_values, predicted_values)
    f1 = f1_score(true_values, predicted_values)
    miou = np.mean(jaccard_score(true_values, predicted_values, average=None))
    mspecificity = specificity_score(true_values, predicted_values)
    print("mDice:", f1)
    print("mIoU:", miou)
    print("Recall:", recall)
    print("Precision:", precision)
    print("mSpecificity:", mspecificity)
from PIL import Image
import numpy as np

def overlay(image, mask, color, alpha, resize=None):
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
    return image_combined


if __name__ == "__main__":
    st.markdown(
    "<h1 style='text-align: center; color: #FF5733; font-size: 30px;'>"
    "LESION SEGMENTATION IN GASTROINTESTINAL ENDOSCOPY IMAGES"
    "</h1>",
    unsafe_allow_html=True
)

    option = st.selectbox(
        'Choose type of dataset?',
        ('Gastric cancer', 'Esophageal cancer', 'Peptic ulcer', 'Positive H. pylori gastritis', 'Negative H. pylori gastritis'))

    # st.write('You selected:', option)
    if option == 'Gastric cancer':
        WholeDatasetName = 'UTDD'
    elif option == 'Esophageal cancer':
        WholeDatasetName = 'UTTQ'
    elif option == 'Peptic ulcer':
        WholeDatasetName = 'LOETHTT'
    elif option == 'Positive H. pylori gastritis':
        WholeDatasetName = 'HPDUONG'
    else:
        WholeDatasetName = 'HPAM'
    # Create a file uploader widget
     # model_path =  '/home/baoanh/baoanh/DATN/ESFPNet/SaveModel/{}_LA_{:1d}'.format(_model_name,numIters)
    ESFPNetBest = torch.load(f"/home/baoanh/baoanh/DATN/ESFPNet/SaveModel/ESFP_B0_Endo_{WholeDatasetName}_LA_1" + '/ESFPNet.pt')
        
    ESFPNetBest.eval()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print('Models moved to GPU.')
    else:
        print('Only CPU available.')
    def binary_loader( path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    # tab1, tab2, tab3, tab4, tab5 = st.tabs(["Gastric cancer", "Esophageal cancer", "Peptic ulcer", "Positive H. pylori gastritis", "Negative H. pylori gastritis"])

    # with tab1:
    st.header(option)
    # WholeDatasetName = "UTDD"
    torch.cuda.empty_cache()
    model_type = 'B0'
    # _model_name = 'ESFP_{}_Endo_{}'.format(model_type,WholeDatasetName)
    init_trainsize = 352
    uploaded_images = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # Check if an image is uploaded
    if uploaded_images:
        with tempfile.TemporaryDirectory() as temp_dir:
            for uploaded_image in uploaded_images:
                image_name = uploaded_image.name
                # print(image_path)
                image_path = os.path.join(temp_dir, image_name)
                # print(image_path)
                with open(image_path, "wb") as f:
                    f.write(uploaded_image.read())
                import time
                start = time.time()
                predicted_file = predict(WholeDatasetName, image_path, init_trainsize )
                end = time.time()
                # print("Predicted time: ", end- start)
                # st.write("Predicted time: ", end- start)
                process_time = (end- start)*1000
                
                # print(predicted_file.shape)
                pil_image = Image.open(uploaded_image)
                np_image = np.array(pil_image)
                col1, col2, col3 = st.columns(3)
                # print(np_image.shape)
                # print(predicted_file.shape)
                image_with_masks = overlay(np_image, predicted_file, color=(0,255,0), alpha=0.3)
                with col1:
                    st.markdown("<h2 style='text-align: center;font-size: 35px;'>Image</h2>", unsafe_allow_html=True)
                    st.image(image_path,  use_column_width=True)

                # Display images in the second column
                with col2:
                    st.markdown("<h2 style='text-align: center;font-size: 35px;'>Prediction</h2>", unsafe_allow_html=True)
                    st.image(predicted_file,  use_column_width=True)
                # Display images in the first column
                with col3:
                    st.markdown("<h2 style='text-align: center;font-size: 35px;'>Overlay Image</h2>", unsafe_allow_html=True)
                    st.image(image_with_masks, use_column_width=True)
                st.subheader(f'Predicted time: :blue[{process_time}ms]', divider='rainbow')
            else:
                st.warning("Please upload images.")
    