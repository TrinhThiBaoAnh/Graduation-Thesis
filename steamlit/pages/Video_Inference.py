"""Media streamings"""
import logging
from pathlib import Path
from typing import Dict, Optional, cast

import av
import cv2
import streamlit as st
from aiortc.contrib.media import MediaPlayer
from streamlit_webrtc import WebRtcMode, WebRtcStreamerContext, webrtc_streamer

from sample_utils.download import download_file
from sample_utils.turn import get_ice_servers
import tempfile
import sys
import cv2
import tempfile
import pandas as pd
from io import StringIO
# loading in and transforming data
import os
import torch
import time
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
sys.path.append("/home/baoanh/baoanh/DATN/ESFPNet")
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
# Function to handle video upload
def handle_video_upload(file):
    video_path = UPLOAD_FOLDER / file.name
    with open(video_path, "wb") as f:
        f.write(file.read())
    return video_path
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
def predict(image):
    # print("Predicting....", type(image))
    image = Image.fromarray(np.uint8(image))
    # Convert the image to RGB mode (if it's not already in RGB mode)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    w,h = image.size 
    image = transform(image).unsqueeze(0)
    image = image.cuda()
    pred = ESFPNetBest(image)
    pred = F.interpolate(pred, size=(h,w), mode='bilinear', align_corners=False)
    pred = pred.sigmoid()
    threshold = torch.tensor([0.5]).to(device)
    pred = (pred > threshold).float() * 1
    pred = pred.data.cpu().numpy().squeeze()
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    return (pred * 255).astype(np.uint8)
def create_player():
    print("Creating player...")
    if "local_file_path" in media_file_info:
        return MediaPlayer(str(media_file_info["local_file_path"]))
    else:
        return MediaPlayer(media_file_info["url"])
    
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    start_infer_time = time.time()
    pred = predict(img)
    end_infer_time = time.time()
    inference_time = end_infer_time - start_infer_time
    output_frame =  overlay(img, pred, color=(0,255,0), alpha=0.3)
    # fps = 1.0 / inference_time
    cv2.putText(output_frame, f"Inference Time: {inference_time:.2f} s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # cv2.putText(output_frame, f"FPS: {fps:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return av.VideoFrame.from_ndarray(output_frame, format="bgr24")

if __name__ == "__main__":

    HERE = Path(__file__).parent
    ROOT = HERE.parent
    logger = logging.getLogger(__name__)

    # Create a temporary directory to store uploaded video
    temp_dir = tempfile.TemporaryDirectory()
    UPLOAD_FOLDER = Path(temp_dir.name)
    
    torch.cuda.empty_cache()
    model_type = 'B0'
    init_trainsize = 352
    # File uploader widget
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    video_path = ""
    if uploaded_file is not None:
        # st.write("Uploading video file ....")
        # start_time = time.time()
        video_path = handle_video_upload(uploaded_file)   
        MEDIAFILES: Dict[str, Dict] = {
            "Endoscopic Video": {
                "local_file_path": video_path,
                "type": "video",
            },
        }
        media_file_label = st.radio("Select a media source to stream", tuple(MEDIAFILES.keys()))
        media_file_info = MEDIAFILES[cast(str, media_file_label)]
        option = st.selectbox(
                'Choose type of dataset?',
                ('Gastric cancer', 'Esophageal cancer'))
        WholeDatasetName = 'UTDD'
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
        # Khai báo một biến để theo dõi thời điểm bắt đầu của video
        
        ESFPNetBest = torch.load(f"/home/baoanh/baoanh/DATN/ESFPNet/SaveModel/ESFP_B0_Endo_{WholeDatasetName}_LA_1" + '/ESFPNet.pt')
        ESFPNetBest.eval()
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print('Models moved to GPU.')
        else:
            print('Only CPU available.')
        st.header(option)

        transform = transforms.Compose([
                    transforms.Resize((init_trainsize, init_trainsize)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])])
        key = f"media-streaming-{media_file_label}"
        ctx: Optional[WebRtcStreamerContext] = st.session_state.get(key)
        
        webrtc_streamer(
            key=key,
            mode=WebRtcMode.RECVONLY,
            rtc_configuration={"iceServers": get_ice_servers()},
            media_stream_constraints={
                "video": media_file_info["type"] == "video",
                "audio": media_file_info["type"] == "audio",
            },
            player_factory=create_player,
            video_frame_callback=video_frame_callback,
        )
    else:
        st.warning("Please upload the video file.")

