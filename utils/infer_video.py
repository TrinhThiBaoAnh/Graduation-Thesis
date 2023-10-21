# loading in and transforming data
import os
import torch
import torch.nn as nn
import imageio
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, jaccard_score
from PIL import Image
import cv2
from Encoder import mit
from Decoder import mlp
from mmcv.cnn import ConvModule
from skimage import img_as_ubyte
from PIL import Image, ImageSequence

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
def video_to_pil_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    transform = transforms.Compose([
                transforms.Resize((init_trainsize, init_trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])])
    frames = []
    original_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        original_frames.append(frame)
        # print(frame.shape)
        if not ret:
            break

        # Convert OpenCV BGR frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_frame = Image.fromarray(frame_rgb).convert('RGB')
        # print("Original Shape:", pil_frame.size)
        
        pil_frame = transform(pil_frame).unsqueeze(0)
        frames.append(pil_frame)

    cap.release()
    return frames, original_frames
def create_gif(image_list, output_gif_path, duration=100, loop=0):
    # Convert each image in the list to a Pillow Image object
    images = [Image.fromarray(image) for image in image_list]

    # Save the images as frames in the GIF file
    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop
    )
def create_gif_2(image_list, mask_list, output_gif_path, duration=100, loop=0):
    filter_images = []
    # print(len(mask_list))
    for i in range(len(image_list)):
        image = image_list[i]
        pred = mask_list[i]
        # print(type(pred))
        print(image.size)
        print(pred.size)
        contours, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
        filter_images.append(image)

    filter_images = [Image.fromarray(filter_image) for filter_image in filter_images]
    filter_images[0].save(
        output_gif_path,
        save_all=True,
        append_images=filter_images[1:],
        duration=duration,
        loop=loop
    )
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
def visualize_predictions(video_path):
    predicted_files = []
    model_path =  './SaveModel/{}_LA_{:1d}'.format(_model_name,1)
    ESFPNetBest = torch.load(model_path + '/ESFPNet.pt')
    ESFPNetBest.eval()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print('Models moved to GPU.')
    else:
        print('Only CPU available.')
    
    output_frames, original_frames = video_to_pil_frames(video_path)
    # gt_shape = original_frames[0].size
    res = []
    # print(gt_shape)
    print("Split frames successfully!")
    for i in range(len(output_frames)):
        image = output_frames[i]
        image = image.cuda()
        pred = ESFPNetBest(image)
        pred = F.upsample(pred, size=(1080, 1344), mode='bilinear', align_corners=False)
        pred = pred.sigmoid()
        threshold = torch.tensor([0.5]).to(device)
        pred = (pred > threshold).float() * 1
        pred = pred.data.cpu().numpy().squeeze()
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        pred = pred * 255
        tmp_img = original_frames[i]
        print(tmp_img.shape)
        print(pred.shape)
        image_with_masks = overlay(tmp_img, pred, color=(0,255,0), alpha=0.3)
        # painted_image = cv2.addWeighted(tmp_img.astype(np.float32), 1 - alpha, pred.astype(np.float32), alpha, 0, dtype=cv2.CV_8U)
        # painted_image = painted_image.astype(np.uint8)
        res.append(image_with_masks)
    output_gif_path = 'output.gif'
    output_gif_path_2 = 'output_2.gif'
    duration = 100  # 100 milliseconds per frame (adjust as needed)
    loop = 0  # Infinite looping (change to a positive integer for a fixed number of loops)
    imageio.mimsave(output_gif_path, res, duration=0.5) 
    # res = [Image.fromarray(image) for image in res]
    # res[0].save(
    #     output_gif_path_2,
    #     save_all=True,
    #     append_images=res[1:],
    #     duration=duration,
    #     loop=loop
    # )
    # create_gif(predicted_files, output_gif_path, duration, loop)
    # create_gif_2(original_frames, predicted_files, output_gif_path_2, duration, loop)
    print(f"GIF created at {output_gif_path}")

    return 0

if __name__ == "__main__":
    # Clear GPU cache
    torch.cuda.empty_cache()
    # configuration
    WholeDatasetName = 'UTDD'
    model_type = 'B0'
    _model_name = 'ESFP_{}_Endo_{}'.format(model_type,WholeDatasetName)
    init_trainsize = 352
    batch_size = 8
    # Paths to folders containing masks and predicted images
    video_path = "/home/baoanh/baoanh/DATN/demo/core/output_clips/clip_0.mp4"
    visualize_predictions(video_path)
