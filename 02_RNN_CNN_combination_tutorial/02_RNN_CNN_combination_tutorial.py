import os
import argparse
import cv2 as cv
import numpy as np
from torch import device

import torch
from torch.utils.data import DataLoader

from torchvision import transforms
import torchvision.transforms.functional as TF

from sequential_sensor_dataset import sequential_sensor_dataset

from CNN_RNN import CNN_RNN

ap = argparse.ArgumentParser()

ap.add_argument('-l', '--input_lidar_file_path', type=str, required=True)
ap.add_argument('-i', '--input_img_file_path', type=str, required=True)
ap.add_argument('-p', '--input_pose_file_path', type=str, required=True)

args = vars(ap.parse_args())

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

preprocess = transforms.Compose([
    transforms.Resize((192, 640)),
    transforms.CenterCrop((192, 640)),
    transforms.ToTensor(),
])

batch_size = 1

dataset = sequential_sensor_dataset(lidar_dataset_path=args['input_lidar_file_path'], 
                                    img_dataset_path=args['input_img_file_path'], 
                                    pose_dataset_path=args['input_pose_file_path'],
                                    train_sequence=['00', '01', '02'], valid_sequence=['01'], test_sequence=['02'],
                                    sequence_length=5,
                                    train_transform=preprocess,
                                    valid_transform=preprocess,
                                    test_transform=preprocess,)

dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

CRNN_VO_model = CNN_RNN(device=device)
CRNN_VO_model.train()

dataloader.dataset.mode = 'training'
for batch_idx, (current_img_tensor, pose_6DOF_tensor) in enumerate(dataloader):

    if (current_img_tensor != []) and (pose_6DOF_tensor != []):

        current_img_tensor = current_img_tensor.to(device).float()
        pose_6DOF_tensor = pose_6DOF_tensor.to(device).float()

        # Data Dimension Standard : Batch Size x Sequence Length x Data Shape
        # Sequential Image = Batch Size x Sequence Length x 3 (Channel) x 376 (Height) x 1241 (Width)
        # Sequential Pose = Batch Size x Sequence Length x 6 (6 DOF)

        # print('---------------------------------')
        # print(current_img_tensor.size())
        # print(pose_6DOF_tensor.size())

        pose_est_output = CRNN_VO_model(current_img_tensor)

        translation_rotation_relative_weight = 100

        CRNN_VO_model.optimizer.zero_grad()
        train_loss = CRNN_VO_model.translation_loss(pose_est_output[:, :, :3], pose_6DOF_tensor[:, :, :3]) \
                     + translation_rotation_relative_weight * CRNN_VO_model.rotation_loss(pose_est_output[:, :, 3:], pose_6DOF_tensor[:, :, 3:])
        train_loss.backward()
        CRNN_VO_model.optimizer.step()

        print('[Total Loss : {}]'.format(train_loss.data))

        ### Sequential Image Stack Display ###
        disp_current_img_tensor = current_img_tensor.clone().detach().cpu()

        img_sequence_list = []
        total_img = []
        seq_len = dataloader.dataset.sequence_length
        for batch_index in range(disp_current_img_tensor.size(0)):

            for seq in range(dataloader.dataset.sequence_length):
                current_img = np.array(TF.to_pil_image(disp_current_img_tensor[batch_index][seq]))
                current_img = cv.cvtColor(current_img, cv.COLOR_RGB2BGR)    # Re-Order the image array into BGR for display purpose
                current_img = cv.resize(current_img, dsize=(int(1280/seq_len), int(240/(seq_len * 0.5))), interpolation=cv.INTER_CUBIC)

                img_sequence_list.append(current_img)

            total_img.append(cv.hconcat(img_sequence_list))
            img_sequence_list = []
        
        final_img_output = cv.vconcat(total_img)

        cv.imshow('Image Sequence Stack', final_img_output)
        cv.waitKey(1)