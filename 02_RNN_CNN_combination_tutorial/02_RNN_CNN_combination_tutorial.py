# Reference : CNN + RNN - Concatenate time distributed CNN with LSTM (https://discuss.pytorch.org/t/solved-concatenate-time-distributed-cnn-with-lstm/15435/2)

import os
import sys
import argparse
import cv2 as cv
import numpy as np
import time

import torch
from torch import device
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
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

DATA_DISPLAY_ON = True

EPOCH = 100

batch_size = 4

sequence_length = 5

dataset = sequential_sensor_dataset(lidar_dataset_path=args['input_lidar_file_path'], 
                                    img_dataset_path=args['input_img_file_path'], 
                                    pose_dataset_path=args['input_pose_file_path'],
                                    train_sequence=['01'], valid_sequence=['01'], test_sequence=['02'],
                                    sequence_length=sequence_length,
                                    train_transform=preprocess,
                                    valid_transform=preprocess,
                                    test_transform=preprocess,)

dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True, collate_fn=dataset.collate_fn)

CRNN_VO_model = CNN_RNN(device=device, hidden_size=500, learning_rate=0.001)
CRNN_VO_model.train()

start_time = str(time.time())

writer = SummaryWriter(flush_secs=1)

plot_step = 0

dataloader.dataset.mode = 'training'
for epoch in range(EPOCH):

    print('[EPOCH : {}]'.format(str(epoch)))

    immediate_loss_list = []
    running_average_loss_list = []
    running_average_loss_length = 20

    if epoch == 0:
        if os.path.exists('./' + start_time) == False:
            print('Creating save directory')
            os.mkdir('./' + start_time)

    for batch_idx, (current_img_tensor, pose_6DOF_tensor) in enumerate(dataloader):

        if (current_img_tensor != None) and (pose_6DOF_tensor != None):

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
            train_loss = CRNN_VO_model.translation_loss(pose_est_output[:, :3], pose_6DOF_tensor[:, -1, :3]) \
                        + translation_rotation_relative_weight * CRNN_VO_model.rotation_loss(pose_est_output[:, 3:], pose_6DOF_tensor[:, -1, 3:])
            train_loss.backward()
            CRNN_VO_model.optimizer.step()

            writer.add_scalar('Immediate Loss', train_loss.data, plot_step)
            plot_step += 1

            if DATA_DISPLAY_ON is True:

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

    torch.save({
        'epoch' : EPOCH,
        'sequence_lenght' : sequence_length,
        'CRNN_VO_model' : CRNN_VO_model.state_dict(),
        'optimizer' : CRNN_VO_model.optimizer.state_dict(),
    }, './' + start_time)