# Reference : CNN + RNN - Concatenate time distributed CNN with LSTM (https://discuss.pytorch.org/t/solved-concatenate-time-distributed-cnn-with-lstm/15435/2)

import os
import sys
import argparse
import cv2 as cv

import numpy as np
# np.random.seed(42)

import datetime

import random
# random.seed(42)

import torch

# torch.manual_seed(42)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

# torch.cuda.manual_seed(42)
# torch.cuda.manual_seed_all(42)

import torch.nn as nn
from torch import device
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF

from sequential_sensor_dataset import sequential_sensor_dataset

from CNN_RNN import CNN_RNN

import matplotlib.pyplot as plt

from tqdm import tqdm
import copy

os.environ['KMP_WARNINGS'] = 'off'

# Argument parser boolean processing (https://eehoeskrap.tistory.com/521)
def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')

ap = argparse.ArgumentParser()

ap.add_argument('-l', '--input_lidar_file_path', type=str, required=True)
ap.add_argument('-i', '--input_img_file_path', type=str, required=True)
ap.add_argument('-p', '--input_pose_file_path', type=str, required=True)

ap.add_argument('-c', '--cuda_num', type=str, required=True)

ap.add_argument('-e', '--training_epoch', type=int, required=False, default=100)
ap.add_argument('-b', '--batch_size', type=int, required=False, default=16)
ap.add_argument('-s', '--sequence_length', type=int, required=False, default=5)
ap.add_argument('-d', '--data_display', type=str2bool, required=False, default=False)

ap.add_argument('-m', '--execution_mode', type=str, required=True, default='training')
ap.add_argument('-t', '--pre_trained_network_path', type=str, required=True)

args = vars(ap.parse_args())

device = torch.device('cuda:' + args['cuda_num'] if torch.cuda.is_available() else 'cpu')
print(device)

train_preprocess = transforms.Compose([
    transforms.Resize((192, 640)),
    transforms.CenterCrop((192, 640)),
    transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter()]), p=0.55),
    transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(kernel_size=3)]), p=0.55),
    # transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    # transforms.RandomErasing(p=0.3),
])

valid_preprocess = transforms.Compose([
    transforms.Resize((192, 640)),
    transforms.CenterCrop((192, 640)),
    transforms.ToTensor(),
])

test_preprocess = transforms.Compose([
    transforms.Resize((192, 640)),
    transforms.CenterCrop((192, 640)),
    transforms.ToTensor(),
])

DATA_DISPLAY_ON = args['data_display']

EPOCH = args['training_epoch']

batch_size = args['batch_size']

sequence_length = args['sequence_length']

mode = args['execution_mode']

training_sequence = ['00', '01', '03', '05', '06', '08', '10']
valid_sequence = ['02', '04', '07', '09']
test_sequence = ['02']

train_dataset = sequential_sensor_dataset(lidar_dataset_path=args['input_lidar_file_path'], 
                                    img_dataset_path=args['input_img_file_path'], 
                                    pose_dataset_path=args['input_pose_file_path'],
                                    train_sequence=training_sequence, 
                                    valid_sequence=valid_sequence, 
                                    test_sequence=test_sequence,
                                    sequence_length=sequence_length,
                                    train_transform=train_preprocess,
                                    valid_transform=valid_preprocess,
                                    test_transform=test_preprocess,
                                    mode='training',)

valid_dataset = sequential_sensor_dataset(lidar_dataset_path=args['input_lidar_file_path'], 
                                    img_dataset_path=args['input_img_file_path'], 
                                    pose_dataset_path=args['input_pose_file_path'],
                                    train_sequence=training_sequence, 
                                    valid_sequence=valid_sequence, 
                                    test_sequence=test_sequence,
                                    sequence_length=sequence_length,
                                    train_transform=train_preprocess,
                                    valid_transform=valid_preprocess,
                                    test_transform=test_preprocess,
                                    mode='validation')

test_dataset = sequential_sensor_dataset(lidar_dataset_path=args['input_lidar_file_path'], 
                                    img_dataset_path=args['input_img_file_path'], 
                                    pose_dataset_path=args['input_pose_file_path'],
                                    train_sequence=training_sequence, 
                                    valid_sequence=valid_sequence, 
                                    test_sequence=test_sequence,
                                    sequence_length=sequence_length,
                                    train_transform=train_preprocess,
                                    valid_transform=valid_preprocess,
                                    test_transform=test_preprocess,
                                    mode='test',)

start_time = str(datetime.datetime.now())

training_shuffle = True
validation_shuffle = True
evaluation_shuffle = False

if mode == 'training':

    # CNN_type = 'new'
    CNN_type = 'mobilenetV3_large'
    # CNN_type = 'vgg16'
    CNN_freeze = False

    hidden_size = 1000
    num_layers = 2

    gradient_clip = 0

    regression = 'full_sequence'

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=training_shuffle, num_workers=10, drop_last=True, collate_fn=train_dataset.collate_fn, prefetch_factor=10, persistent_workers=False, pin_memory=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=validation_shuffle, num_workers=10, drop_last=True, collate_fn=valid_dataset.collate_fn, prefetch_factor=10, persistent_workers=False, pin_memory=True)

    print('Mode : Training')
    print('Training Epoch : ' + str(EPOCH))
    print('Batch Size : ' + str(batch_size))
    print('Sequence Length : ' + str(sequence_length))

    train_learning_rate = 0.001

    CRNN_VO_model = CNN_RNN(device=device, cnn_type=CNN_type, cnn_freeze=CNN_freeze, rnn_type='lstm', bidirection='False', regression=regression, input_sequence_length=sequence_length, hidden_size=hidden_size, num_layers=num_layers, learning_rate=train_learning_rate)

    # Tensorboard run command : tensorboard --logdir=./runs
    training_writer = SummaryWriter(log_dir='./runs/' + start_time + '/CRNN_VO_training')
    validation_writer = SummaryWriter(log_dir='./runs/' + start_time + '/CRNN_VO_validation')

    plot_step_training = 0
    plot_step_validation = 0

    for epoch in range(EPOCH):

        print('[EPOCH : {}]'.format(str(epoch)))
        
        ### Training ####################################################################
        CRNN_VO_model.train()

        if epoch == 0:
            if os.path.exists('./' + start_time) == False:
                print('Creating save directory')
                os.mkdir('./' + start_time)

                hyperparam_file = open('./' + start_time + '/CRNN_VO_param.txt', 'w')
                hyperparam_file.write('batch_size : {} \n'.format(args['batch_size']))
                hyperparam_file.write('sequence_length : {} \n'.format(args['sequence_length']))
                hyperparam_file.write('training_sequence : {} \n'.format(training_sequence))
                hyperparam_file.write('valid_sequence : {} \n'.format(valid_sequence))
                hyperparam_file.write('test_sequence : {} \n'.format(test_sequence))
                hyperparam_file.write('train_learning_rate : {} \n'.format(train_learning_rate))
                hyperparam_file.write('gradient_clip : {} \n'.format(gradient_clip))
                hyperparam_file.write('training preprocess : {} \n'.format(train_preprocess))
                hyperparam_file.write('validation preprocess : {} \n'.format(valid_preprocess))
                hyperparam_file.write('regression : {} \n'.format(regression))
                hyperparam_file.close()

        print('Current State [Training] - [EPOCH : {}]'.format(str(epoch)))

        for batch_idx, (current_seq, current_img_tensor, pose_6DOF_tensor) in enumerate(tqdm(train_dataloader)):

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
                train_loss = CRNN_VO_model.translation_loss(pose_est_output[:, :3], pose_6DOF_tensor[:, :3])
                train_loss.backward()
                if gradient_clip != 0:
                    torch.nn.utils.clip_grad_norm_(CRNN_VO_model.parameters(), gradient_clip)
                CRNN_VO_model.optimizer.step()

                training_writer.add_scalar('Immediate Loss (Translation) | CNN Type : {} | CNN Freeze : {} | Batch Size : {} | Sequence Length : {} | Hidden Size : {} | RNN Num : {} | Clip : {} | Regression : {} | Learning Rate : {} | Optimizer : {} | Shuffle : {}'.format(CNN_type, CNN_freeze,batch_size, sequence_length, hidden_size, num_layers, gradient_clip, regression, train_learning_rate, type(CRNN_VO_model.optimizer), training_shuffle), train_loss.item(), plot_step_training)
                plot_step_training += 1

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


        ### Validation ###############################################################
        CRNN_VO_model.eval()
    
        valid_translation_loss = nn.MSELoss()

        print('Current State [Validation] - [EPOCH : {}]'.format(str(epoch)))

        with torch.no_grad():

            for batch_idx, (current_seq, current_img_tensor, pose_6DOF_tensor) in enumerate(tqdm(valid_dataloader)):

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

                    valid_loss = valid_translation_loss(pose_est_output[:, :3], pose_6DOF_tensor[:, :3])

                    validation_writer.add_scalar('Immediate Loss (Translation) | CNN Type : {} | CNN Freeze : {} | Batch Size : {} | Sequence Length : {} | Hidden Size : {} | RNN Num : {} | Clip : {} | Regression : {} | Learning Rate : {} | Optimizer : {} | Shuffle : {}'.format(CNN_type, CNN_freeze,batch_size, sequence_length, hidden_size, num_layers, gradient_clip, regression, train_learning_rate, type(CRNN_VO_model.optimizer), training_shuffle), valid_loss.item(), plot_step_validation)
                    plot_step_validation += 1

        torch.save({
            'epoch' : EPOCH,
            'sequence_length' : sequence_length,
            'CRNN_VO_model' : copy.deepcopy(CRNN_VO_model.state_dict()),
            'optimizer' : copy.deepcopy(CRNN_VO_model.optimizer.state_dict()),
        }, './' + start_time + '/CRNN_VO_model.pt')

        torch.save(CRNN_VO_model, './' + start_time + '/CRNN_VO_full_model.pt')

elif mode == 'test':

    CNN_type = 'mobilenetV3_large'
    # CNN_type = 'vgg16'
    CNN_freeze = False

    hidden_size = 1000
    num_layers = 2

    regression = 'full_sequence'

    test_batch_size = 1

    test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=evaluation_shuffle, num_workers=4, drop_last=True, collate_fn=test_dataset.collate_fn, prefetch_factor=20, persistent_workers=True)
   
    print('Mode : Test')
    print('Batch Size : ' + str(test_batch_size))
    print('Sequence Length : ' + str(sequence_length))

    test_learning_rate = 0.001

    CRNN_VO_model = CNN_RNN(device=device, cnn_type=CNN_type, cnn_freeze=CNN_freeze, rnn_type='lstm', bidirection='False', regression=regression, input_sequence_length=sequence_length, hidden_size=hidden_size, num_layers=num_layers, learning_rate=test_learning_rate)

    checkpoint = torch.load(args['pre_trained_network_path'], map_location='cuda:' + args['cuda_num'])

    CRNN_VO_model.load_state_dict(checkpoint['CRNN_VO_model'])

    CRNN_VO_model.eval()

    test_writer = SummaryWriter(log_dir='./runs/' + start_time + '/CRNN_VO_test', flush_secs=1)
    plot_step_test = 0

    current_seq_num = 0
    prev_seq_num = 0

    path_x = 0
    path_z = 0

    groundtruth_x = 0
    groundtruth_z = 0

    vo_est_file = open('./CRNN_VO_est.txt', 'w')
    vo_groundtruth_file = open('./CRNN_groundtruth.txt', 'w')

    with torch.no_grad():

        for batch_idx, (current_seq, current_img_tensor, pose_6DOF_tensor) in enumerate(tqdm(test_dataloader)):

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

                test_loss = CRNN_VO_model.translation_loss(pose_est_output[:, :3], pose_6DOF_tensor[:, :3]) \
                            # + translation_rotation_relative_weight * CRNN_VO_model.rotation_loss(pose_est_output[:, 3:], pose_6DOF_tensor[:, 3:])

                test_writer.add_scalar('[Test] Immediate Loss (Translation) | Batch Size : {} | Sequence Length : {}'.format(test_batch_size, sequence_length), test_loss.item(), plot_step_test)
                plot_step_test += 1

                current_seq_num = current_seq

                # Detect sequence change for reset plotting
                if (batch_idx != 0) and (current_seq_num != prev_seq_num):

                    pass

                else:

                    VO_est = pose_est_output.cpu().numpy()
                    VO_groundtruth = pose_6DOF_tensor.cpu().numpy()
                    
                    trajectory_x_est = []
                    trajectory_z_est = []
                    
                    trajectory_x_truth = []
                    trajectory_z_truth = []

                    for i in range(len(VO_est)):

                        path_x += VO_est[:, 0][i]
                        trajectory_x_est.append(path_x)

                        path_z += VO_est[:, 2][i]
                        trajectory_z_est.append(path_z)

                    path_x = trajectory_x_est[-1]
                    path_z = trajectory_z_est[-1]

                    for i in range(len(VO_groundtruth)):

                        groundtruth_x += VO_groundtruth[:, 0][i]
                        trajectory_x_truth.append(groundtruth_x)

                        groundtruth_z += VO_groundtruth[:, 2][i]
                        trajectory_z_truth.append(groundtruth_z)

                    groundtruth_x = trajectory_x_truth[-1]
                    groundtruth_z = trajectory_z_truth[-1]

                    np.savetxt(vo_est_file, np.dstack((np.array(trajectory_x_est), np.array(trajectory_z_est)))[0])
                    np.savetxt(vo_groundtruth_file, np.dstack((np.array(trajectory_x_truth), np.array(trajectory_z_truth)))[0])

                prev_seq_num = current_seq_num
