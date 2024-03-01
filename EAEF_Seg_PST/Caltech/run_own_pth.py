import os
import argparse
import time
import datetime
import sys
import shutil
import stat
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from util.caltech_dataset import caltech_dataset
from util.util import compute_results, visualize
from sklearn.metrics import confusion_matrix
from model.EAEFNet import EAEFNet
import cv2
#############################################################################################
parser = argparse.ArgumentParser(description='Test with pytorch')
#############################################################################################
parser.add_argument('--model_name', '-m', type=str, default='EAEFNet')
parser.add_argument('--weight_name', '-w', type=str,
                    default='EAEFNet50_hight_iou_2')
parser.add_argument('--file_name', '-f', type=str, default='best.pth')
parser.add_argument('--dataset_split', '-d', type=str,
                    default='test')  # test, test_day, test_night
parser.add_argument('--gpu', '-g', type=int, default=0)
#############################################################################################
parser.add_argument('--img_height', '-ih', type=int, default=608)
parser.add_argument('--img_width', '-iw', type=int, default=960)
parser.add_argument('--num_workers', '-j', type=int, default=0)
parser.add_argument('--n_class', '-nc', type=int, default=5)
parser.add_argument('--data_dir', '-dr', type=str, default='./test/')
args = parser.parse_args()
#############################################################################################

if __name__ == '__main__':
    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')
    model_dir = os.path.join('./runs/', args.weight_name)
    if os.path.exists(model_dir) is False:
        sys.exit("the %s does not exit." % (model_dir))
    model_file = os.path.join(model_dir, args.file_name)
    if os.path.exists(model_file) is True:
        print('use the final model file.')
    else:
        sys.exit('no model file found.')
    print('testing %s: %s on GPU #%d with pytorch' %
          (args.model_name, args.weight_name, args.gpu))

    conf_total = np.zeros((args.n_class, args.n_class))
    model = EAEFNet.EAEFNet(args.n_class)
    if args.gpu >= 0:
        model.cuda(args.gpu)
    print('loading model file %s... ' % model_file)
    pretrained_weight = torch.load(
        model_file, map_location=lambda storage, loc: storage.cuda(args.gpu))
    own_state = model.state_dict()

    for name, param in pretrained_weight.items():
        if name not in own_state:
            continue
        own_state[name].copy_(param)
    print('done!')

    for name, param in pretrained_weight.items():
        if name not in own_state:
            print(name)
            continue
        own_state[name].copy_(param)
    print('done!')

    batch_size = 1  # do not change this parameter!
    test_dataset = caltech_dataset(
        data_dir=args.data_dir, split='test')

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    ave_time_cost = 0.0

    model.eval()
    with torch.no_grad():
        for it, (images, thermal, labels, names) in enumerate(test_loader):
            # if it > 0:
            #     break
            images = Variable(images).cuda(args.gpu)
            thermal = Variable(thermal).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            images = torch.cat([images, thermal], dim=1)
            start_time = time.time()
            logit, logits = model(images)
            end_time = time.time()
            if it >= 5:  # # ignore the first 5 frames
                ave_time_cost += (end_time - start_time)
            # convert tensor to numpy 1d array
            label = labels.cpu().numpy().squeeze().flatten()
            print(max(label[label < 10]))
            # prediction and label are both 1-d array, size: minibatch*640*480
            prediction = logits.argmax(1).cpu().numpy().squeeze().flatten()
            # generate confusion matrix frame-by-frame
            # conf is an n_class*n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf = confusion_matrix(
                y_true=label, y_pred=prediction, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            conf_total += conf

            predicted_tensor = logits.argmax(1).unsqueeze(1)
            scale = max(1, 255 // args.n_class)
            predicted_tensor = logits.argmax(1).unsqueeze(1) * scale
            predicted_tensor = torch.cat(
                (predicted_tensor, predicted_tensor, predicted_tensor), 1)
            pred_img = predicted_tensor[0, :, :, :].cpu().numpy()
            cv2.imwrite(os.path.join(
                "./results", names[0]), np.swapaxes(np.swapaxes(pred_img, 0, 2), 1, 0))
            print(names[0])
            # save demo images
            # visualize(image_name=names, predictions=logits.argmax(
            #     1), weight_name='Pred_' + args.weight_name)
            # print("%s, %s, frame %d/%d, %s, time cost: %.2f ms, demo result saved."
            #       % (
            #           args.model_name, args.weight_name, it +
            #           1, len(test_loader), names,
            #           (end_time - start_time) * 1000))

    precision_per_class, recall_per_class, iou_per_class = compute_results(
        conf_total)
    conf_total_matfile = os.path.join(
        'D:/pst900_thermal_rgb-master/RTFNet/runs/Pred_' + args.weight_name, 'conf_' + args.weight_name + '.mat')

    print('\n###########################################################################')
    print('\n%s: %s test results (with batch size %d) on %s using %s:' % (
        args.model_name, args.weight_name, batch_size, datetime.date.today(), torch.cuda.get_device_name(args.gpu)))
    print('\n* the tested dataset name: %s' % args.dataset_split)
    print('* the tested image count: %d' % len(test_loader))
    print('* the tested image size: %d*%d' % (args.img_height, args.img_width))
    print('* the weight name: %s' % args.weight_name)
    print('* the file name: %s' % args.file_name)
    print(
        "* recall per class: \n   0: %.6f,1: %.6f, 2: %.6f, 3: %.6f,  4: %.6f,5: %.6f,6: %.6f,7: %.6f,8: %.6f,9: %.6f"
        % (recall_per_class[0], recall_per_class[1], recall_per_class[2], recall_per_class[3], recall_per_class[4], recall_per_class[5], recall_per_class[6], recall_per_class[7], recall_per_class[8], recall_per_class[9]))
    print(
        "* iou per class: \n     0: %.6f,1: %.6f, 2: %.6f, 3: %.6f,  4: %.6f,5: %.6f,6: %.6f,7: %.6f,8: %.6f,9: %.6f"
        % (iou_per_class[0], iou_per_class[1], iou_per_class[2], iou_per_class[3], iou_per_class[4], iou_per_class[5], iou_per_class[6], iou_per_class[7], iou_per_class[8], iou_per_class[9]))
    print("\n* average values (np.mean(x)): \n recall: %.6f, iou: %.6f"
          % (recall_per_class.mean(), iou_per_class.mean()))
    print("* average values (np.mean(np.nan_to_num(x))): \n recall: %.6f, iou: %.6f"
          % (np.mean(np.nan_to_num(recall_per_class)), np.mean(np.nan_to_num(iou_per_class))))
    print(
        '\n* the average time cost per frame (with batch size %d): %.2f ms, namely, the inference speed is %.2f fps' % (
            batch_size, ave_time_cost * 1000 / (len(test_loader) - 5),
            1.0 / (ave_time_cost / (len(test_loader) - 5))))  # ignore the first 10 frames
    # print('\n* the total confusion matrix: ')
    # np.set_printoptions(precision=8, threshold=np.inf, linewidth=np.inf, suppress=True)
    # print(conf_total)
    print('\n###########################################################################')
