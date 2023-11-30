import os
import argparse
import torch
import time
import pickle
import numpy as np

from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import VideoAnomalyDataset_C3D, VideoAnomalyDataset_C3D_NM, VideoAnomalyDataset_C3D_AM
from models import model
from utils import _get_grads, l2_between_dicts

from tqdm import tqdm
from aggregate import remake_video_output, evaluate_auc, remake_video_3d_output
import random
from torchvision import transforms
import cv2


torch.backends.cudnn.benchmark = False
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

# Config
def get_configs():
    parser = argparse.ArgumentParser(description="VAD-Jigsaw config")
    parser.add_argument("--val_step", type=int, default=1)
    parser.add_argument("--print_interval", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gpu_id", type=str, default=1)
    parser.add_argument("--log_date", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--static_threshold", type=float, default=0.3)
    parser.add_argument("--sample_num", type=int, default=7)
    parser.add_argument("--filter_ratio", type=float, default=0.5)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--test_checkpoint", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="shanghaitech", choices=['shanghaitech', 'ped2', 'avenue'])
    args = parser.parse_args()

    args.device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
    if args.dataset in ['shanghaitech', 'avenue']:
        args.filter_ratio = 0.8
    elif args.dataset == 'ped2':
        args.filter_ratio = 0.5
    return args


def train(args):
    torch.cuda.set_device(args.device)
    if not args.log_date:
        running_date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    else:
        running_date = args.log_date
    print("The running_data : {}".format(running_date))
    for k, v in vars(args).items():
        print("-------------{} : {}".format(k, v))

    # Load Data
    data_dir = f"F:/{args.dataset}/testing/frames"
    detect_pkl = f'detect/{args.dataset}_test_detect_result_yolov3.pkl'

    vad_tgt_dataset = VideoAnomalyDataset_C3D(data_dir,
                                              dataset=args.dataset,
                                              detect_dir=detect_pkl,
                                              fliter_ratio=args.filter_ratio,
                                              frame_num=args.sample_num,
                                              static_threshold=args.static_threshold)

    vad_tgt_dataloader = DataLoader(vad_tgt_dataset, batch_size=64, shuffle=True, num_workers=0,
                                    pin_memory=True)

    # data_dir = f"F:/{args.dataset}/testing/frames"
    # detect_pkl = f'detect/{args.dataset}_test_detect_result_yolov3.pkl'
    #
    # vad_tgt2_dataset = VideoAnomalyDataset_C3D(data_dir,
    #                                           dataset=args.dataset,
    #                                           detect_dir=detect_pkl,
    #                                           fliter_ratio=args.filter_ratio,
    #                                           frame_num=args.sample_num,
    #                                           static_threshold=args.static_threshold)
    #
    # vad_tgt2_dataloader = DataLoader(vad_tgt2_dataset, batch_size=32, shuffle=True, num_workers=0,
    #                                 pin_memory=True)
    vad_src_dataset = VideoAnomalyDataset_C3D_NM('D:/UBnormal_frames/training/frames',
                                                 dataset='ubnormal',
                                                 detect_dir='./detect/ubnormal_train_detect_result_yolov3.pkl',
                                                 fliter_ratio=0.5,
                                                 frame_num=args.sample_num,
                                                 static_threshold=args.static_threshold)
    vad_src_nm_dataloader = DataLoader(vad_src_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2,
                                       pin_memory=True)

    vad_src_dataset = VideoAnomalyDataset_C3D_AM('D:/UBnormal_frames/training/frames',
                                                 dataset='ubnormal',
                                                 detect_dir='./detect/ubnormal_train_detect_result_yolov3.pkl',
                                                 fliter_ratio=0.5,
                                                 frame_num=args.sample_num,
                                                 static_threshold=args.static_threshold)
    vad_src_am_dataloader = DataLoader(vad_src_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2,
                                       pin_memory=True)

    net = model.WideBranchNet(time_length=args.sample_num, num_classes=[args.sample_num ** 2, 81])
    classifier = model.discriminator()
    adaparams = model.Adaparams()
    mapping = model.MappingNetwork()
    tempo_classifier = model.pretext_classifier()


    if args.checkpoint is not None:
        state = torch.load(args.checkpoint)
        state_keys = list(state.keys())
        keys_idx = 0
        print('load ' + args.checkpoint)
        for _, child in net.named_children():
            for sub_name, sub_child in child.named_children():
                try:
                    sub_child.weight.data = state[state_keys[keys_idx]]
                    keys_idx += 1
                except Exception:
                    continue
        # net.cuda()
        # classifier.cuda()
        # smoothed_auc, smoothed_auc_avg, _ = val(args, net)
        # exit(0)
    if args.test_checkpoint is not None:
        state = torch.load(args.test_checkpoint)
        print('load:'+args.test_checkpoint)
        net.load_state_dict(state['net_state_dict'],strict=True)
        mapping.load_state_dict(state['mapping_state_dict'],strict=True)
        classifier.load_state_dict(state['classifier_state_dict'],strict=True)
        tempo_classifier.load_state_dict(state['temp_classifier_state_dict'],strict=True)
        net.cuda(args.device)
        mapping.cuda(args.device)
        classifier.cuda(args.device)
        tempo_classifier.cuda(args.device)
        adaparams.cuda(args.device)
        smoothed_auc, smoothed_auc_avg, smoothed_auc_, smoothed_auc_avg_, temp_timestamp = val(args, net, mapping,classifier, adaparams,tempo_classifier)
        exit(0)


    net.cuda(args.device)
    net = net.train()
    classifier.cuda(args.device)
    classifier = classifier.train()
    adaparams.cuda(args.device)
    adaparams = adaparams.train()
    mapping.cuda(args.device)
    mapping = mapping.train()
    tempo_classifier.cuda(args.device)
    tempo_classifier = tempo_classifier.train()

    criterion = nn.CrossEntropyLoss(reduction='mean')
    mse_criterion = nn.MSELoss()
    optimizer = optim.Adam(params=list(net.parameters())+list(classifier.parameters()), lr=1e-5)
    optimizer_adaparams = optim.Adam(params=adaparams.parameters(), lr=1e-6)
    optimizer_mapping = optim.Adam(params=list(mapping.parameters())+list(tempo_classifier.parameters()), lr=1e-6)

    # Train
    log_dir = './log/{}/'.format(running_date)
    writer = SummaryWriter(log_dir)

    t0 = time.time()
    global_step = 0

    max_acc = -1
    max_acc_= -1
    timestamp_in_max = None

    iter_num = 1

    len_target = len(vad_tgt_dataloader) - 1
    iter_target = iter(vad_tgt_dataloader)

    # len_target2 = len(vad_tgt2_dataloader) - 1
    # iter_target2 = iter(vad_tgt2_dataloader)

    len_src_am = len(vad_src_am_dataloader) - 1
    iter_src_am = iter(vad_src_am_dataloader)

    for epoch in range(args.epochs):
        for it, data in enumerate(vad_src_nm_dataloader):
            if iter_num % len_src_am == 0:
                iter_src_am = iter(vad_src_am_dataloader)
            if iter_num % len_target == 0:
                iter_target = iter(vad_tgt_dataloader)


            data_src_am = iter_src_am.next()
            obj_am, cls_labels_am = data_src_am['obj'].cuda(args.device, non_blocking=True), data_src_am[
                'cls_label'].cuda(args.device, non_blocking=True)

            obj_nm, cls_labels_nm, obj_js, temp_labels, spat_labels, t_flag = data['obj'].cuda(args.device, non_blocking=True), data['cls_label'].cuda(
                args.device, non_blocking=True), data['obj_js'].cuda(args.device, non_blocking=True), data['label'], data["trans_label"], data["temporal"]

            temp_labels = temp_labels[t_flag].long().view(-1).cuda(args.device)

            obj_am = obj_am[:len(obj_nm)]
            cls_labels_am = cls_labels_am[:len(cls_labels_nm)]

            obj = torch.cat([obj_nm, obj_am, obj_js], dim=0).cuda(args.device)
            cls_labels = torch.cat([cls_labels_nm, cls_labels_am], dim=0).long().cuda(args.device)

            z_ori, z_aug = net.fea1(obj)
            z_ori, z_aug = net.fea2(z_ori,z_aug)
            z_ori = net.fea_forward(z_ori)
            z_aug = net.fea_forward(z_aug)

            z_js = z_ori[-len(obj_js):]

            temp_logits = tempo_classifier(z_js)
            temp_logits = temp_logits.view(-1, args.sample_num)
            loss_js = criterion(temp_logits, temp_labels)

            loss_reg = mse_criterion(adaparams(z_aug[:-len(obj_js)] - z_ori[:-len(obj_js)]), torch.zeros_like(z_aug[:-len(obj_js)]))
            loss_cls = F.cross_entropy(classifier(z_ori[:-len(obj_js)]),cls_labels) + F.cross_entropy(classifier(z_aug[:-len(obj_js)]),cls_labels)
            loss = loss_reg + loss_cls + loss_js*0.05

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            z_ori, z_aug = net.fea1(obj)
            z_ori, z_aug = net.fea2(z_ori, z_aug)
            z_ori, z_aug = net.fea_forward(z_ori), net.fea_forward(z_aug)

            loss_reg = mse_criterion(adaparams(z_aug[:-len(obj_js)] - z_ori[:-len(obj_js)]), torch.zeros_like(z_aug[:-len(obj_js)]))
            loss_cls = F.cross_entropy(classifier(z_ori[:-len(obj_js)]),cls_labels) + F.cross_entropy(classifier(z_aug[:-len(obj_js)]),cls_labels)

            dict_reg = _get_grads(optimizer, net, loss_reg)
            dict_cls = _get_grads(optimizer, net, loss_cls)
            # dict_js = _get_grads(optimizer, net, loss_js)
            penalty = l2_between_dicts(dict_reg, dict_cls, normalize=True)

            optimizer_adaparams.zero_grad()
            penalty.backward()
            optimizer_adaparams.step()

            writer.add_scalar('Train/Loss', loss.item(), global_step=global_step)
            writer.add_scalar('Train/cls_loss', loss_cls.item(), global_step=global_step)
            writer.add_scalar('Train/reg_loss', loss_reg.item(), global_step=global_step)
            writer.add_scalar('Train/js_loss', loss_js.item(), global_step=global_step)
            writer.add_scalar('Train/penalty', penalty.item(), global_step=global_step)

            if (global_step + 1) % args.print_interval == 0:
                print("[{}:{}/{}]\tloss: {:.6f}  Cls_loss: {:.6f}  Reg_loss: {:.6f}  Js_loss: {:.6f}  Penalty: {:.6f}   \ttime: {:.6f}". \
                      format(epoch, it + 1, len(vad_src_nm_dataloader), loss.item(), loss_cls.item(),
                             loss_reg.item(), loss_js.item(), penalty.item(),
                             time.time() - t0))
                t0 = time.time()

            global_step += 1
            iter_num += 1

            if global_step % 50 == 0:
                for it, data_tgt in enumerate(vad_tgt_dataloader):
                    # if iter_num % len_target2 == 0:
                    #     iter_target2 = iter(vad_tgt2_dataloader)
                    # data_tgt2 = iter_target2.next()
                    obj_tgt, temp_labels, spat_labels, t_flag, obj_tgt_js = data_tgt['obj'].cuda(args.device,non_blocking=True), data_tgt['label'], data_tgt['trans_label'], data_tgt['temporal'], data_tgt['obj_js'].cuda(args.device)
                    # obj_tgt = torch.cat([obj_tgt,data_tgt2['obj'].cuda(args.device)],dim=0).cuda(args.device)
                    obj_tgt = torch.cat([obj_tgt, obj_tgt_js], dim=0).cuda(args.device)

                    z_ori, z_aug = net.fea1(obj_tgt)
                    z_ori, z_aug = mapping.fea1(z_ori), mapping.fea1(z_aug)
                    z_ori, z_aug = net.fea2(z_ori, z_aug)
                    z_ori, z_aug = mapping.fea2(z_ori), mapping.fea2(z_aug)
                    z_ori, z_aug = net.fea3(z_ori), net.fea3(z_aug)
                    z_ori, z_aug = mapping.fea3(z_ori), mapping.fea3(z_aug)
                    z_ori, z_aug = net.flat(z_ori), net.flat(z_aug)
                    z_ori_out1 = tempo_classifier(z_ori[-len(obj_tgt_js):])

                    temp_labels = temp_labels[t_flag].long().view(-1).cuda(args.device)

                    temp_loss = criterion(z_ori_out1[t_flag].view(-1, args.sample_num), temp_labels)
                    loss_reg = mse_criterion(adaparams(z_aug[:-len(obj_tgt_js)] - z_ori[:-len(obj_tgt_js)]),torch.zeros_like((z_ori[:-len(obj_tgt_js)])))

                    loss_mapping = loss_reg + temp_loss*0.05

                    optimizer_mapping.zero_grad()
                    loss_mapping.backward()
                    optimizer_mapping.step()
                    writer.add_scalar('Train/temp_loss', temp_loss.item(), global_step=global_step)
                    writer.add_scalar('Train/reg_loss_ssl', loss_reg.item(), global_step=global_step)

                    if it == 0:
                        break

            if global_step % args.val_step == 0:

                smoothed_auc, smoothed_auc_avg, smoothed_auc_, smoothed_auc_avg_ , temp_timestamp = val(args, net, mapping, classifier, adaparams, tempo_classifier)
                writer.add_scalar('Test/smoothed_auc', smoothed_auc, global_step=global_step)
                writer.add_scalar('Test/smoothed_auc_avg', smoothed_auc_avg, global_step=global_step)

                if smoothed_auc > max_acc:
                    max_acc = smoothed_auc
                    timestamp_in_max = temp_timestamp
                    save = './checkpoint/{}_{}.pth'.format('best', running_date)
                    torch.save({'net_state_dict':net.state_dict(),'mapping_state_dict':mapping.state_dict(),'classifier_state_dict':classifier.state_dict(),'temp_classifier_state_dict':tempo_classifier.state_dict()}, save)
                if smoothed_auc_ > max_acc_:
                    max_acc_ = smoothed_auc_
                    timestamp_in_max = temp_timestamp
                    save = './checkpoint/{}_{}__.pth'.format('best', running_date)
                    torch.save({'net_state_dict':net.state_dict(),'mapping_state_dict':mapping.state_dict(),'classifier_state_dict':classifier.state_dict(),'temp_classifier_state_dict':tempo_classifier.state_dict()}, save)

                print('cur max: ' + str(max_acc) + ' in ' + timestamp_in_max)
                print('cur max: ' + str(max_acc_) + ' in ' + timestamp_in_max)
                net = net.train()
                mapping = mapping.train()
                classifier = classifier.train()
                adaparams = adaparams.train()
                tempo_classifier = tempo_classifier.train()


def val(args, net=None, mapping=None, classifier=None,  adaparams=None, tempo_classifier=None):
    if not args.log_date:
        running_date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    else:
        running_date = args.log_date
    print("The running_date : {}".format(running_date))

    # Load Data
    data_dir = f"F:/{args.dataset}/testing/frames"
    detect_pkl = f'detect/{args.dataset}_test_detect_result_yolov3.pkl'

    testing_dataset = VideoAnomalyDataset_C3D(data_dir,
                                              dataset=args.dataset,
                                              detect_dir=detect_pkl,
                                              fliter_ratio=args.filter_ratio,
                                              frame_num=args.sample_num)
    testing_data_loader = DataLoader(testing_dataset, batch_size=256, shuffle=False, num_workers=2, drop_last=False)

    net.eval()
    classifier.eval()
    mapping.eval()
    adaparams.eval()
    tempo_classifier.eval()

    video_output = {}
    mse = 0
    val_iter = 0
    for data in tqdm(testing_data_loader):
        videos = data["video"]
        frames = data["frame"].tolist()
        obj = data["obj"].cuda(args.device)
        val_iter += 1
        with torch.no_grad():
            z_ori, z_aug = net.fea1(obj)
            z_ori, z_aug = mapping.fea1(z_ori), mapping.fea1(z_aug)
            z_ori, z_aug = net.fea2(z_ori, z_aug)
            z_ori, z_aug = mapping.fea2(z_ori), mapping.fea2(z_aug)
            z_ori, z_aug = net.fea3(z_ori), net.fea3(z_aug)
            z_ori, z_aug = mapping.fea3(z_ori), mapping.fea3(z_aug)
            z_ori, z_aug = net.flat(z_ori), net.flat(z_aug)

            logits = classifier(z_ori)

            temp_logits = tempo_classifier(z_ori)
            temp_logits = temp_logits.view(-1, 7, 7)
        mse += F.mse_loss(z_ori,z_aug)
        scores = logits[:,0].cpu().numpy()
        # spat_probs = F.softmax(spat_logits, -1)
        # diag = torch.diagonal(spat_probs, offset=0, dim1=-2, dim2=-1)
        # s_scores = diag.min(-1)[0].cpu().numpy()

        temp_probs = F.softmax(temp_logits, -1)
        diag2 = torch.diagonal(temp_probs, offset=0, dim1=-2, dim2=-1)
        t_scores = diag2.min(-1)[0].cpu().numpy()

        # with torch.no_grad():
        #     temp_logits, spat_logits= net(obj,'eval')
        #     temp_logits = temp_logits.view(-1, args.sample_num, args.sample_num)
        #     spat_logits = spat_logits.view(-1, 9, 9)
        #
        # spat_probs = F.softmax(spat_logits, -1)
        # diag = torch.diagonal(spat_probs, offset=0, dim1=-2, dim2=-1)
        # scores = diag.min(-1)[0].cpu().numpy()
        #
        # temp_probs = F.softmax(temp_logits, -1)
        # diag2 = torch.diagonal(temp_probs, offset=0, dim1=-2, dim2=-1)
        # scores2 = diag2.min(-1)[0].cpu().numpy()

        # scores3 = mem_dist.cpu().numpy()

        for video_, frame_, cls_score, t_score_ in zip(videos, frames, scores, t_scores):
            if video_ not in video_output:
                video_output[video_] = {}
            if frame_ not in video_output[video_]:
                video_output[video_][frame_] = []
            video_output[video_][frame_].append([cls_score, t_score_])

    print('mse_loss : {}'.format(mse/val_iter))
    micro_auc, macro_auc, micro_auc_, macro_auc_ = save_and_evaluate(video_output, running_date, dataset=args.dataset)
    return micro_auc, macro_auc, micro_auc_, macro_auc_, running_date


def save_and_evaluate(video_output, running_date, dataset='shanghaitech'):
    pickle_path = './log/video_output_ori_{}.pkl'.format(running_date)
    with open(pickle_path, 'wb') as write:
        pickle.dump(video_output, write, pickle.HIGHEST_PROTOCOL)
    if dataset == 'shanghaitech':
        video_output_spatial, video_output_temporal, video_output_complete = remake_video_output(video_output,
                                                                                                 dataset=dataset)
    else:
        # video_output_spatial, video_output_temporal, video_output_complete = remake_video_output(video_output,
        #                                                                                          dataset=dataset)
        video_output_spatial, video_output_temporal, video_output_complete = remake_video_3d_output(video_output,
                                                                                                    dataset=dataset)

    try:
        smoothed_res, smoothed_auc_list = evaluate_auc(video_output_spatial, dataset=dataset)
        smoothed_res_micro = smoothed_res.auc
        smoothed_res_macro = np.mean(smoothed_auc_list)
    except Exception:
        smoothed_res, smoothed_auc_list = evaluate_auc(video_output_spatial, dataset=dataset)
        smoothed_res_micro = smoothed_res.auc
        smoothed_res_macro = np.mean(smoothed_auc_list)
        # print('spatial_logits_error!!!')
        # smoothed_res_micro = 0.0
        # smoothed_res_macro = 0.0

    try:
        evaluate_auc(video_output_spatial, dataset=dataset)
    except Exception:
        print('spatial_logits_error!!!')
    try:
        evaluate_auc(video_output_temporal, dataset=dataset)
    except Exception:
        evaluate_auc(video_output_temporal, dataset=dataset)
        # print('temporal_logits_error!!!')
    try:
        smoothed_res_, smoothed_auc_list_ = evaluate_auc(video_output_complete, dataset=dataset)
        smoothed_res_micro_ = smoothed_res_.auc
        smoothed_res_macro_ = np.mean(smoothed_auc_list_)
    except Exception:
        smoothed_res_micro_ = 0.0
        smoothed_res_macro_ = 0.0
    # smoothed_res_micro_ = 0.0
    # smoothed_res_macro_ = 0.0

    return smoothed_res_micro, smoothed_res_macro, smoothed_res_micro_, smoothed_res_macro_


if __name__ == '__main__':
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    args = get_configs()
    train(args)
