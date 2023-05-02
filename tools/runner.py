import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from scipy import stats
from tools import builder, helper
from utils import misc
import time
import csv
#from models.cnn_model import GCNnet_artisticswimming
from models.group_aware_attention import Encoder_Blocks
#from utils.multi_gpu import *
from models.cnn_simplified import GCNnet_artisticswimming_simplified
from models.linear_for_bp import Linear_For_Backbone
from mindspore.dataset import GeneratorDataset
#from thop import profile


def test_net(args):
    print('Tester start ... ')
    train_dataset_generator, test_dataset_generator = builder.dataset_builder(args)
    test_dataset = GeneratorDataset(test_dataset_generator, ["data", "target"], shuffle=False, num_parallel_workers=args.workers)
    test_dataset = test_dataset.batch(batch_size=args.bs_test)
    test_dataloader = test_dataset.create_tuple_iterator()
    '''test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs_test,
                                                  shuffle=False, num_workers=int(args.workers),
                                                  pin_memory=True)'''
    base_model, regressor = builder.model_builder(args)
    # load checkpoints
    builder.load_model(base_model, regressor, args)

    # if using RT, build a group
    group = builder.build_group(train_dataset_generator, args)

    '''# CUDA
    global use_gpu
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        # base_model = base_model
        regressor = regressor
        torch.backends.cudnn.benchmark = True'''

    #  DP
    # base_model = nn.DataParallel(base_model)
    '''regressor = nn.DataParallel(regressor)'''

    test(base_model, regressor, test_dataset, group, args)


def run_net(args):
    '''if is_main_process():
        print('Trainer start ... ')'''
    print('Trainer start ... ')
    # build dataset
    train_dataset_generator, test_dataset_generator = builder.dataset_builder(args)
    if args.use_multi_gpu:
        raise NotImplementedError()
        train_dataloader = build_dataloader(train_dataset_generator,
                                            batch_size=args.bs_train,
                                            shuffle=True,
                                            num_workers=args.workers,
                                            persistent_workers=True,
                                            seed=set_seed(args.seed))
    else:
        train_dataset = GeneratorDataset(train_dataset_generator, ['data_feature', 'data_final_score', 'data_difficulty', 'data_boxes', 'data_cnn_features', 
                                'target_feature', 'target_final_score', 'target_difficulty', 'target_boxes', 'target_cnn_features'], shuffle=False, num_parallel_workers=args.workers)
        train_dataset = train_dataset.batch(batch_size=args.bs_train)
        train_dataloader = train_dataset.create_tuple_iterator()
        '''train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs_train,
                                                       shuffle=False, num_workers=int(args.workers),
                                                       pin_memory=True)'''
    test_dataset = GeneratorDataset(test_dataset_generator, ['data_feature', 'data_final_score', 'data_difficulty', 'data_boxes', 'data_cnn_features', 
                                'target_feature', 'target_final_score', 'target_difficulty', 'target_boxes', 'target_cnn_features'], shuffle=False, num_parallel_workers=args.workers)
    test_dataset = test_dataset.batch(batch_size=args.bs_test)
    test_dataloader = test_dataset.create_tuple_iterator()
    '''test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs_test,
                                                  shuffle=False, num_workers=int(args.workers),
                                                  pin_memory=True)'''

    # Set data position
    #device = get_device()
    device = None

    # build model
    base_model, regressor = builder.model_builder(args)

    '''input1 = torch.randn(2, 2049)
    flops, params = profile(regressor, inputs=(input1, ))
    print(f'[regressor]flops: ', flops, 'params: ', params)'''

    if args.warmup:
        '''len_train_dataloader = 0
        for _ in train_dataloader:
            len_train_dataloader += 1
        num_steps = len_train_dataloader * args.max_epoch'''
        num_steps = train_dataset.get_dataset_size() * args.max_epoch
        cosine_decay_lr = nn.CosineDecayLR(min_lr = 1e-8, max_lr = args.lr, decay_steps=num_steps)
        #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    # Set models and optimizer(depend on whether to use goat)
    if args.use_goat:
        if args.use_cnn_features:
            gcn = GCNnet_artisticswimming_simplified(args)

            '''input1 = torch.randn(1, 540, 8, 1024)
            input2 = torch.randn(1, 540, 8, 4)
            flops, params = profile(gcn, inputs=(input1, input2))
            print(f'[GCNnet_artisticswimming_simplified]flops: ', flops, 'params: ', params)'''
        else:
            raise NotImplementedError()
            gcn = GCNnet_artisticswimming(args)
            gcn.loadmodel(args.stage1_model_path)
        attn_encoder = Encoder_Blocks(args.qk_dim, 1024, args.linear_dim, args.num_heads, args.num_layers, args.attn_drop)
        linear_bp = Linear_For_Backbone(args)
        optimizer = nn.Adam([
            {'params': gcn.trainable_params(), 'lr': args.lr * args.lr_factor},
            {'params': regressor.trainable_params()},
            {'params': linear_bp.trainable_params()},
            {'params': attn_encoder.trainable_params()}
        ], learning_rate=cosine_decay_lr if args.warmup else args.lr, weight_decay=args.weight_decay)
        #scheduler = None
        if args.use_multi_gpu:
            raise NotImplementedError()
            wrap_model(gcn, distributed=args.distributed)
            wrap_model(attn_encoder, distributed=args.distributed)
            wrap_model(linear_bp, distributed=args.distributed)
            wrap_model(regressor, distributed=args.distributed)
        else:
            pass
            '''gcn = gcn.to(device=device)
            attn_encoder = attn_encoder.to(device=device)
            linear_bp = linear_bp.to(device=device)
            regressor = regressor.to(device=device)'''
    else:
        gcn = None
        attn_encoder = None
        linear_bp = Linear_For_Backbone(args)
        optimizer = nn.Adam([{'params': regressor.trainable_params()}, {'params': linear_bp.trainable_params()}], learning_rate=cosine_decay_lr if args.warmup else args.lr, weight_decay=args.weight_decay)
        #scheduler = None
        if args.use_multi_gpu:
            raise NotImplementedError()
            wrap_model(regressor, distributed=args.distributed)
            wrap_model(linear_bp, distributed=args.distributed)
        else:
            pass
            '''regressor = regressor.to(device=device)
            linear_bp = linear_bp.to(device=device)'''


    # if using RT, build a group
    group = builder.build_group(train_dataset_generator, args)
    # CUDA
    '''global use_gpu
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        torch.backends.cudnn.benchmark = True'''

    # parameter setting
    start_epoch = 0
    global epoch_best, rho_best, L2_min, RL2_min
    epoch_best = 0
    rho_best = 0
    L2_min = 1000
    RL2_min = 1000

    # resume ckpts
    if args.resume:
        start_epoch, epoch_best, rho_best, L2_min, RL2_min = \
            builder.resume_train(base_model, regressor, optimizer, args)
        print('resume ckpts @ %d epoch( rho = %.4f, L2 = %.4f , RL2 = %.4f)' % (
            start_epoch - 1, rho_best, L2_min, RL2_min))

    #  DP
    # regressor = nn.DataParallel(regressor)
    # if args.use_goat:
    #     gcn = nn.DataParallel(gcn)
    #     attn_encoder = nn.DataParallel(attn_encoder)

    # loss
    mse = nn.MSELoss()
    nll = nn.NLLLoss()

    # trainval

    # training
    for epoch in range(start_epoch, args.max_epoch):
        if args.use_multi_gpu:
            raise NotImplementedError()
            train_dataloader.sampler.set_epoch(epoch)
        true_scores = []
        pred_scores = []
        num_iter = 0
        # base_model.train()  # set model to training mode
        regressor.set_train()
        linear_bp.set_train()
        if args.use_goat:
            gcn.set_train()
            attn_encoder.set_train()
        # if args.fix_bn:
        #     base_model.apply(misc.fix_bn)  # fix bn

        for idx, data_get in enumerate(train_dataset.create_dict_iterator()):
            # break
            num_iter += 1
            opti_flag = False

            data = {}
            target = {}
            '''data['feature'] = data_get['data_feature']
            data['final_score'] = data_get['data_final_score']
            data['difficulty'] = data_get['data_difficulty']'''
            data['boxes'] = data_get['data_boxes'].float()
            data['cnn_features'] = data_get['data_cnn_features'].float()
            '''target['feature'] = data_get['target_feature']
            target['final_score'] = data_get['target_final_score']
            target['difficulty'] = data_get['target_difficulty']'''
            target['boxes'] = data_get['target_boxes'].float()
            target['cnn_features'] = data_get['target_cnn_features'].float()

            true_scores.extend(data_get['data_final_score'].asnumpy())
            # data preparing
            # featue_1 is the test video ; video_2 is exemplar
            if args.benchmark == 'MTL':
                feature_1 = data_get['data_feature'].float()  # N, C, T, H, W
                if args.usingDD:
                    label_1 = data['completeness'].float().reshape(-1, 1)
                    label_2 = target['completeness'].float().reshape(-1, 1)
                else:
                    label_1 = data_get['data_final_score'].float().reshape(-1, 1)
                    label_2 = data_get['target_final_score'].float().reshape(-1, 1)
                if not args.dive_number_choosing and args.usingDD:
                    assert (data_get['data_difficulty'] == data_get['target_difficulty']).all()
                diff = data_get['data_difficulty'].float().reshape(-1, 1)
                feature_2 = data_get['target_feature'].float()  # N, C, T, H, W

            else:
                raise NotImplementedError()

            # forward
            if num_iter == args.step_per_update:
                num_iter = 0
                opti_flag = True

            '''optimizer = nn.Adam(linear_bp.trainable_params(), 1e-2)
            def forward_fn_test(feature_1, feature_2):
                y = linear_bp(feature_1)  # B,540,1024
                loss = mse(y, y)
                print(y.shape)
                return loss, y
            
            forward_fn_test(feature_1, feature_2)
            grad_fn_test = ms.value_and_grad(forward_fn_test, None, optimizer.parameters, has_aux=True)
            (loss, _), grads = grad_fn_test(feature_1, feature_2)'''

            helper.network_forward_train(base_model, regressor, pred_scores, feature_1, label_1, feature_2, label_2,
                                         diff, group, mse, nll, optimizer, opti_flag, epoch, idx + 1,
                                         train_dataset.get_dataset_size(), args, data, target, gcn, attn_encoder, device, linear_bp)

            '''if args.warmup:
                lr_scheduler.step()'''

        # analysis on results
        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)
        rho, p = stats.spearmanr(pred_scores, true_scores)
        L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
        RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / \
              true_scores.shape[0]
        '''if is_main_process():
            print('[Training] EPOCH: %d, correlation: %.4f, L2: %.4f, RL2: %.4f, lr1: %.4f' % (
                epoch, rho, L2, RL2, optimizer.param_groups[0]['lr']))'''
        print('[Training] EPOCH: %d, correlation: %.4f, L2: %.4f, RL2: %.4f, lr1: %.4f' % (
                epoch, rho, L2, RL2, optimizer.get_lr()))

        '''if is_main_process():
            validate(base_model, regressor, test_dataloader, epoch, optimizer, group, args, gcn, attn_encoder, device, linear_bp)
            # helper.save_checkpoint(base_model, regressor, optimizer, epoch, epoch_best, rho_best, L2_min, RL2_min,
            #                        'last',
            #                        args)
            print('[TEST] EPOCH: %d, best correlation: %.6f, best L2: %.6f, best RL2: %.6f' % (
                epoch, rho_best, L2_min, RL2_min))'''
        validate(base_model, regressor, test_dataset, epoch, optimizer, group, args, gcn, attn_encoder, device, linear_bp)
        # helper.save_checkpoint(base_model, regressor, optimizer, epoch, epoch_best, rho_best, L2_min, RL2_min,
        #                        'last',
        #                        args)
        print('[TEST] EPOCH: %d, best correlation: %.6f, best L2: %.6f, best RL2: %.6f' % (
            epoch, rho_best, L2_min, RL2_min))
        # scheduler lr
        '''if scheduler is not None:
            scheduler.step()'''


# TODO: 修改以下所有;修改['difficulty'].float
def validate(base_model, regressor, test_dataset, epoch, optimizer, group, args, gcn, attn_encoder, device, linear_bp):
    print("Start validating epoch {}".format(epoch))
    global use_gpu
    global epoch_best, rho_best, L2_min, RL2_min
    true_scores = []
    pred_scores = []
    # base_model.eval()  # set model to eval mode
    regressor.set_train(False)
    linear_bp.set_train(False)
    if args.use_goat:
        gcn.set_train(False)
        attn_encoder.set_train(False)
    batch_num = test_dataset.get_dataset_size()

    datatime_start = time.time()
    for batch_idx, data_get in enumerate(test_dataset.create_dict_iterator(), 0):
        datatime = time.time() - datatime_start
        start = time.time()
        data = {}
        data['feature'] = data_get['data_feature']
        data['final_score'] = data_get['data_final_score']
        data['difficulty'] = data_get['data_difficulty']
        data['boxes'] = data_get['data_boxes']
        data['cnn_features'] = data_get['data_cnn_features']
        target_len = data_get['target_final_score'].shape[1]
        target = [{'feature': data_get['target_feature'][:, i, :, :], 'final_score': data_get['target_final_score'][:, i], 'difficulty': data_get['target_difficulty'][:, i],
                'boxes': data_get['target_boxes'][:, i, :, :], 'cnn_features': data_get['target_cnn_features'][:, i, :, :]} for i in range(target_len)]
        true_scores.extend(data['final_score'].asnumpy())
        # data prepare
        if args.benchmark == 'MTL':
            feature_1 = data['feature'].float()  # N, C, T, H, W
            if args.usingDD:
                label_2_list = [item['completeness'].float().reshape(-1, 1) for item in target]
            else:
                label_2_list = [item['final_score'].float().reshape(-1, 1) for item in target]
            diff = data['difficulty'].float().reshape(-1, 1)
            feature_2_list = [item['feature'].float() for item in target]
            # check
            if not args.dive_number_choosing and args.usingDD:
                for item in target:
                    assert (diff == item['difficulty'].reshape(-1, 1)).all()
        else:
            raise NotImplementedError()
        helper.network_forward_test(base_model, regressor, pred_scores, feature_1, feature_2_list, label_2_list,
                                    diff, group, args, data, target, gcn, attn_encoder, device, linear_bp)
        batch_time = time.time() - start
        if batch_idx % args.print_freq == 0:
            print('[TEST][%d/%d][%d/%d] \t Batch_time %.6f \t Data_time %.6f '
                    % (epoch, args.max_epoch, batch_idx, batch_num, batch_time, datatime))
        datatime_start = time.time()

        # analysis on results
        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)
        rho, p = stats.spearmanr(pred_scores, true_scores)
        L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
        RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / \
              true_scores.shape[0]
        if L2_min > L2:
            L2_min = L2
        if RL2_min > RL2:
            RL2_min = RL2
        if rho > rho_best:
            rho_best = rho
            epoch_best = epoch
            print('-----New best found!-----')
            helper.save_outputs(pred_scores, true_scores, args)
            helper.save_checkpoint(base_model, regressor, optimizer, epoch, epoch_best, rho_best, L2_min, RL2_min,
                                    'best', args)
        if epoch == args.max_epoch - 1:
            log_best(rho_best, RL2_min, epoch_best, args)

        print('[TEST] EPOCH: %d, correlation: %.6f, L2: %.6f, RL2: %.6f' % (epoch, rho, L2, RL2))


def test(base_model, regressor, test_dataset, group, args, gcn, attn_encoder, device):
    global use_gpu
    true_scores = []
    pred_scores = []
    # base_model.set_train(False) # set model to eval mode
    regressor.set_train(False)
    if args.use_goat:
        gcn.set_train(False)
        attn_encoder.set_train(False)
    batch_num = test_dataset.get_dataset_size()

    datatime_start = time.time()
    for batch_idx, (data, target) in enumerate(test_dataset, 0):
        datatime = time.time() - datatime_start
        start = time.time()
        true_scores.extend(data['final_score'].asnumpy())
        # data prepare
        if args.benchmark == 'MTL':
            featue_1 = data['feature'].float()  # N, C, T, H, W
            if args.usingDD:
                label_2_list = [item['completeness'].float().reshape(-1, 1) for item in target]
            else:
                label_2_list = [item['final_score'].float().reshape(-1, 1) for item in target]
            diff = data['difficulty'].float().reshape(-1, 1)
            feature_2_list = [item['feature'].float() for item in target]
            # check
            if not args.dive_number_choosing and args.usingDD:
                for item in target:
                    assert (diff == item['difficulty'].float().reshape(-1, 1)).all()
        elif args.benchmark == 'Seven':
            featue_1 = data['feature'].float()  # N, C, T, H, W
            feature_2_list = [item['feature'].float() for item in target]
            label_2_list = [item['final_score'].float().reshape(-1, 1) for item in target]
            diff = None
        else:
            raise NotImplementedError()
        helper.network_forward_test(base_model, regressor, pred_scores, featue_1, feature_2_list, label_2_list,
                                    diff, group, args, data, target, gcn, attn_encoder, device)
        batch_time = time.time() - start
        if batch_idx % args.print_freq == 0:
            print('[TEST][%d/%d] \t Batch_time %.2f \t Data_time %.2f '
                    % (batch_idx, batch_num, batch_time, datatime))
        datatime_start = time.time()

        # analysis on results
        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)
        rho, p = stats.spearmanr(pred_scores, true_scores)
        L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
        RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / \
              true_scores.shape[0]
        print('[TEST] correlation: %.6f, L2: %.6f, RL2: %.6f' % (rho, L2, RL2))


def log_best(rho_best, RL2_min, epoch_best, args):
    # log for best
    with open(args.result_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if args.use_goat:
            if args.use_formation:
                mode = 'Formation'
            elif args.use_bp:
                mode = 'BP'
            elif args.use_self:
                mode = 'SELF'
            else:
                mode = 'GOAT'
        else:
            mode = 'Ori'

        if args.use_i3d_bb:
            backbone = 'I3D'
        elif args.use_swin_bb:
            backbone = 'SWIN'
        else:
            backbone = 'BP_BB'

        log_list = [format(rho_best, '.4f'), epoch_best, args.use_goat, args.lr, args.max_epoch, args.warmup,
                    args.seed, args.train_backbone, args.num_selected_frames, args.num_heads,
                    args.num_layers, args.random_select_frames, args.bs_train, args.bs_test, args.linear_dim, RL2_min, mode, backbone]
        writer.writerow(log_list)