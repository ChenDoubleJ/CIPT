import argparse


def parse_signal_args():
    parser = argparse.ArgumentParser(description='Improving NLOS/LOS Classification Accuracy in Urban Canyon Based on '
                                                 'Channel-Independent Patch Transformer with Temporal Information')

    # random seed
    parser.add_argument('--random_seed', type=int, default=4, help='random seed')

    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    #
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    #
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')

    parser.add_argument('--enc_in', type=int, default=3,
                        help='encoder input size')  # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=3, help='decoder input size')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')  # 128
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')  # 8
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=3, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')  # 512
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')  # 0.

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--epochs', type=int, default=300, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    args = parser.parse_args()

    return args
