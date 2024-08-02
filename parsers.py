import argparse

parser = argparse.ArgumentParser()


parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--seed', type=int, default=2024, help='Random seed')
parser.add_argument('--log_file', default='log/', help='log dir path')
parser.add_argument('-preprocess', type=bool, default=False, help="preprocess dataset")    #when you first run code, you should set it to true.
parser.add_argument('--data_name', type=str, default="weibo", choices=['android','weibo', 'memetracker', 'twitter', 'douban','christianity'], help="dataset")
parser.add_argument('--max_len', type=int, default=200, help='max seq length')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--patience', type=int, default=15, help='the number of epoch to wait before early stop')
parser.add_argument('--eval_interval', type=int, default=1, help='the number of epoch to eval')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR')
parser.add_argument('--decay_step', type=int, default=100, help='Decay step for StepLR')
parser.add_argument('--att_head', type=int, default=8, help='attention head')
parser.add_argument('--n_warmup_steps', type=int, default=1000, help='the dim of hidden emb')
parser.add_argument('--hidden_size', type=int, default=64, help='the dim of hidden emb')
parser.add_argument('--compress_emb', type=int, default=48, help='the dim of hidden emb')
parser.add_argument('--num_blocks', type=int, default=4, help='layer num of transformer layer')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout ratio')
parser.add_argument('--steps', type=int, default=32, help='the max diffusion step')
parser.add_argument('--beta_start', type=float, default=1e-4, help='the min beta')
parser.add_argument('--beta_end', type=float, default=0.02, help='the max beta')
parser.add_argument('--schedule_sampler_name', type=str, default='lossaware', help='the sample method of timesteps')
parser.add_argument('--beta_sche', type=str, default='linear', help='the method of generate betas')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
parser.add_argument('--emb_dropout', type=float, default=0.3, help='emb dropout ratio')
parser.add_argument('--rescale_timesteps', type=bool, default=False, help='rescale timesteps or not')




