import argparse

parser = argparse.ArgumentParser(description="BadNet with backdoor attack on MNIST")
parser.add_argument('--dataset',default='mnist')
parser.add_argument('--datapath',default='./datasets/')
parser.add_argument('--download',action='store_true')
parser.add_argument('--portion_rate',type=float,default=0.1)
parser.add_argument('--batch_size',type=int,default=64)
parser.add_argument('--trigger_label',type=int,default=7)

args = parser.parse_args()