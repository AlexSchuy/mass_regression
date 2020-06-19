from argparse import ArgumentParser

from training.datasets.wlnu import WlnuParser
from training.datasets.h import HiggsParser

def main():
    parser = ArgumentParser(conflict_handler='resolve')

    subparsers = parser.add_subparsers(dest='dataset')
    wlnu_parser = WlnuParser(parser, subparsers)
    higgs_parser = HiggsParser(parser, subparsers)
    args = parser.parse_args()
    if args.dataset == wlnu_parser.name:
        wlnu_parser.parse(args)
    elif args.dataset == higgs_parser.name:
        higgs_parser.parse(args)

if __name__ == '__main__':
    main()
