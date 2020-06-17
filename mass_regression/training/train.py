from argparse import ArgumentParser

from training.wlnu import WlnuParser


def main():
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(dest='dataset')
    wlnu_parser = WlnuParser(parser, subparsers)
    args = parser.parse_args()
    if args.dataset == wlnu_parser.name:
        wlnu_parser.parse(args)


if __name__ == '__main__':
    main()
