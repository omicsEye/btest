import argparse
import datetime
import os
import warnings

from .viz import b_scatter


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datax', '-x', help="path to dataX", type=str, required=True)
    parser.add_argument('--datay', '-y', help="path to dataY", type=str, required=True)
    parser.add_argument('--b_test', '-b', help="path to b_test results", type=str, required=True)
    parser.add_argument('--ind', '-i', help="list of indexes starting from zero to plot in format 1,2,3,... or 1-10",
                        type=str, required=True)
    parser.add_argument('--out', '-o', help="path to output directory",
                        type=str, required=True)
    parser.add_argument('--min', '-m', help="minimum var to include", type=int, default=0)
    return parser.parse_args()


def main():
    # Parse arguments from command line
    warnings.filterwarnings("ignore")
    warnings.simplefilter('ignore')

    args = parse_arguments()
    dt_label = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = str(args.out + '_' + dt_label)
    os.makedirs(output_dir)
    print(args.ind)
    if '-' in args.ind:
        start, end = args.ind.split('-')
        n_ind = range(int(start),int(end))
    elif ',' in args.ind:
        n_ind = args.ind.split(',')
    n_ind = list(map(int, n_ind))
    b_scatter(args.datax, args.datay, b_test=args.b_test,
              n_ind=n_ind,
              min_var=args.min,
              output_dir=output_dir)
    return print('Done!')


# main()
if __name__ == "__main__":
    main()
