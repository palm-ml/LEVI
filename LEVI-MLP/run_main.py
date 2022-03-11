import os
import argparse
import os.path as p 
# from main import main

PATH = p.dirname(__file__)

dataset_list = {'recovery': ['Artificial','SJAFFE','Yeast_spoem','Yeast_spo5','Yeast_dtt','Yeast_cold','Yeast_heat','Yeast_spo','Yeast_diau','Yeast_elu','Yeast_cdc','Yeast_alpha','SBU_3DFE','Movie'],
                'classification': ['CAL500','corel5k','emotions','enron','image','llog','medical','msra','scene','slashdot','yeast','bibtex','tmc2007','rcv1subset1','rcv1subset2']}
dataset_type = ['norm','binary','norm','binary','norm','binary','binary','norm','norm','binary','norm','binary','binary','norm','norm']

parser = argparse.ArgumentParser(description='VAE_LE  process')

parser.add_argument('--type','-t',type=str, default='classification',
                    help = 'recovery or classification')
parser.add_argument('--dataset_id','-id',type=int, default=0)

# training args
parser.add_argument('--epochs', '-e', type=int, default=200,
                    help = 'number of epochs to train (default: 500)')
parser.add_argument('--learning_rate','-lr', type=float, default=0.001,
                    help = 'learning rate (default: 0.001)')
parser.add_argument('--keep_prob','-k', type=float, default=0.9,
                    help = 'keep ratio of the dropout settings (default: 0.9)')

# model args
parser.add_argument('--n_hidden','-hidden', type=int, default=500,
                    help = 'number of the hidden nodes (default: 150)')
parser.add_argument('--dim_z','-dim_z', type=int, default=200,
                    help='dimension of the variable Z (default: 100)')
parser.add_argument('--alpha','-a', type=float, default=0.01,
                    help = 'balance parameter of the loss function (default=1.0)')
parser.add_argument('--beta','-beta', type=float, default=0.01, 
                    help = 'balance parameter of the loss function (default=1.0)')
parser.add_argument('--gamma','-gamma', type=float, default=0, 
                    help = 'balance parameter of the loss function (default=1.0)')
parser.add_argument('--ld','-ld', type=float, default=1,
                    help = 'balance parameter of the loss function (default=1.0)')

# other args
parser.add_argument('--gpu', '-gpu', type = int, default = 0, 
                    help = 'device of gpu id (default: 0)')
parser.add_argument('--seed', '-seed', type = int, default = 0,
                    help = 'random seed (default: 0)')

parser.add_argument('--src_path',type=str,default=p.join(PATH,'datasets'))
parser.add_argument('--adj_path',type=str,default=p.join(PATH,'adjmat'))
parser.add_argument('--dst_path', type=str, default=p.join(PATH, 'results'))

args = vars(parser.parse_args())
def check_adj(ds):
    if not p.exists("adjmat/{}".format(ds)):
        os.mkdir("adjmat/{}".format(args['dataset']))
        return False
    else:
        if len(os.listdir("adjmat/{}".format(ds))) != 10:
            return False
    return True
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    assert args['type'] in ['recovery', 'classification']
    args['dataset'] = dataset_list[args['type']][args['dataset_id']]
    print('---------------------------------------------------------------------')
    # check adj
    if not check_adj(args['dataset']):
        import matlab
        import matlab.engine
        print('Not Found Adj Matrix. Creating adj matrix now.')
        eng = matlab.engine.start_matlab()
        adj = eng.gen_adj(args['dataset'])
        print("The adj matrix has been created. Please restart this program")
        exit()
    from main import main
    if args['type'] == 'recovery':
        print(args)
        main(args)
    else:
        args['data_type'] = dataset_type[args['dataset_id']]
        for i in range(1,11):
            args['split'] = i
            print(args)
            main(args)
        print("Begin Classification.")
        os.chdir('Classification/')
        import matlab
        import matlab.engine
        eng = matlab.engine.start_matlab()
        adj = eng.main(nargout=0)

