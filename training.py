# general utilities
import copy
import datetime
from time import time, clock
from os import system
from os.path import isfile
import sys
from importlib import import_module
import argparse

# numerical libraries
import numpy as np
from nbodykit.lab import ArrayMesh, FFTPower
from nbodykit.algorithms.fftpower import ProjectedFFTPower

# HDF5
import h5py

# multiprocessing (for parallel HDF5 reads)
from mpi4py import MPI

# pytorch
import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, get_worker_info
from torchsummary import summary

# plotting
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from matplotlib.patches import Rectangle

"""
TODO
!!! LOSS SEEMS PERIODIC -- WHAT'S GOING ON???
"""

MAX_SEED = 2**32

class CustomLoss(nn.Module) :#{{{
    def __init__(self) :#{{{
        super(CustomLoss, self).__init__()
    #}}}
    def forward(self, pred, targ) :#{{{
        torch.pow(pred, 4.0, out = pred)
        torch.pow(targ, 4.0, out = targ)
        torch.add(pred, targ, alpha = -1.0, out = targ)
        return torch.abs(torch.sum(targ))
    #}}}
#}}}

class LnLoss(nn.Module) :#{{{
    def __init__(self, **kwargs) :#{{{
        super(LnLoss, self).__init__()
        self.__pwr = kwargs['pwr']
    #}}}
    def forward(self, pred, targ) :#{{{
        return torch.sum(torch.pow(
            torch.abs(torch.add(pred, targ, alpha = -1.0)),
            torch.tensor(self.__pwr, requires_grad = False).to(pred.device, dtype = torch.float32)
            ))
    #}}}
#}}}

_namesofplaces = {#{{{
    'Conv': nn.Conv3d,
    'ConvTranspose': nn.ConvTranspose3d,
    'BatchNorm': nn.BatchNorm3d,
    'ReLU': nn.ReLU,
    'LeakyReLU': nn.LeakyReLU,
    'MSELoss': nn.MSELoss,
    'L1Loss': nn.L1Loss,
    'LnLoss': LnLoss,
    'Adam': torch.optim.Adam,
    'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
    'StepLR': torch.optim.lr_scheduler.StepLR,
    'None': None,
    }
#}}}
def _merge(source, destination):#{{{
    # overwrites field in destination if field exists in source, otherwise just merges
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            _merge(value, node)
        else:
            destination[key] = value
    return destination
#}}}

class _ArgParser(object) :#{{{
    def __init__(self) :#{{{
        _modes    = ['train', 'valid', ]
#        _scalings = ['default', 'default_smoothed', 'default_centred', 'default_centred_nobatt', 
#                     'default_battcorrected', 'default_centred_battcorrected', 'default_centred_nobatt_battcorrected', ]
        _scalings = ['default_centred_battcorrected'] # the others are not implemented yet

        self.__parser = argparse.ArgumentParser(
            description='Code to train on DM only sims and hydro sims to \
                         learn mapping between DM and electron pressure.'
            )

        # define command line options
        self.__parser.add_argument(
            '-m', '--mode',
            nargs = '?',
            required = True,
            help = 'Currently from %s.'%_modes,
            )
        self.__parser.add_argument(
            '-n', '--network',
            nargs = '?',
            required = True,
            help = 'Name of the network architecture to be used.',
            )
        self.__parser.add_argument(
            '-c', '--config',
            nargs = '+',
            required = True,
            help = 'Any number of config files. Later configs overwrite earlier ones. \
                    For example, if passed --config c1 c2, in case of conflict \
                    the entry in c2 will be taken.',
            )
        self.__parser.add_argument(
            '-s', '--scaling',
            nargs = '?',
            required = True,
            help = 'Scaling of the target, currently from %s.'%_scalings
            )
        self.__parser.add_argument(
            '-o', '--output',
            nargs = '?',
            required = True,
            help = 'Identifier of output files (loss and trained_network).'
            )
        self.__parser.add_argument(
            '-t', '--time',
            nargs = '?',
            type = float,
            default = 24.0,
            help = 'Maximum runtime in hrs.'
            )
        self.__parser.add_argument(
            '-psfid', '--powerspectrumfid',
            nargs = '?',
            default = 'psfid',
            help = 'File to store fiducial powerspectrum.'
            )
        self.__parser.add_argument(
            '-pspred', '--powerspectrumpred',
            nargs = '?',
            default = 'pspred',
            help = 'File to store predicted powerspectrum.'
            )
        self.__parser.add_argument(
            '-projpsfid', '--projpowerspectrumfid',
            nargs = '?',
            default = 'projpsfid',
            help = 'File to store fiducial projected powerspectrum.'
            )
        self.__parser.add_argument(
            '-projpspred', '--projpowerspectrumpred',
            nargs = '?',
            default = 'projpspred',
            help = 'File to store predicted projected powerspectrum.'
            )
        self.__parser.add_argument(
            '-cross', '--crosspower',
            nargs = '?',
            default = 'cross',
            help = 'File to store correlation coefficient r(k)'
            )
        self.__parser.add_argument(
            '-opfid', '--onepointfid',
            nargs = '?',
            default = 'opfid',
            help = 'File to store fiducial powerspectrum.'
            )
        self.__parser.add_argument(
            '-oppred', '--onepointpred',
            nargs = '?',
            default = 'oppred',
            help = 'File to store predicted powerspectrum.'
            )
        self.__parser.add_argument(
            '-fpred', '--fieldpred',
            nargs = '?',
            default = 'fieldpred',
            help = 'File to store predicted gas field.'
            )
        self.__parser.add_argument(
            '-v', '--verbose',
            action = 'store_true',
            help = 'If you want some diagnostic outputs.',
            )
        self.__parser.add_argument(
            '-p', '--parallel',
            action = 'store_true',
            help = 'If several GPUs are available.',
            )
        self.__parser.add_argument(
            '--nocopy',
            action = 'store_true',
            help = 'Do not copy the data to the local GPU node disk.',
            )
        self.__parser.add_argument(
            '-d', '--debug',
            action = 'store_true',
            help = 'For short debugging runs.',
            )
        self.__parser.add_argument(
            '-sb', '--savebest',
            action = 'store_true',
            help = 'Do you want to save the network with the best validation loss?',
            )
        self.__parser.add_argument(
            '--ignoreexisting',
            action = 'store_true',
            help = 'If this flag is set, training will start at random initialization,\
                    regardless of whether the network has already been trained and saved\
                    to disk.'
            )
        self.__parser.add_argument(
            '-npath', '--network_path',
            nargs = '?',
            default='/home/lthiele/DM_to_electrons/Networks',
            help = 'Path where the network architectures are stored.',
            )
        self.__parser.add_argument(
            '-cpath', '--config_path',
            nargs = '?',
            default='/home/lthiele/DM_to_electrons/Configs',
            help = 'Path where the configuration files are stored.',
            )
        
        # parse now
        self.__args = self.__parser.parse_args()

        # set paths
        sys.path.append(self.__args.network_path)
        sys.path.append(self.__args.config_path)

        # consistency checks
        assert self.__args.mode    in _modes,    'Only %s modes implemented so far, passed %s.'%(_modes, self.__args.mode)
        assert self.__args.scaling in _scalings, 'Only %s scalings implemented so far, passed %s.'%(_scalings, self.__args.scaling)

        # read config files
        self.final_config = self.__read_configs()
    #}}}
    def __read_configs(self) :#{{{
        __configs = []
        for c in self.__args.config :
            __configs.append(import_module(c).this_config)
        assert len(__configs) > 0
        if self.__args.verbose :
            for ii, c in enumerate(__configs) :
                print '%s :'%self.__args.config[ii]
                print c
        for ii in xrange(1, len(__configs)) :
            __configs[0] = _merge(__configs[ii], __configs[0])
        if self.__args.verbose :
            print 'Using the following configuration :'
            print __configs[0]
        return __configs[0]
    #}}}
    def __getattr__(self, name) :#{{{
        return self.__args.__getattribute__(name)
    #}}}
    def __getitem__(self, name) :#{{{
        return self.final_config[name]
    #}}}
#}}}

class GlobalData(object) :#{{{
    def __init__(self) :#{{{
        # which box to work on
        self.box_sidelength = ARGS['box_sidelength']
        self.gas_sidelength = (ARGS['gas_sidelength']*self.box_sidelength)/2048
        self.DM_sidelength  = (ARGS['DM_sidelength']*self.box_sidelength)/2048
        assert not self.gas_sidelength%2, 'Only even gas_sidelength is supported.'
        assert not self.DM_sidelength%2,  'Only even DM_sidelength is supported.'

        # training hyperparameters
        self.__loss_function    = _namesofplaces[ARGS['loss_function']]
        self.__loss_function_kw = ARGS['loss_function_%s_kw'%ARGS['loss_function']]

        self.__optimizer        = _namesofplaces[ARGS['optimizer']]
        self.__optimizer_kw     = ARGS['optimizer_%s_kw'%ARGS['optimizer']]

        self.Nsamples           = ARGS['Nsamples']

        self.__data_loader_kw   = ARGS['data_loader_kw']
        self.num_workers = self.__data_loader_kw['num_workers'] if 'num_workers' in self.__data_loader_kw else 1

        self.__lr_scheduler     = _namesofplaces[ARGS['lr_scheduler']]
        if self.__lr_scheduler is not None :
            self.__lr_scheduler_kw = ARGS['lr_scheduler_%s_kw'%ARGS['lr_scheduler']]

        # random noise in target
        self.gas_noise = ARGS['gas_noise']
        self.DM_noise = ARGS['DM_noise']

        # sample selector
        self.sample_selector_kw = ARGS['sample_selector_kw']

        # individual boxes
        self.individual_boxes_fraction  = ARGS['individual_boxes_fraction']
        self.individual_boxes_size      = ARGS['individual_boxes_size']
        self.individual_boxes_max_index = ARGS['individual_boxes_max_index']

        # some sanity tests
        assert 0.0 <= self.individual_boxes_fraction <= 1.0

        # pretraining
        self.__pretraining_epochs = ARGS['pretraining_epochs']

        # breakpoints -- at those epochs, the network is saved separately
        self.__breakpoints = ARGS['breakpoints']

        # target transformation
        #   want the target transformation to happen on GPU, so push these there
        self.__tau   = ARGS['target_transformation_kw']['tau']
        self.__epoch_max = ARGS['target_transformation_kw']['epoch_max']
        self.__gamma = ARGS['target_transformation_kw']['gamma']
        self.__kappa = ARGS['target_transformation_kw']['kappa']
        self.__delta = ARGS['target_transformation_kw']['delta']

        # where to find and put data files
        self.__input_path  = ARGS['input_path']
        self.__individual_boxes_path = ARGS['individual_boxes_path']
        self.__output_path = ARGS['output_path']

        if GPU_AVAIL and not ARGS.debug and not ARGS.nocopy :
            print 'Starting to copy data to /tmp'
            system('cp %s%s/size_%d.hdf5 /tmp/'%(self.__input_path, ARGS.scaling, self.box_sidelength))
            self.data_path = '/tmp/size_%d.hdf5'%self.box_sidelength

            if self.individual_boxes_fraction > 0.0 :
                print 'Starting to copy individual boxes to /tmp'
                system('mkdir /tmp/individual_boxes')
                system('cp %s%s/size_%d/* /tmp/individual_boxes'%(self.__individual_boxes_path, ARGS.scaling, self.box_sidelength))
                self.individual_boxes_path = '/tmp/individual_boxes/'
            else :
                self.individual_boxes_path = None

            print 'Finished copying data to /tmp, took %.2e seconds'%(time()-START_TIME) # 4.06e+02 sec for 2048, ~12 sec for 1024
        else :
            self.data_path = '%s%s/size_%d.hdf5'%(self.__input_path, ARGS.scaling, self.box_sidelength)
            
            if self.individual_boxes_fraction > 0.0 :
                self.individual_boxes_path = '%s%s/size_%d/'%(self.__individual_boxes_path, ARGS.scaling, self.box_sidelength)
            else :
                self.individual_boxes_path = None

        # Some hardcoded values
        self.block_shapes = {'training':   (self.box_sidelength, self.box_sidelength           , (1428*self.box_sidelength)/2048),
                             'validation': (self.box_sidelength,(1368*self.box_sidelength)/2048, ( 620*self.box_sidelength)/2048),
                             'testing':    (self.box_sidelength,( 680*self.box_sidelength)/2048, ( 620*self.box_sidelength)/2048),
                            }
        self.box_size = 205.0 # Mpc/h

        # keep track of training
        self.training_steps = []
        self.validation_steps = []
        self.training_loss = []
        self.validation_loss = []

        # needs to be updated from the main code
        self.net = None
        self.optimizer = None
        self.lr_scheduler = None
    #}}}
    def loss_function_(self) :#{{{
        return self.__loss_function(**self.__loss_function_kw)
    #}}}
    def optimizer_(self) :#{{{
        return self.__optimizer(
            params = self.net.parameters(),
            **self.__optimizer_kw
            )
    #}}}
    def lr_scheduler_(self) :#{{{
        if self.__lr_scheduler is not None :
            return self.__lr_scheduler(
                self.optimizer,
                **self.__lr_scheduler_kw
                )
        else :
            return None
    #}}}
    def data_loader_(self, data) :#{{{
        return DataLoader(
            data,
            **self.__data_loader_kw
            )
    #}}}
    def update_training_loss(self, this_loss) :#{{{
        self.training_loss.append(this_loss)
        self.training_steps.append(len(self.training_loss))
    #}}}
    def update_validation_loss(self, this_loss) :#{{{
        self.validation_loss.append(this_loss)
        self.validation_steps.append(EPOCH)
    #}}}
    def pretraining(self) :#{{{
        return EPOCH <= self.__pretraining_epochs
    #}}}
    def target_as_model(self) :#{{{
        return EPOCH <= self.__pretraining_epochs/2
    #}}}
    def target_transformation(self, x) :#{{{
        __t = (
              torch.tensor(min(EPOCH, self.__epoch_max), requires_grad = False).to(x.device, dtype = torch.float32)
            / torch.tensor(self.__tau, requires_grad = False).to(x.device, dtype = torch.float32)
            )
        
        __gamma = torch.tensor(self.__gamma, requires_grad = False).to(x.device, dtype = torch.float32)
        __kappa = torch.tensor(self.__kappa, requires_grad = False).to(x.device, dtype = torch.float32)
        __delta = torch.tensor(self.__delta, requires_grad = False).to(x.device, dtype = torch.float32)
        
        __alpha = __delta + (1.0 - __delta) * torch.exp(- __t)
        return __kappa*__alpha*torch.log1p(x/__gamma) + (1.0-__alpha)*x
    #}}}
    def stop_training(self) :#{{{
        return (time()-START_TIME)/60./60. > ARGS.time
    #}}}
    def save_loss(self, name) :#{{{
        if not ARGS.debug :
            np.savez(
                self.__output_path+name,
                training_steps = self.training_steps,
                training_loss = self.training_loss,
                validation_steps = self.validation_steps,
                validation_loss = self.validation_loss,
                )
        else :
            if ARGS.verbose :
                print 'Not saving loss since in debugging mode.'
    #}}}
    def save_network(self, name, best = False) :#{{{
        if not ARGS.debug :
            if (not best) or (self.validation_loss[-1] < np.min(np.array(self.validation_loss)[:-1]) if len(self.validation_loss) > 1 else False) :
                __state = {
                    'network_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
                    'configs': ARGS.final_config,
                    'consistency': {
                        'epoch': EPOCH,
                        'box_sidelength': self.box_sidelength,
                        'DM_sidelength': self.DM_sidelength,
                        'gas_sidelength': self.gas_sidelength,
                        'scaling': ARGS.scaling,
                        },
                    }
                torch.save(__state, self.__output_path + name)
        else :
            if ARGS.verbose :
                print 'Not saving network since in debugging mode.'
    #}}}
    def load_network(self, name) :#{{{
        # need to initialize network first
        try :
            __state = torch.load(self.__output_path + name, map_location = {'cuda:0': 'cuda:0' if GPU_AVAIL else 'cpu', 'cpu': 'cpu'})
            self.net.load_state_dict(__state['network_state_dict'])
            if self.optimizer is not None :
                self.optimizer.load_state_dict(__state['optimizer_state_dict'])
                if ARGS.verbose :
                    print '\tLoaded optimizer state dict.'
            if self.lr_scheduler is not None :
                if __state['lr_scheduler_state_dict'] is not None :
                    self.lr_scheduler.load_state_dict(__state['lr_scheduler_state_dict'])
                    if ARGS.verbose :
                        print '\tLoaded lr scheduler state dict.'
                else :
                    if ARGS.verbose :
                        print '\tNo state dict for lr scheduler found in loaded network.'
            else :
                if __state['lr_scheduler_state_dict'] is not None :
                    if ARGS.verbose :
                        print '\tState dict for lr scheduler found but no lr scheduling requested in this run.'

            global EPOCH
            EPOCH = __state['consistency']['epoch']
            # Consistency checks
            assert self.box_sidelength == __state['consistency']['box_sidelength'], 'Box sidelength does not match.'
            assert self.DM_sidelength == __state['consistency']['DM_sidelength'], 'DM sidelength does not match.'
            assert self.gas_sidelength == __state['consistency']['gas_sidelength'], 'gas sidelength does not match.'
# TODO
#            assert ARGS.scaling == __state['consistency']['scaling'], 'scaling does not match.'
            if ARGS.verbose :
                print 'Loaded network %s from disk,\n\tstarting at epoch %d.'%(name, EPOCH)
        except IOError :
            if ARGS.mode == 'valid' :
                raise IOError('Trained network not found.')
            if ARGS.verbose :
                print 'Failed to load network %s from disk.\n Starting training with random initialization.'%name
    #}}}
    def breakpoint_reached(self) :#{{{
        return EPOCH in self.__breakpoints
    #}}}
#}}}

class Stepper(object) :#{{{
    def __init__(self, mode) :#{{{
        self.mode = mode
        self.xlength = GLOBDAT.block_shapes[mode][0] - GLOBDAT.DM_sidelength
        self.ylength = GLOBDAT.block_shapes[mode][1] - GLOBDAT.DM_sidelength
        self.zlength = GLOBDAT.block_shapes[mode][2] - GLOBDAT.DM_sidelength

        self.max_x_index = self.__max_index(self.xlength+1)
        self.max_y_index = self.__max_index(self.ylength+1)
        self.max_z_index = self.__max_index(self.zlength+1)
    #}}}
    def __max_index(self, total_length) :#{{{
        if total_length%GLOBDAT.gas_sidelength == 0 : # gas sidelength fits perfectly
            return total_length/GLOBDAT.gas_sidelength # CORRECT
        else : # need some patching at the end
            return total_length/GLOBDAT.gas_sidelength + 1 # CORRECT
    #}}}
    def __getitem__(self, index) :#{{{
        __xx = index%(self.max_x_index+1)
        __yy = (index/(self.max_x_index+1))%(self.max_y_index+1)
        __zz = index/((self.max_x_index+1)*(self.max_y_index+1))

        if __xx < self.max_x_index :
            __xx *= GLOBDAT.gas_sidelength
        elif __xx == self.max_x_index :
            __xx = self.xlength
        else :
            raise RuntimeError('I should not be here.')
        if __yy < self.max_y_index :
            __yy *= GLOBDAT.gas_sidelength
        elif __yy == self.max_y_index :
            __yy = self.ylength
        else :
            raise RuntimeError('I should not be here.')
        if __zz < self.max_z_index :
            __zz *= GLOBDAT.gas_sidelength
        elif __zz == self.max_z_index :
            __zz = self.zlength
        else :
            raise RuntimeError('I should not be here.')

        return __xx, __yy, __zz
    #}}}
    def __len__(self) :#{{{
        return (self.max_x_index+1)*(self.max_y_index+1)*(self.max_z_index+1)
    #}}}
#}}}

class PositionSelector(object) :#{{{
    def __init__(self, mode, seed = 0, **kwargs) :#{{{
        self.mode = mode

        self.xlength = GLOBDAT.block_shapes[self.mode][0] - GLOBDAT.DM_sidelength
        self.ylength = GLOBDAT.block_shapes[self.mode][1] - GLOBDAT.DM_sidelength
        self.zlength = GLOBDAT.block_shapes[self.mode][2] - GLOBDAT.DM_sidelength

        self.empty_fraction = kwargs['empty_fraction'] if 'empty_fraction' in kwargs else 1.0
        assert 0.0 <= self.empty_fraction <= 1.0
        if self.mode == 'training' :
            self.rnd_generator = np.random.RandomState((hash(str(clock()+seed))+hash(self.mode))%MAX_SEED)
        elif self.mode == 'validation' :
            self.rnd_generator = np.random.RandomState(seed)
        if self.empty_fraction < 1.0 :
            if ARGS.verbose :
                print 'In PositionSelector(%s) : Reading %s.'%(self.mode, kwargs['pos_mass_file'])
            with h5py.File(kwargs['pos_mass_file'], 'r') as f :
                self.pos  = f['/%s/coords'%self.mode][:] # kpc/h
                self.log_mass = np.log10(1e10*f['/%s/M500c'%self.mode][:]) # log Msun/h
            assert self.pos.shape[0] == self.log_mass.shape[0]
            assert self.pos.shape[1] == 3
            # cnvert to pixel coordinates
            self.pos = (self.pos*float(GLOBDAT.box_sidelength)/(1e3*GLOBDAT.box_size)).astype(int)
            assert np.max(self.pos[:,0]) < GLOBDAT.block_shapes[self.mode][0], '%d > %d'%(np.max(self.pos[:,0]),GLOBDAT.block_shapes[self.mode][0])
            assert np.max(self.pos[:,1]) < GLOBDAT.block_shapes[self.mode][1], '%d > %d'%(np.max(self.pos[:,1]),GLOBDAT.block_shapes[self.mode][1])
            assert np.max(self.pos[:,2]) < GLOBDAT.block_shapes[self.mode][2], '%d > %d'%(np.max(self.pos[:,2]),GLOBDAT.block_shapes[self.mode][2])

            # sort according to mass
            __sorting_indices = np.argsort(self.log_mass)
            self.log_mass = self.log_mass[__sorting_indices]
            self.pos = self.pos[__sorting_indices, :]
            # compute mass intervals symmetrically
            __dlog_mass_l = np.diff(self.log_mass)
            __dlog_mass_r = -np.diff(self.log_mass[::-1])[::-1]
            self.dlog_mass = 0.5 * (__dlog_mass_l[:-1] + __dlog_mass_r[1:])
            self.dlog_mass = np.concatenate((__dlog_mass_r[:1], self.dlog_mass))
            self.dlog_mass = np.concatenate((self.dlog_mass, __dlog_mass_l[-1:]))
            self.weights = eval(compile(kwargs['halo_weight_fct'], '<string>', 'eval'))(self.log_mass, self.dlog_mass)
            self.weights = np.broadcast_to(self.weights, self.log_mass.shape).copy()
            self.weights /= np.sum(self.weights) # normalize probabilities
            assert np.all(self.weights >= 0.0)
    #}}}
    def is_biased(self) :#{{{
        return self.empty_fraction < 1.0
    #}}}
    def __call__(self) :#{{{
        if self.rnd_generator.rand() < self.empty_fraction :
            xx_rnd = self.rnd_generator.randint(0, high = self.xlength+1)
            yy_rnd = self.rnd_generator.randint(0, high = self.ylength+1)
            zz_rnd = self.rnd_generator.randint(0, high = self.zlength+1)
        else :
            __rnd_halo_index = self.rnd_generator.choice(len(self.log_mass), p = self.weights)
            # displacements of halo center from center of the box
            __rnd_displacements = self.rnd_generator.randint(0, high = GLOBDAT.DM_sidelength, size = 3)
            xx_rnd = np.minimum(self.xlength, np.maximum(0, self.pos[__rnd_halo_index, 0] - __rnd_displacements[0]))
            yy_rnd = np.minimum(self.ylength, np.maximum(0, self.pos[__rnd_halo_index, 1] - __rnd_displacements[1]))
            zz_rnd = np.minimum(self.zlength, np.maximum(0, self.pos[__rnd_halo_index, 2] - __rnd_displacements[2]))
        return xx_rnd, yy_rnd, zz_rnd
    #}}}
#}}}

class InputData(Dataset) :#{{{
    def __init__(self, mode, stepper = None) :#{{{
        __modes = ['training', 'validation', 'testing', ]
        __types = ['DM', 'gas', 'gas_model', ]

        self.mode = mode
        self.stepper = stepper
        assert self.mode in __modes

        self.do_random_transformations = True if self.stepper is None else False

        self.xlength = GLOBDAT.block_shapes[mode][0] - GLOBDAT.DM_sidelength
        self.ylength = GLOBDAT.block_shapes[mode][1] - GLOBDAT.DM_sidelength
        self.zlength = GLOBDAT.block_shapes[mode][2] - GLOBDAT.DM_sidelength

        self.files = []
        self.individual_box_files = []
        self.datasets = {}
        self.individual_box_datasets = {}
        self.individual_boxes_Nfiles = 0
        for t in __types :
            self.datasets[t] = []
            self.individual_box_datasets[t] = []
        self.position_selectors = []
        self.rnd_generators = []
        for ii in xrange(max(GLOBDAT.num_workers, 1)) :
            self.files.append(h5py.File(GLOBDAT.data_path, 'r', driver='mpio', comm=MPI.COMM_WORLD))
            for t in __types :
                self.datasets[t].append(self.files[-1]['%s/%s'%(t, self.mode)])
            self.position_selectors.append(PositionSelector(
                self.mode, ii, **GLOBDAT.sample_selector_kw
                ))
            if self.mode == 'training' :
                self.rnd_generators.append(np.random.RandomState((hash(str(clock()+ii))+hash(self.mode))%MAX_SEED))
            elif self.mode == 'validation' :
                self.rnd_generators.append(np.random.RandomState(ii)) # use always the same seed for validation set (easier comparison)
        
        if GLOBDAT.individual_boxes_fraction > 0.0 and self.mode == 'training' :
            for jj in xrange(GLOBDAT.individual_boxes_max_index + 1) :
                if isfile('%shnew%d.hdf5'%(GLOBDAT.individual_boxes_path, jj)) :
                    self.individual_boxes_Nfiles += 1
                    for ii in xrange(max(GLOBDAT.num_workers, 1)) :
                        self.individual_box_files.append(h5py.File('%shnew%d.hdf5'%(GLOBDAT.individual_boxes_path, jj),
                                                         'r', driver='mpio', comm=MPI.COMM_WORLD))
                        for t in __types :
                            self.individual_box_datasets[t].append(self.individual_box_files[-1]['%s/%s'%(t, self.mode)])
            assert self.individual_boxes_Nfiles > 0, 'No individual boxes found.'

        # read the backward transformations
        self.get_back = {}
        for t in __types :
            self.get_back[t] = eval(compile(self.files[0]['%s/get_back'%t].attrs['function'], '<string>', 'eval'))
            # test the transformation
            assert np.fabs(self.get_back[t](self.files[0]['%s/get_back'%t].attrs['test_in'])/self.files[0]['%s/get_back'%t].attrs['test_out'] - 1.0) < 1e-3, t

        # sanity check
        assert self.stepper is None if self.position_selectors[0].is_biased() else True

        self.xx_indices_rnd = None
        self.yy_indices_rnd = None
        self.zz_indices_rnd = None
    #}}}
    def generate_rnd_indices(self) :#{{{
        if ARGS.verbose :
            print 'In InputData(%s) : generate_rnd_indices'%self.mode
        self.xx_indices_rnd = np.empty(GLOBDAT.Nsamples[self.mode], dtype=int)
        self.yy_indices_rnd = np.empty(GLOBDAT.Nsamples[self.mode], dtype=int)
        self.zz_indices_rnd = np.empty(GLOBDAT.Nsamples[self.mode], dtype=int)
        for ii in xrange(GLOBDAT.Nsamples[self.mode]) :
            self.xx_indices_rnd[ii], self.yy_indices_rnd[ii], self.zz_indices_rnd[ii] = self.position_selectors[0]()
    #}}}
    def __randomly_transform(self, arr1, arr2, arr3, __ID) :#{{{
        # reflections
        if self.rnd_generators[__ID].rand() < 0.5 :
            arr1 = arr1[::-1,:,:]
            arr2 = arr2[::-1,:,:]
            arr3 = arr3[::-1,:,:]
        if self.rnd_generators[__ID].rand() < 0.5 :
            arr1 = arr1[:,::-1,:]
            arr2 = arr2[:,::-1,:]
            arr3 = arr3[:,::-1,:]
        if self.rnd_generators[__ID].rand() < 0.5 :
            arr1 = arr1[:,::-1]
            arr2 = arr2[:,::-1]
            arr3 = arr3[:,::-1]
        # transpositions
        prand = self.rnd_generators[__ID].rand()
        if prand < 1./6. :
            arr1 = np.transpose(arr1, axes=(0,2,1))
            arr2 = np.transpose(arr2, axes=(0,2,1))
            arr3 = np.transpose(arr3, axes=(0,2,1))
        elif prand < 2./6. :
            arr1 = np.transpose(arr1, axes=(2,1,0))
            arr2 = np.transpose(arr2, axes=(2,1,0))
            arr3 = np.transpose(arr3, axes=(2,1,0))
        elif prand < 3./6. :
            arr1 = np.transpose(arr1, axes=(1,0,2))
            arr2 = np.transpose(arr2, axes=(1,0,2))
            arr3 = np.transpose(arr3, axes=(1,0,2))
        elif prand < 4./6. :
            arr1 = np.transpose(arr1, axes=(2,0,1))
            arr2 = np.transpose(arr2, axes=(2,0,1))
            arr3 = np.transpose(arr3, axes=(2,0,1))
        elif prand < 5./6. :
            arr1 = np.transpose(arr1, axes=(1,2,0))
            arr2 = np.transpose(arr2, axes=(1,2,0))
            arr3 = np.transpose(arr3, axes=(1,2,0))
        return arr1, arr2, arr3
    #}}}
    def __index_3D(self, index) :#{{{
        return (
            index%self.xlength,
            (index/self.xlength)%self.ylength,
            index/(self.ylength*self.xlength)
            )
    #}}}
    def getitem_deterministic(self, xx, yy, zz, __ID) :#{{{
        assert isinstance(xx, int)
        assert isinstance(yy, int)
        assert isinstance(zz, int)
        assert xx <= self.xlength
        assert yy <= self.ylength
        assert zz <= self.zlength
        DM = self.datasets['DM'][__ID][
            xx : xx+GLOBDAT.DM_sidelength,
            yy : yy+GLOBDAT.DM_sidelength,
            zz : zz+GLOBDAT.DM_sidelength
            ]
        gas = self.datasets['gas'][__ID][
            xx+(GLOBDAT.DM_sidelength-GLOBDAT.gas_sidelength)/2 : xx+(GLOBDAT.DM_sidelength+GLOBDAT.gas_sidelength)/2,
            yy+(GLOBDAT.DM_sidelength-GLOBDAT.gas_sidelength)/2 : yy+(GLOBDAT.DM_sidelength+GLOBDAT.gas_sidelength)/2,
            zz+(GLOBDAT.DM_sidelength-GLOBDAT.gas_sidelength)/2 : zz+(GLOBDAT.DM_sidelength+GLOBDAT.gas_sidelength)/2,
            ]
        gas_model = self.datasets['gas_model'][__ID][
            xx+(GLOBDAT.DM_sidelength-GLOBDAT.gas_sidelength)/2 : xx+(GLOBDAT.DM_sidelength+GLOBDAT.gas_sidelength)/2,
            yy+(GLOBDAT.DM_sidelength-GLOBDAT.gas_sidelength)/2 : yy+(GLOBDAT.DM_sidelength+GLOBDAT.gas_sidelength)/2,
            zz+(GLOBDAT.DM_sidelength-GLOBDAT.gas_sidelength)/2 : zz+(GLOBDAT.DM_sidelength+GLOBDAT.gas_sidelength)/2,
            ]
        return DM, gas, gas_model
    #}}}
    def get_from_individual_box(self, __ID) :#{{{
        # TODO there's a lot of code duplication with the previous function,
        #      can maybe simplify this ?
        indx = self.rnd_generators[__ID].randint(0, self.individual_boxes_Nfiles)
        xx = self.rnd_generators[__ID].randint(0, GLOBDAT.individual_boxes_size - GLOBDAT.DM_sidelength)
        yy = self.rnd_generators[__ID].randint(0, GLOBDAT.individual_boxes_size - GLOBDAT.DM_sidelength)
        zz = self.rnd_generators[__ID].randint(0, GLOBDAT.individual_boxes_size - GLOBDAT.DM_sidelength)
        DM = self.individual_box_datasets['DM'][__ID + indx * max(GLOBDAT.num_workers, 1)][
            xx : xx+GLOBDAT.DM_sidelength,
            yy : yy+GLOBDAT.DM_sidelength,
            zz : zz+GLOBDAT.DM_sidelength
            ]
        gas = self.individual_box_datasets['gas'][__ID + indx * max(GLOBDAT.num_workers, 1)][
            xx+(GLOBDAT.DM_sidelength-GLOBDAT.gas_sidelength)/2 : xx+(GLOBDAT.DM_sidelength+GLOBDAT.gas_sidelength)/2,
            yy+(GLOBDAT.DM_sidelength-GLOBDAT.gas_sidelength)/2 : yy+(GLOBDAT.DM_sidelength+GLOBDAT.gas_sidelength)/2,
            zz+(GLOBDAT.DM_sidelength-GLOBDAT.gas_sidelength)/2 : zz+(GLOBDAT.DM_sidelength+GLOBDAT.gas_sidelength)/2,
            ]
        gas_model = self.individual_box_datasets['gas_model'][__ID + indx * max(GLOBDAT.num_workers, 1)][
            xx+(GLOBDAT.DM_sidelength-GLOBDAT.gas_sidelength)/2 : xx+(GLOBDAT.DM_sidelength+GLOBDAT.gas_sidelength)/2,
            yy+(GLOBDAT.DM_sidelength-GLOBDAT.gas_sidelength)/2 : yy+(GLOBDAT.DM_sidelength+GLOBDAT.gas_sidelength)/2,
            zz+(GLOBDAT.DM_sidelength-GLOBDAT.gas_sidelength)/2 : zz+(GLOBDAT.DM_sidelength+GLOBDAT.gas_sidelength)/2,
            ]
        return DM, gas, gas_model
    #}}}
    def __getitem__(self, index) :#{{{
        __ID = get_worker_info().id if GLOBDAT.num_workers > 0 else 0
        __got_from_individual_box = False
        if self.stepper is not None :
            assert ARGS.mode != 'train', 'It does not make sense to have a stepper in training mode.'
            assert self.mode != 'training', 'It does not make sense to have a stepper in training mode.'
            __xx, __yy, __zz = self.stepper[index]
        elif self.mode == 'traning' and self.rnd_generators[__ID].rand() < GLOBDAT.individual_boxes_fraction :
            __xx, __yy, __zz = 0.0, 0.0, 0.0
            __got_from_individual_box = True
            DM, gas, gas_model = self.get_from_individual_box(__ID)
        elif self.xx_indices_rnd is None or self.yy_indices_rnd is None or self.zz_indices_rnd is None :
            __xx, __yy, __zz = self.position_selectors[__ID]()
        else :
            assert self.mode != 'training', 'This will produce always the same samples, not recommended in training mode.'
            __xx, __yy, __zz = self.xx_indices_rnd[index], self.yy_indices_rnd[index], self.zz_indices_rnd[index]

        if not __got_from_individual_box :
            DM, gas, gas_model = self.getitem_deterministic(
                __xx, __yy, __zz,
                __ID
                )

        if self.do_random_transformations :
            DM, gas, gas_model = self.__randomly_transform(DM, gas, gas_model, __ID)
        if GLOBDAT.gas_noise > 0.0 and self.mode == 'training' :
            # the first condition makes sure we don't have numerical problems
            # the second condition makes sure that the validation loss is a true representation
            #   (since the artificial noise is only meant to facilitate training)
            gas *= self.rnd_generators[__ID].normal(1.0, GLOBDAT.gas_noise, size = gas.shape)
            gas_model *= self.rnd_generators[__ID].normal(1.0, GLOBDAT.gas_noise, size = gas_model.shape)
        if GLOBDAT.DM_noise > 0.0 and self.mode == 'training' :
            DM *= self.rnd_generators[__ID].normal(1.0, GLOBDAT.DM_noise, size = DM.shape)
        assert DM.shape[0]  == DM.shape[1]  == DM.shape[2]  == GLOBDAT.DM_sidelength,  DM.shape
        assert gas.shape[0] == gas.shape[1] == gas.shape[2] == GLOBDAT.gas_sidelength, gas.shape
        assert gas_model.shape[0] == gas_model.shape[1] == gas_model.shape[2] == GLOBDAT.gas_sidelength, gas_model.shape

        # TODO why do we need to .copy() again?
#        if GLOBDAT.target_as_model() :
#            return (
#                torch.from_numpy(DM.copy()).unsqueeze(0),
#                torch.from_numpy(gas.copy()).unsqueeze(0),
#                torch.from_numpy(gas.copy()).unsqueeze(0),
#                torch.from_numpy(np.array([__xx, __yy, __zz], dtype=int)),
#                )
#        else :
        return (
            torch.from_numpy(DM.copy()).unsqueeze(0),
            torch.from_numpy(gas.copy()).unsqueeze(0),
            torch.from_numpy(gas_model.copy()).unsqueeze(0),
            torch.from_numpy(np.array([__xx, __yy, __zz], dtype=int)),
            )
        # add fake dimension (channel dimension is 0 here)
        """
        return (
            torch.from_numpy(np.empty((64,64,64))).unsqueeze(0),
            torch.from_numpy(np.empty((32,32,32))).unsqueeze(0),
            torch.from_numpy(np.array([__xx, __yy, __zz])),
            )
        """
    #}}}
    def __len__(self) :#{{{
        if self.stepper is None :
            return GLOBDAT.Nsamples[self.mode]
        else :
            return len(self.stepper)
    #}}}
    def __enter__(self) :#{{{
        return self
    #}}}
    def __exit__(self, exc_type, exc_value, exc_traceback) :#{{{
        for f in self.files :
            f.close()
        for f in self.individual_box_files :
            f.close()
    #}}}
#}}}

class Identity(nn.Module) :#{{{
    def __init__(self) :
        super(Identity, self).__init__()
    def forward(self, x) :
        return x
#}}}

class BasicLayer(nn.Module) :#{{{
    __default_param = {#{{{
        'inplane': None,
        'outplane': None,

        'conv': 'Conv',
        'conv_kw': {
            'stride': 1,
            'padding': 1,
            'kernel_size': 3,
            'bias': True,
            },

        'batch_norm': 'BatchNorm',
        'batch_norm_kw': {
            'momentum': 0.1,
            },

        'activation': 'ReLU',
        'activation_kw': {
            'inplace': False,
            },

        'crop_output': False,

        'dropout': False,
        'dropout_kw': {
            'p': 0.5,
            'inplace': False,
            },
        }
    #}}}
    @staticmethod
    def __crop_tensor(x, w) :#{{{
        x = x.narrow(2,w/2,x.shape[2]-w)
        x = x.narrow(3,w/2,x.shape[3]-w)
        x = x.narrow(4,w/2,x.shape[4]-w)
        return x.contiguous()
    #}}}
    def __init__(self, layer_dict) :#{{{
        super(BasicLayer, self).__init__()
        self.__merged_dict = _merge(
            layer_dict,
            copy.deepcopy(BasicLayer.__default_param)
            )

        if self.__merged_dict['conv'] is not None :
            self.__conv_fct = _namesofplaces[self.__merged_dict['conv']](
                self.__merged_dict['inplane'], self.__merged_dict['outplane'],
                **self.__merged_dict['conv_kw']
                )
        else :
            self.__conv_fct = Identity()

        if self.__merged_dict['crop_output'] :
            self.__crop_fct = lambda x : BasicLayer.__crop_tensor(x, self.__merged_dict['crop_output'])
        else :
            self.__crop_fct = Identity()

        if self.__merged_dict['dropout'] :
            self.__dropout_fct = nn.Dropout3d(
                **self.__merged_dict['dropout_kw']
                )
        else :
            self.__dropout_fct = Identity()

        if self.__merged_dict['batch_norm'] is not None :
            self.__batch_norm_fct = _namesofplaces[self.__merged_dict['batch_norm']](
                self.__merged_dict['outplane'],
                **self.__merged_dict['batch_norm_kw']
                )
        else :
            self.__batch_norm_fct = Identity()
        
        if self.__merged_dict['activation'] is not None :
            self.__activation_fct = _namesofplaces[self.__merged_dict['activation']](
                **self.__merged_dict['activation_kw']
                )
        else :
            self.__activation_fct = Identity()
    #}}}
    def forward(self, x) :#{{{
        x = self.__activation_fct(self.__batch_norm_fct(self.__dropout_fct(self.__crop_fct(self.__conv_fct(x)))))
        return x
    #}}}
#}}}

class Network(nn.Module) :#{{{
    def __init__(self, network_dict) :#{{{
        super(Network, self).__init__()
        self.network_dict = network_dict

        self.__blocks = nn.ModuleList()
        # even index blocks are in, odd are out, the last one is the bottom through block
        # last index is 2*(NLevels-1)
        for ii in xrange(self.network_dict['NLevels']-1) :
            if ii < self.network_dict['NLevels'] - 1 : # not in the bottom block
                self.__blocks.append(
                    Network.__feed_forward_block(
                        self.network_dict['Level_%d'%ii]['in']
                        )
                    )
                self.__blocks.append(
                    Network.__feed_forward_block(
                        self.network_dict['Level_%d'%ii]['out']
                        )
                    )
        self.__blocks.append(
            Network.__feed_forward_block(
                self.network_dict['Level_%d'%(self.network_dict['NLevels']-1)]['through']
                )
            )

        if 'model_block' in self.network_dict :
            if not self.network_dict['feed_model'] :
                raise RuntimeError('You provided a model block but do not require model feed. Aborting.')
            self.__model_block = Network.__feed_forward_block(
                self.network_dict['model_block']
                )
        else :
            self.__model_block = None

        self.is_frozen = False
    #}}}
    @staticmethod
    def __feed_forward_block(input_list) :#{{{
        layers = []
        for layer_dict in input_list :
            layers.append(BasicLayer(layer_dict))
        return nn.Sequential(*layers)
    #}}}
    @staticmethod
    def __crop_tensor(x, w) :#{{{
        x = x.narrow(2,w/2,x.shape[2]-w)
        x = x.narrow(3,w/2,x.shape[3]-w)
        x = x.narrow(4,w/2,x.shape[4]-w)
        return x.contiguous()
    #}}}
    def __freeze(self) :#{{{
        for ii in xrange(2*self.network_dict['NLevels'] - 1) :
            if ii == 1 : # this is the last block
                continue
            for p in self.__blocks[ii].parameters() :
                p.requires_grad = False
    #}}}
    def __thaw(self) :#{{{
        for ii in xrange(2*self.network_dict['NLevels'] - 1) :
            if ii == 1 : # this is the last block
                continue
            for p in self.__blocks[ii].parameters() :
                p.requires_grad = True
    #}}}
    def __update_mode(self) :#{{{
        if GLOBDAT.pretraining() and not self.is_frozen :
            self.__freeze()
            self.is_frozen = True
            if ARGS.verbose :
                print 'Network frozen (apart from last block).'
        elif not GLOBDAT.pretraining() and self.is_frozen :
            self.__thaw()
            self.is_frozen = False
            if ARGS.verbose :
                print 'Network thawed.'
    #}}}
    def forward(self, x, xmodel) :#{{{
# TODO
#        self.__update_mode()
#        if not GLOBDAT.pretraining() :
        if True :
            intermediate_data = []

            # contracting path
            for ii in xrange(self.network_dict['NLevels']-1) :
                x = self.__blocks[2*ii](x)
                if self.network_dict['Level_%d'%ii]['concat'] :
                    intermediate_data.append(x.clone())
                else :
                    intermediate_data.append(None)

            # bottom level
            x = self.__blocks[2*(self.network_dict['NLevels']-1)](x)

            # expanding path
            for ii in xrange(self.network_dict['NLevels']-2, -1, -1) :
                if self.network_dict['Level_%d'%ii]['concat'] :
                    if self.network_dict['Level_%d'%ii]['resize_to_gas'] :
                        intermediate_data[ii] = Network.__crop_tensor(
                            intermediate_data[ii],
                            intermediate_data[ii].shape[-1]/GLOBDAT.DM_sidelength * (GLOBDAT.DM_sidelength - GLOBDAT.gas_sidelength) # TODO CHECK
                            )
                    x = torch.cat((x, intermediate_data[ii]), dim = 1)
                if ii == 0 and self.network_dict['feed_model'] :
                    if self.__model_block is not None :
                        xmodel = torch.cat((xmodel, self.__model_block(xmodel)), dim = 1)
                        # include a skip connection
                    x = torch.cat((x, xmodel), dim = 1)
                x = self.__blocks[2*ii+1](x)

        else :
            raise NotImplementedError('Currently this path is not up to date.')
            x = torch.cat(
                (torch.zeros(
                    xmodel.size()[0],
                    self.network_dict['Level_1']['out'][-1]['outplane'],
                    xmodel.size()[2], xmodel.size()[3], xmodel.size()[4]
                    ).to(xmodel.device),
                xmodel
                ), dim = 1)
            x = self.__blocks[2*0+1](x) # apply only the last block
        return x
    #}}}
#}}}

class Mesh(object) :#{{{
    def __init__(self, arr) :#{{{
        self.arr = arr
        self.mesh = None
    #}}}
    def read_mesh(self) :#{{{
        if self.mesh is not None :
            print 'Already read mesh, No action taken.'
        if self.arr is None :
            raise RuntimeError('Not set the array yet for Mesh.')
        else :
            self.mesh = ArrayMesh(
                self.arr - np.mean(self.arr),
                BoxSize=(GLOBDAT.box_size/float(GLOBDAT.box_sidelength))*np.array(self.arr.shape)
                )
    #}}}
    def compute_powerspectrum(self, axis = None) :#{{{
        if self.mesh is None :
            raise RuntimeError('You need to read the mesh to memory first.')
        if axis is None :
            return FFTPower(self.mesh, mode = '1d')
        else :
            assert isinstance(axis, int)
            assert 0 <= axis < 3
            axes = [0, 1, 2]
            axes.remove(axis)
            axes = tuple(axes)
            return ProjectedFFTPower(self.mesh, axes = axes)
    #}}}
    def cross_power(self, other) :#{{{
        if self.mesh is None or other.mesh is None :
            raise RuntimeError('One of the meshes you passed is not read to memory yet.')
        return FFTPower(self.mesh, second = other.mesh, mode = '1d')
    #}}}
#}}}

class Analysis(object) :#{{{
    def __init__(self, data) :#{{{
        # data is instance of InputData
        self.data = data

        self.original_field  = None
        self.predicted_field = None

        self.original_power_spectrum = None
        self.predicted_power_spectrum = None
        self.original_projpower_spectrum = None
        self.predicted_projpower_spectrum = None
        self.cross = None

        self.mean_all_original = None
        self.mean_all_predicted = None
    #}}}
    @staticmethod
    def apodize(arr) :#{{{
        __L = 100
        __ramp = np.linspace(0., 1., num = __L)
        arr[:+__L,:,:] *= __ramp[::+1,None,None]
        arr[-__L:,:,:] *= __ramp[::-1,None,None]
        arr[:,:+__L,:] *= __ramp[None,::+1,None]
        arr[:,-__L:,:] *= __ramp[None,::-1,None]
        arr[:,:,:+__L] *= __ramp[None,None,::+1]
        arr[:,:,-__L:] *= __ramp[None,None,::-1]
        return arr
    #}}}
    def read_original(self) :#{{{
        self.original_field = self.data.get_back['gas'](
            self.data.datasets['gas'][0][
                (GLOBDAT.DM_sidelength-GLOBDAT.gas_sidelength)/2 : self.data.xlength+(GLOBDAT.DM_sidelength+GLOBDAT.gas_sidelength)/2,
                (GLOBDAT.DM_sidelength-GLOBDAT.gas_sidelength)/2 : self.data.ylength+(GLOBDAT.DM_sidelength+GLOBDAT.gas_sidelength)/2,
                (GLOBDAT.DM_sidelength-GLOBDAT.gas_sidelength)/2 : self.data.zlength+(GLOBDAT.DM_sidelength+GLOBDAT.gas_sidelength)/2,
                ]
            )
#        self.original_field_mesh = Mesh(Analysis.apodize(self.original_field))
        self.original_field_mesh = Mesh(self.original_field)
        self.original_field_mesh.read_mesh()
    #}}}predicted_projpower_spectrum
    def predict_whole_volume(self) :#{{{
        # initialize prediction to -1 everywhere to check whether the whole volume
        # really is filled.
        self.predicted_field = (-1.0)*np.ones(
            (self.data.xlength+GLOBDAT.gas_sidelength, self.data.ylength+GLOBDAT.gas_sidelength, self.data.zlength+GLOBDAT.gas_sidelength),
            dtype = np.float32
            )
        loader = GLOBDAT.data_loader_(self.data)
        for t, data in enumerate(loader) :
            with torch.no_grad() :
                pred = GLOBDAT.net(
                    torch.autograd.Variable(data[0].to(DEVICE), requires_grad = False),
                    torch.autograd.Variable(data[2].to(DEVICE), requires_grad = False)
                    )[:,0,...].cpu().numpy()
            __coords = data[-1].numpy()
            for ii in xrange(pred.shape[0]) : # loop over batch dimension
                self.predicted_field[
                    __coords[ii,0] : __coords[ii,0] + GLOBDAT.gas_sidelength,
                    __coords[ii,1] : __coords[ii,1] + GLOBDAT.gas_sidelength,
                    __coords[ii,2] : __coords[ii,2] + GLOBDAT.gas_sidelength
                    ] = self.data.get_back['gas'](pred[ii,...])
#        self.predicted_field_mesh = Mesh(Analysis.apodize(self.predicted_field))

        self.predicted_field_mesh = Mesh(self.predicted_field)
        self.predicted_field_mesh.read_mesh()
    #}}}
    def compute_powerspectrum(self, mode) :#{{{
        if mode is 'original' :
            r = self.original_field_mesh.compute_powerspectrum()
            self.original_power_spectrum = {
                'k': r.power['k'],
                'P': r.power['power'].real,
                }
            np.savez(
                ARGS['summary_path']+'%s_%s_%d.npz'%(ARGS.powerspectrumfid, ARGS.scaling, ARGS['box_sidelength']),
                **self.original_power_spectrum
                )
            if ARGS.verbose :
                print 'Computed original power spectrum and saved to %s.npz'%ARGS.powerspectrumfid
        elif mode is 'predicted' :
            r = self.predicted_field_mesh.compute_powerspectrum()
            self.predicted_power_spectrum = {
                'k': r.power['k'],
                'P': r.power['power'].real,
                }
            np.savez(
                ARGS['summary_path']+'%s_%s_%d.npz'%(ARGS.powerspectrumpred, ARGS.output, ARGS['box_sidelength']),
                **self.predicted_power_spectrum
                )
            if ARGS.verbose :
                print 'Computed predicted power spectrum and saved to %s_%s.npz'%(ARGS.powerspectrumpred, ARGS.output)
        else :
            raise RuntimeError('Invalid mode in compute_powerspectrum, only original and predicted allowed.')
    #}}}
    def compute_projected_powerspectrum(self, mode) :#{{{
        if mode is 'original' :
            r = self.original_field_mesh.compute_powerspectrum(2)
            self.original_projpower_spectrum = {
                'k': r.power['k'],
                'P': r.power['power'].real,
                }
            np.savez(
                ARGS['summary_path']+'%s_%s_%d.npz'%(ARGS.projpowerspectrumfid, ARGS.scaling, ARGS['box_sidelength']),
                **self.original_projpower_spectrum
                )
            if ARGS.verbose :
                print 'Computed original projected power spectrum and saved to %s_%s_%d.npz'%(ARGS.projpowerspectrumfid, ARGS.scaling, ARGS['box_sidelength'])
        elif mode is 'predicted' :
            r = self.predicted_field_mesh.compute_powerspectrum(2)
            self.predicted_projpower_spectrum = {
                'k': r.power['k'],
                'P': r.power['power'].real,
                }
            np.savez(
                ARGS['summary_path']+'%s_%s_%d.npz'%(ARGS.projpowerspectrumpred, ARGS.output, ARGS['box_sidelength']),
                **self.predicted_projpower_spectrum
                )
            if ARGS.verbose :
                print 'Computed predicted projected power spectrum and saved to %s_%s_%d.npz'%(ARGS.projpowerspectrumpred, ARGS.output, ARGS['box_sidelength'])
        else :
            raise RuntimeError('Invalid mode in compute_powerspectrum, only original and predicted allowed.')
    #}}}
    def compute_correlation_coeff(self) :#{{{
        r = self.original_field_mesh.cross_power(self.predicted_field_mesh)
        assert np.allclose(r.power['k'], self.original_power_spectrum['k'])
        assert np.allclose(r.power['k'], self.predicted_power_spectrum['k'])
        self.cross = {
            'k': r.power['k'],
            'r': r.power['power'].real/np.sqrt(self.original_power_spectrum['P']*self.predicted_power_spectrum['P']),
            }
        np.savez(
            ARGS['summary_path']+'%s_%s_%d.npz'%(ARGS.crosspower, ARGS.output, ARGS['box_sidelength']),
            **self.cross
            )
        if ARGS.verbose :
            print 'Computed correlation coefficient and saved to %s_%s_%d.npz'%(ARGS.crosspower, ARGS.output, ARGS['box_sidelength'])

    #}}}
    def compute_onepoint(self, mode) :#{{{
        if mode is 'original' :
            h, edges = np.histogram(
                self.original_field,
                bins = np.linspace(1e-2, 1e1, num = 101),
                density = False,
                )
            h = h.astype(float)/float(self.original_field.size)
            self.mean_all_original = np.mean(self.original_field)
            std_all  = np.std(self.original_field)
            mean_high = np.mean(self.original_field[self.original_field>std_all])
            np.savez(
                ARGS['summary_path']+'%s_%s_%d.npz'%(ARGS.onepointfid, ARGS.scaling, ARGS['box_sidelength']),
                h = h,
                edges = edges,
                mean_all = self.mean_all_original,
                std_all = std_all,
                mean_high = mean_high,
                )
            if ARGS.verbose :
                print 'Computed fiducial one-point PDF and saved to %s.npz'%ARGS.onepointfid
        elif mode is 'predicted' :
            h, edges = np.histogram(
                self.predicted_field,
                bins = np.linspace(1e-2, 1e1, num = 101),
                density = False,
                )
            h = h.astype(float)/float(self.predicted_field.size)
            self.mean_all_predicted = np.mean(self.predicted_field)
            std_all = np.std(self.predicted_field)
            mean_high = np.mean(self.predicted_field[self.predicted_field>std_all])
            np.savez(
                ARGS['summary_path']+'%s_%s_%d.npz'%(ARGS.onepointpred, ARGS.output, ARGS['box_sidelength']),
                h = h,
                edges = edges,
                mean_all = self.mean_all_predicted,
                std_all = std_all,
                mean_high = mean_high,
                )
            if ARGS.verbose :
                print 'Computed predicted one-point PDF and saved to %s_%s.npz'%(ARGS.onepointpred, ARGS.output)
    #}}}
    def save_predicted_volume(self) :#{{{
        np.save(
            ARGS['summary_path']+'%s_%s_%d'%(ARGS.fieldpred, ARGS.output, ARGS['box_sidelength']),
            self.predicted_field
            )
    #}}}
#}}}

if __name__ == '__main__' :
    # set global variables#{{{
    global START_TIME
    global GPU_AVAIL
    global DEVICE
    global ARGS
    global GLOBDAT

    START_TIME = time()
    GPU_AVAIL = torch.cuda.is_available()
    if GPU_AVAIL :
        print 'Found %d GPUs.'%torch.cuda.device_count()
        torch.set_num_threads(4)
        DEVICE = torch.device('cuda:0')
    else :
        print 'Could not find GPU.'
        torch.set_num_threads(2)
        DEVICE = torch.device('cpu')
    ARGS = _ArgParser()
    GLOBDAT = GlobalData()
    #}}}

    # OUTPUT VALIDATION
    if ARGS.mode == 'valid' :#{{{
# TODO
#        try :
        GLOBDAT.net = nn.DataParallel(Network(import_module(ARGS.network).this_network))
        GLOBDAT.load_network('trained_network_%s.pt'%ARGS.output)
#        except RuntimeError :
#            GLOBDAT.net = Network(import_module(ARGS.network).this_network)
#            GLOBDAT.load_network('trained_network_%s.pt'%ARGS.output)

        GLOBDAT.net.to(DEVICE)
        GLOBDAT.net.eval()
            
        # VISUAL OUTPUT INSPECTION
        if False :#{{{
            with InputData('validation') as validation_set :
                validation_loader = GLOBDAT.data_loader_(validation_set)

                NPAGES = 4
                for t, data in enumerate(validation_loader) :
                    if t == NPAGES : break

                    with torch.no_grad() :
                        targ = copy.deepcopy(data[1])
                        orig = copy.deepcopy(data[0])
                        mdel = copy.deepcopy(data[2])
                        pred = GLOBDAT.net(
                            torch.autograd.Variable(data[0], requires_grad=False),
                            torch.autograd.Variable(data[2], requires_grad=False),
                            )
                    
                    NPlots = 8
                    fig = plt.figure(figsize=(8.5, 11.0))
                    gs0 = gs.GridSpec(NPlots/2, 2, figure=fig, wspace = 0.2)
                    for ii in xrange(NPlots) :
                        plane_gas = np.random.randint(0, high=GLOBDAT.gas_sidelength)
                        plane_DM  = plane_gas + (GLOBDAT.DM_sidelength-GLOBDAT.gas_sidelength)/2

                        gs00 = gs.GridSpecFromSubplotSpec(
                            2, 2, subplot_spec=gs0[ii%(NPlots/2), ii/(NPlots/2)],
                            wspace = 0.1
                            )
                        ax_orig = plt.subplot(gs00[0, 0])
                        ax_targ = plt.subplot(gs00[0, 1])
                        ax_mdel = plt.subplot(gs00[1, 0])
                        ax_pred = plt.subplot(gs00[1, 1])
                        fig.add_subplot(ax_orig)
                        fig.add_subplot(ax_mdel)
                        fig.add_subplot(ax_targ)
                        fig.add_subplot(ax_pred)

        #                if ARGS.scaling=='log' :
                        if True :
                            ax_orig.matshow(
                                orig[ii,0,plane_DM,:,:].numpy(),
                                extent=(0, GLOBDAT.DM_sidelength, 0, GLOBDAT.DM_sidelength),
                                )
                            ax_targ.matshow(
                                targ[ii,0,plane_gas,:,:].numpy(),
                                )
                            ax_mdel.matshow(
                                mdel[ii,0,plane_gas,:,:].numpy(),
                                vmin = ax_targ.get_images()[-1].get_clim()[0],
                                vmax = ax_targ.get_images()[-1].get_clim()[1],
                                )
                            ax_pred.matshow(
                                pred[ii,0,plane_gas,:,:].numpy(),
                                vmin = ax_targ.get_images()[-1].get_clim()[0],
                                vmax = ax_targ.get_images()[-1].get_clim()[1],
                                )
        #                elif ARGS.scaling=='linear' :
        #                    ax_orig.matshow(
        #                        np.log(np.sum(np.exp(orig[ii,0,:,:,:].numpy()), axis=0)),
        #                        extent=(0, GLOBDAT.DM_sidelength, 0, GLOBDAT.DM_sidelength),
        #                        )
        #                    ax_targ.matshow(
        #                        np.log(np.sum(targ[ii,0,:,:,:].numpy(), axis=0)),
        #                        )
        #                    ax_pred.matshow(
        #                        np.log(np.sum(pred[ii,0,:,:,:].numpy(), axis=0)),
        #                        )

                        rect = Rectangle(
                            (
                                (GLOBDAT.DM_sidelength-GLOBDAT.gas_sidelength)/2,
                                (GLOBDAT.DM_sidelength-GLOBDAT.gas_sidelength)/2,
                            ),
                            width=GLOBDAT.gas_sidelength, height=GLOBDAT.gas_sidelength,
                            axes = ax_orig,
                            edgecolor = 'red',
                            fill = False,
                            )
                        ax_orig.add_patch(rect)

                        ax_orig.set_xticks([])
                        ax_mdel.set_xticks([])
                        ax_targ.set_xticks([])
                        ax_pred.set_xticks([])
                        ax_orig.set_yticks([])
                        ax_mdel.set_yticks([])
                        ax_targ.set_yticks([])
                        ax_pred.set_yticks([])

                        ax_orig.set_ylabel('Input DM field')
                        ax_mdel.set_ylabel('Model')
                        ax_targ.set_ylabel('Target')
                        ax_pred.set_ylabel('Prediction')

                    plt.suptitle('Output: %s, Scaling: %s'%(ARGS.output, ARGS.scaling), family='monospace')
                    plt.show()
                    #plt.savefig('./Outputs/comparison_target_predicted_%s_pg%d.pdf'%(ARGS.output,t))
                    #plt.cla()
        #}}}

        # SUMMARY STATISTICS
        if True :#{{{
            stepper = Stepper('validation')
            with InputData('validation', stepper) as validation_set :

                a = Analysis(validation_set)

                a.read_original()
                a.compute_powerspectrum('original')
                a.compute_projected_powerspectrum('original')
                a.compute_onepoint('original')
                a.predict_whole_volume()
                a.compute_powerspectrum('predicted')
                a.compute_projected_powerspectrum('predicted')
                a.compute_onepoint('predicted')
                a.compute_correlation_coeff()
                #a.save_predicted_volume()

                if False :
                    plt.loglog(
                        a.original_power_spectrum['k'],
                        a.original_power_spectrum['P'],
                        label = 'original',
                        )
                    plt.loglog(
                        a.predicted_power_spectrum['k'],
                        a.predicted_power_spectrum['P'],
                        label = 'predicted',
                        )
                    plt.xlabel('k')
                    plt.ylabel('P(k)')
                    plt.legend()
                    plt.show()
        #}}}
    #}}}

    # TRAINING
    if ARGS.mode == 'train' :#{{{
        global EPOCH
        EPOCH = 1

        GLOBDAT.net = Network(import_module(ARGS.network).this_network)

        if ARGS.verbose :
            print 'Putting network in parallel mode.'
        GLOBDAT.net = nn.DataParallel(GLOBDAT.net)

#        if ARGS.verbose and not GPU_AVAIL :
        if False :
            print 'Summary of %s'%ARGS.network
            summary(
                GLOBDAT.net,
                [
                    (1, GLOBDAT.DM_sidelength, GLOBDAT.DM_sidelength, GLOBDAT.DM_sidelength),
                    (1, GLOBDAT.gas_sidelength, GLOBDAT.gas_sidelength, GLOBDAT.gas_sidelength)
                ]
                )

        GLOBDAT.net.to(DEVICE)

        loss_function_train = GLOBDAT.loss_function_()
        loss_function_valid = GLOBDAT.loss_function_()
        GLOBDAT.optimizer = GLOBDAT.optimizer_()
        GLOBDAT.lr_scheduler = GLOBDAT.lr_scheduler_()

        if not ARGS.ignoreexisting :
            GLOBDAT.load_network('trained_network_%s.pt'%ARGS.output)

# TODO
#        with InputData('training') as training_set, InputData('validation') as validation_set :
        with InputData('validation') as validation_set :

# TODO
#            training_loader = GLOBDAT.data_loader_(training_set)

            # keep the validation data always the same
            # (reduces noise in the output and makes diagnostics easier)
            validation_set.generate_rnd_indices()
            validation_loader = GLOBDAT.data_loader_(validation_set)

            while True : # train until time is up

                # loop over one epoch
                GLOBDAT.net.train()

                # TODO
                with InputData('training') as training_set :
                    training_loader = GLOBDAT.data_loader_(training_set)
                # END TODO

                    for t, data_train in enumerate(training_loader) :
                        
                        GLOBDAT.optimizer.zero_grad()
                        __pred = GLOBDAT.net(
                            torch.autograd.Variable(data_train[0].to(DEVICE), requires_grad=False),
                            torch.autograd.Variable(data_train[2].to(DEVICE), requires_grad=False)
                            )
                        __targ = torch.autograd.Variable(data_train[1].to(DEVICE), requires_grad=False)
                        loss = loss_function_train(
                            GLOBDAT.target_transformation(__pred),
                            GLOBDAT.target_transformation(__targ)
                            )
                        
                        if ARGS.verbose and (not GPU_AVAIL or ARGS.debug) :
                            print '\ttraining loss : %.3e'%loss.item()

                        GLOBDAT.update_training_loss(loss.item())
                        loss.backward()
                        GLOBDAT.optimizer.step()
                        
                        if GLOBDAT.stop_training() and not ARGS.debug :
                            GLOBDAT.save_loss('loss_%s.npz'%ARGS.output)
                            if ARGS.savebest :
                                GLOBDAT.save_network('trained_network_%s_end.pt'%ARGS.output, False)
                            else :
                                GLOBDAT.save_network('trained_network_%s.pt'%ARGS.output, False)
                            sys.exit(0)
                    # end loop over one epoch

                # evaluate on the validation set
                GLOBDAT.net.eval()
                _loss = 0.0
                for t_val, data_val in enumerate(validation_loader) :
                    with torch.no_grad() :
                        __pred = GLOBDAT.net(
                            torch.autograd.Variable(data_val[0].to(DEVICE), requires_grad=False),
                            torch.autograd.Variable(data_val[2].to(DEVICE), requires_grad=False)
                            )
                        __targ = torch.autograd.Variable(data_val[1].to(DEVICE), requires_grad=False)
                        _loss += loss_function_valid(__pred, __targ).item()
                # end evaluate on validation set

                if ARGS.verbose :
                    print 'validation loss : %.6e'%(_loss/(t_val+1.0))
                GLOBDAT.update_validation_loss(_loss/(t_val+1.0))

                if GLOBDAT.lr_scheduler is not None :
                    if isinstance(GLOBDAT.lr_scheduler, _namesofplaces['ReduceLROnPlateau']) :
                        GLOBDAT.lr_scheduler.step(_loss)
                    elif isinstance(GLOBDAT.lr_scheduler, _namesofplaces['StepLR']) :
                        GLOBDAT.lr_scheduler.step()
                    else :
                        raise NotImplementedError('Unknown learning rate scheduler, do not know how to call.')

                if not ARGS.debug :
                    GLOBDAT.save_loss('loss_%s.npz'%ARGS.output)
                    GLOBDAT.save_network('trained_network_%s.pt'%ARGS.output, ARGS.savebest)
                    if GLOBDAT.breakpoint_reached() :
                        GLOBDAT.save_network('trained_network_%s_%d.pt'%(ARGS.output, EPOCH), False)

                EPOCH += 1
    #}}}
