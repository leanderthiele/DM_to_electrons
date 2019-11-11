import copy
from time import time, clock
from os import system
import sys
from importlib import import_module
import numpy as np
from nbodykit.lab import ArrayMesh, FFTPower
import h5py
import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchsummary import summary

from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from matplotlib.patches import Rectangle

import argparse

"""
TODO
write code for testing
"""

class _ArgParser(object) :#{{{
    def __init__(self) :#{{{
        _modes    = ['train', 'valid', ]
        _scalings = ['log', 'linear', 'sqrt', 'cbrt', ]

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

        # consistency checks
        assert self.__args.mode    in _modes,    'Only %s modes implemented so far, passed %s.'%(_modes, self.__args.mode)
        assert self.__args.scaling in _scalings, 'Only %s scalings implemented so far, passed %s.'%(_scalings, self.__args.scaling)
    #}}}
    def __getattr__(self, name) :#{{{
        return self.__args.__getattribute__(name)
    #}}}
#}}}

_names = {#{{{
    'Conv': nn.Conv3d,
    'ConvTranspose': nn.ConvTranspose3d,
    'BatchNorm': nn.BatchNorm3d,
    'ReLU': nn.ReLU,
    'MSELoss': nn.MSELoss,
    'L1Loss': nn.L1Loss,
    'Adam': torch.optim.Adam,
    'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
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

class GlobalData(object) :#{{{
    def __init__(self, *configs) :#{{{
        configs = list(configs)
        assert len(configs) > 0, 'You need to pass at least one config file.'

        # merge config files
        for ii in xrange(1, len(configs)) :
            configs[0] = _merge(configs[ii], configs[0])

        if ARGS.verbose :
            print 'Using the following configuration :'
            print configs[0]

        # which box to work on
        self.box_sidelength = configs[0]['box_sidelength']
        self.gas_sidelength = (configs[0]['gas_sidelength']*self.box_sidelength)/2048
        self.DM_sidelength  = (configs[0]['DM_sidelength']*self.box_sidelength)/2048
        assert not self.gas_sidelength%2, 'Only even gas_sidelength is supported.'
        assert not self.DM_sidelength%2,  'Only even DM_sidelength is supported.'

        # training hyperparameters
        self.__loss_function    = _names[configs[0]['loss_function']]
        self.__loss_function_kw = configs[0]['loss_function_%s_kw'%configs[0]['loss_function']]

        self.__optimizer        = _names[configs[0]['optimizer']]
        self.__optimizer_kw     = configs[0]['optimizer_%s_kw'%configs[0]['optimizer']]

        self.Nsamples           = configs[0]['Nsamples']
        self.__train_time       = configs[0]['train_time'] # minutes

        self.__data_loader_kw   = configs[0]['data_loader_kw']
        self.num_workers = self.__data_loader_kw['num_workers'] if 'num_workers' in self.__data_loader_kw else 1

        self.__lr_scheduler     = _names[configs[0]['lr_scheduler']]
        if self.__lr_scheduler is not None :
            self.__lr_scheduler_kw = configs[0]['lr_scheduler_%s_kw'%configs[0]['lr_scheduler']]

        # where to find and put data files
        self.__input_path  = configs[0]['input_path']
        self.__output_path = configs[0]['output_path']

        if GPU_AVAIL :
            print 'Starting to copy data to /tmp'
            system('cp %s%s/size_%d.hdf5 /tmp/'%(self.__input_path, ARGS.scaling, self.box_sidelength))
            print 'Finished copying data to /tmp, took %.2e seconds'%(time()-START_TIME) # 4.06e+02 sec for 2048, ~12 sec for 1024
            self.data_path = '/tmp/size_%d.hdf5'%self.box_sidelength
        else :
            self.data_path = '%s%s/size_%d.hdf5'%(self.__input_path, ARGS.scaling, self.box_sidelength)

        # Some hardcoded values
        self.block_shapes = {'training':   (self.box_sidelength, self.box_sidelength           , (1428*self.box_sidelength)/2048),
                             'validation': (self.box_sidelength,(1368*self.box_sidelength)/2048, ( 620*self.box_sidelength)/2048),
                             'testing':    (self.box_sidelength,( 680*self.box_sidelength)/2048, ( 620*self.box_sidelength)/2048),
                            }
        self.box_size = 205.0 # Mpc

        # keep track of training
        self.training_steps = []
        self.validation_steps = []
        self.training_loss = []
        self.validation_loss = []

        # needs to be updated from the main code
        self.net = None
    #}}}
    def loss_function(self) :#{{{
        return self.__loss_function()
    #}}}
    def optimizer(self) :#{{{
        return self.__optimizer(
            params = self.net.parameters(),
            **self.__optimizer_kw
            )
    #}}}
    def lr_scheduler(self, optimizer) :#{{{
        if self.__lr_scheduler is not None :
            return self.__lr_scheduler(
                optimizer,
                **self.__lr_scheduler_kw
                )
        else :
            return None
    #}}}
    def data_loader(self, data) :#{{{
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
        self.validation_steps.append(len(self.training_loss))
    #}}}
    def save_loss(self, name) :#{{{
        np.savez(
            self.__output_path+name,
            training_steps = self.training_steps,
            training_loss = self.training_loss,
            validation_steps = self.validation_steps,
            validation_loss = self.validation_loss,
            )
    #}}}
    def stop_training(self) :#{{{
        return (time()-START_TIME)/60. > self.__train_time
    #}}}
    def save_network(self, name) :#{{{
        torch.save(self.net.state_dict(), self.__output_path+name)
    #}}}
    def load_network(self, name) :#{{{
        # need to initialize network first
        self.net.load_state_dict(torch.load(self.__output_path+name, map_location='cuda:0' if GPU_AVAIL else 'cpu'))
        self.net.to(DEVICE)
        self.net.eval()
    #}}}
#}}}

class InputData(Dataset) :#{{{
    def __init__(self, globdat, mode) :#{{{
        """
        mode = 'training' , 'validation' , 'testing'
        """
        self.globdat = globdat
        self.mode = mode
        self.xlength = self.globdat.block_shapes[mode][0] - self.globdat.DM_sidelength
        self.ylength = self.globdat.block_shapes[mode][1] - self.globdat.DM_sidelength
        self.zlength = self.globdat.block_shapes[mode][2] - self.globdat.DM_sidelength

        self.files = []
        self.DM_datasets = []
        self.gas_datasets = []
        self.rnd_generators = []
        for ii in xrange(self.globdat.num_workers) :
            self.files.append(h5py.File(self.globdat.data_path, 'r'))
            self.DM_datasets.append(self.files[-1]['DM/'+self.mode])
            self.gas_datasets.append(self.files[-1]['gas/'+self.mode])
            self.rnd_generators.append(np.random.RandomState(int(1e6*clock())+ii))

        # need these if we want to transform back
        self.DM_training_mean    = self.file['DM'].attrs['training_mean']
        self.DM_training_stddev  = self.file['DM'].attrs['training_stddev']
        self.gas_training_mean   = self.file['gas'].attrs['training_mean']
        self.gas_training_stddev = self.file['gas'].attrs['training_stddev']
        
        self.xx_indices_rnd = None
        self.yy_indices_rnd = None
        self.zz_indices_rnd = None
    #}}}
    def generate_rnd_indices(self) :#{{{
        self.xx_indices_rnd = np.random.randint(0, high=self.xlength, size=self.globdat.Nsamples[self.mode])
        self.yy_indices_rnd = np.random.randint(0, high=self.ylength, size=self.globdat.Nsamples[self.mode])
        self.zz_indices_rnd = np.random.randint(0, high=self.zlength, size=self.globdat.Nsamples[self.mode])
    #}}}
    def __generate_rnd_index(self,  __this_ID) :
        xx_rnd = self.rnd_generators[__this_ID].randint(0, high=self.xlength)
        yy_rnd = self.rnd_generators[__this_ID].randint(0, high=self.ylength)
        zz_rnd = self.rnd_generators[__this_ID].randint(0, high=self.zlength)
        return xx_rnd, yy_rnd, zz_rnd
    def __randomly_transform(self, arr1, arr2, __this_ID) :#{{{
        # reflections
        if self.rnd_generators[__this_ID].random.rand() < 0.5 :
            arr1 = arr1[::-1,:,:]
            arr1 = arr1[::-1,:,:]
        if self.rnd_generators[__this_ID].rand() < 0.5 :
            arr1 = arr1[:,::-1,:]
            arr2 = arr2[:,::-1,:]
        if self.rnd_generators[__this_ID].rand() < 0.5 :
            arr1 = arr1[:,::-1]
            arr2 = arr2[:,::-1]
        # transpositions
        prand = self.rnd_generators[__this_ID].rand()
        if prand < 1./6. :
            arr1 = np.transpose(arr1, axes=(0,2,1))
            arr2 = np.transpose(arr2, axes=(0,2,1))
        elif prand < 2./6. :
            arr1 = np.transpose(arr1, axes=(2,1,0))
            arr2 = np.transpose(arr2, axes=(2,1,0))
        elif prand < 3./6. :
            arr1 = np.transpose(arr1, axes=(1,0,2))
            arr2 = np.transpose(arr2, axes=(1,0,2))
        elif prand < 4./6. :
            arr1 = np.transpose(arr1, axes=(2,0,1))
            arr2 = np.transpose(arr2, axes=(2,0,1))
        elif prand < 5./6. :
            arr1 = np.transpose(arr1, axes=(1,2,0))
            arr2 = np.transpose(arr2, axes=(1,2,0))
        return arr1, arr2
    #}}}
    def __index_3D(self, index) :#{{{
        return (
            index%self.xlength,
            (index/self.xlength)%self.ylength,
            index/(self.ylength*self.xlength)
            )
    #}}}
    def getitem_deterministic(self, xx, yy, zz, __this_ID) :#{{{
        assert xx < self.xlength
        assert yy < self.ylength
        assert zz < self.zlength
        DM = self.DM_datasets[__this_ID][
            xx : xx+self.globdat.DM_sidelength,
            yy : yy+self.globdat.DM_sidelength,
            zz : zz+self.globdat.DM_sidelength
            ]
        gas = self.gas_datasets[__this_ID][
            xx+(self.globdat.DM_sidelength-self.globdat.gas_sidelength)/2 : xx+(self.globdat.DM_sidelength+self.globdat.gas_sidelength)/2,
            yy+(self.globdat.DM_sidelength-self.globdat.gas_sidelength)/2 : yy+(self.globdat.DM_sidelength+self.globdat.gas_sidelength)/2,
            zz+(self.globdat.DM_sidelength-self.globdat.gas_sidelength)/2 : zz+(self.globdat.DM_sidelength+self.globdat.gas_sidelength)/2,
            ]
        return DM, gas
    #}}}
    def __getitem__(self, index) :#{{{
        __this_ID = torch.utils.data.get_worker_info().id
        if self.xx_indices_rnd is None or self.yy_indices_rnd is None or self.zz_indices_rnd is None :
            DM, gas = self.getitem_deterministic(
                *self.__generate_rnd_index(__this_ID), __this_ID = __this_ID
                )
        else : # useful to keep validation set manageable (reduce noise in validation loss)
            DM, gas = self.getitem_deterministic(
                self.xx_indices_rnd[index],
                self.yy_indices_rnd[index],
                self.zz_indices_rnd[index],
                __this_ID
                )
        DM, gas = InputData.__randomly_transform(DM, gas, __this_ID)
        assert DM.shape[0]==DM.shape[1]==DM.shape[2]==self.globdat.DM_sidelength, DM.shape
        assert gas.shape[0]==gas.shape[1]==gas.shape[2]==self.globdat.gas_sidelength, gas.shape
        return torch.from_numpy(DM.copy()).unsqueeze(0), torch.from_numpy(gas.copy()).unsqueeze(0)
        # add fake dimension
    #}}}
    def __len__(self) :#{{{
        return self.globdat.Nsamples[self.mode]
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
        }
    #}}}
    @staticmethod
    def __crop_tensor(x) :#{{{
        x = x.narrow(2,0,x.shape[2]-1).narrow(3,0,x.shape[3]-1).narrow(4,0,x.shape[4]-1).contiguous()
        return x
    #}}}
    def __init__(self, layer_dict) :#{{{
        super(BasicLayer, self).__init__()
        self.__merged_dict = _merge(
            layer_dict,
            copy.deepcopy(BasicLayer.__default_param)
            )

        if self.__merged_dict['conv'] is not None :
            self.__conv_fct = _names[self.__merged_dict['conv']](
                self.__merged_dict['inplane'], self.__merged_dict['outplane'],
                **self.__merged_dict['conv_kw']
                )
        else :
            self.__conv_fct = Identity()

        if self.__merged_dict['crop_output'] :
            self.__crop_fct = BasicLayer.__crop_tensor
        else :
            self.__crop_fct = Identity()

        if self.__merged_dict['batch_norm'] is not None :
            self.__batch_norm_fct = _names[self.__merged_dict['batch_norm']](
                self.__merged_dict['outplane'],
                **self.__merged_dict['batch_norm_kw']
                )
        else :
            self.__batch_norm_fct = Identity()
        
        if self.__merged_dict['activation'] is not None :
            self.__activation_fct = _names[self.__merged_dict['activation']](
                **self.__merged_dict['activation_kw']
                )
        else :
            self.__activation_fct = Identity()
    #}}}
    def forward(self, x) :#{{{
        x = self.__activation_fct(self.__batch_norm_fct(self.__crop_fct(self.__conv_fct(x))))
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
    #}}}
    @staticmethod
    def __feed_forward_block(input_list) :#{{{
        layers = []
        for layer_dict in input_list :
            layers.append(BasicLayer(layer_dict))
        return nn.Sequential(*layers)
    #}}}
    def forward(self, x) :#{{{
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
                x = torch.cat((x, intermediate_data[ii]), dim = 1)
            x = self.__blocks[2*ii+1](x)
        return x
    #}}}
#}}}

class Mesh(object) :#{{{
    def __init__(self, globdat, arr) :#{{{
        self.globdat = globdat
        self.arr = arr
        self.mesh = None
    def read_mesh(self) :
        if self.mesh is not None :
            print 'Already read mesh, No action taken.'
        if self.arr is None :
            raise RuntimeError('Not set the array yet for Mesh.')
        else :
            self.mesh = ArrayMesh(
                self.arr,
                BoxSize=(self.globdat.box_size/float(self.globdat.box_sidelength))*float(self.arr.shape)
                )
    #}}}
    def compute_powerspectrum(self) :#{{{
        if self.mesh is None :
            raise RuntimeError('You need to read the mesh to memory first.')
        return FFTPower(self.mesh, mode='1d')
    #}}}
#}}}

class Analysis(object) :#{{{
    def __init__(self, globdat, data) :#{{{
        # data is instance of InputData
        self.globdat = globdat
        self.data = data

        self.predicted_field = None

        # TODO rescale to original units
        #self.original_field_mesh = Mesh(
        #    self.globdat,
        #    self.data.gas_dataset[
        #        (self.globdat.DM_sidelength-self.globdat.gas_sidelength)/2 : self.globdat.xlength+(self.globdat.DM_sidelength-self.globdat.gas_sidelength)/2,
        #        (self.globdat.DM_sidelength-self.globdat.gas_sidelength)/2 : self.globdat.ylength+(self.globdat.DM_sidelength-self.globdat.gas_sidelength)/2,
        #        (self.globdat.DM_sidelength-self.globdat.gas_sidelength)/2 : self.globdat.zlength+(self.globdat.DM_sidelength-self.globdat.gas_sidelength)/2,
        #        ]
        #    )
        #self.predicted_field_mesh = Mesh(self.globdat, self.predicted_field)

        self.original_power_spectrum = None
        self.predicted_power_spectrum = None
    #}}}
    def predict_single_cube(self, xx, yy, zz) :#{{{
        DM, gas_original = self.data.getitem_deterministic(xx, yy, zz)
        if GPU_AVAIL :
            gas_pred = self.globdat.net(
                torch.autograd.Variable(torch.from_numpy(DM.copy()).unsqueeze(0), requires_grad=False).cuda()
                )
        else :
            gas_pred = self.globdat.net(
                torch.autograd.Variable(torch.from_numpy(DM.copy()).unsqueeze(0), requires_grad=False)
                )
        return gas_original, gas_pred.data.cpu().numpy()
    #}}}
    def predict_whole_volume(self) :#{{{
        self.predicted_field = np.empty(
            (self.globdat.xlength, self.globdat.ylength, self.globdat.zlength),
            dtype = np.float32
            )
        xx, yy, zz = 0, 0, 0
        break_xx, break_yy, break_zz = False, False, False
        while True :
            while True :
                while True :
                    _, gas_pred = self.predict_single_cube(xx, yy, zz)
                    self.predicted_field[
                        xx : xx + self.globdat.gas_sidelength,
                        yy : yy + self.globdat.gas_sidelength,
                        zz : zz + self.globdat.gas_sidelength
                        ] = gas_pred
                    if zz < self.data.zlength - self.globdat.gas_sidelength :
                        zz += self.globdat.gas_sidelength
                    elif break_zz :
                        break
                    else :
                        zz = self.data.zlength - self.globdat.gas_sidelength
                        break_zz = True
                if yy < self.data.ylength - self.globdat.gas_sidelength :
                    yy += self.globdat.gas_sidelength
                elif break_yy :
                    break;
                else :
                    yy = self.data.ylength - self.globdat.gas_sidelength
                    break_yy = True
            if xx < self.data.xlength - self.globdat.gas_sidelength :
                xx += self.globdat.gas_sidelength
            elif break_xx : 
                break
            else :
                xx = self.data.xlength - self.globdat.gas_sidelength
                break_xx = True
    #}}}
    def compute_powerspectrum(self, mode) :#{{{
        if mode is 'original' :
            if self.original_power_spectrum is not None :
                print 'Already computed this original power spectrum. No action taken.'
            else :
                self.original_power_spectrum = self.original_field_mesh.compute_powerspectrum()
        elif mode is 'predicted' :
            if self.predicted_power_spectrum is not None :
                print 'Already computed this predicted power spectrum. No action taken.'
            else :
                self.predicted_power_spectrum = self.predicted_field_mesh.compute_powerspectrum()
        else :
            raise RuntimeError('Invalid mode in compute_powerspectrum, only original and predicted allowed.')
    #}}}
#}}}

if __name__ == '__main__' :
    # set global variables#{{{
    global ARGS
    global START_TIME
    global CONFIGS
    global GPU_AVAIL
    global DEVICE

    START_TIME = time()
    ARGS = _ArgParser()
    sys.path.append(ARGS.network_path)
    sys.path.append(ARGS.config_path)
    CONFIGS = []
    for c in ARGS.config :
        CONFIGS.append(import_module(c).this_config)
    if ARGS.verbose :
        for ii in xrange(len(ARGS.config)) :
            print '%s :'%ARGS.config[ii]
            print CONFIGS[ii]
    GPU_AVAIL = torch.cuda.is_available()
    if GPU_AVAIL :
        print 'Found %d GPUs.'%torch.cuda.device_count()
        torch.set_num_threads(4)
        DEVICE = torch.device('cuda:0')
    else :
        print 'Could not find GPU.'
        torch.set_num_threads(2)
        DEVICE = torch.device('cpu')
    #}}}

    # OUTPUT VALIDATION
    if ARGS.mode=='valid' :#{{{

        globdat = GlobalData(*CONFIGS)

        globdat.net = nn.DataParallel(Network(import_module(ARGS.network).this_network))
        globdat.load_network('trained_network_%s.pt'%ARGS.output)
            
        validation_set = InputData(globdat, 'validation')
        validation_set.generate_rnd_indices()
        validation_loader = globdat.data_loader(validation_set)

        NPAGES = 4
        for t, data in enumerate(validation_loader) :
            if t == NPAGES : break

            with torch.no_grad() :
                targ = copy.deepcopy(data[1])
                orig = copy.deepcopy(data[0])
                pred = globdat.net(
                    torch.autograd.Variable(data[0], requires_grad=False)
                    )
            
            NPlots = 8
            fig = plt.figure(figsize=(8.5, 11.0))
            gs0 = gs.GridSpec(NPlots/2, 2, figure=fig, wspace = 0.2)
            for ii in xrange(NPlots) :
                plane_gas = np.random.randint(0, high=globdat.gas_sidelength)
                plane_DM  = plane_gas + (globdat.DM_sidelength-globdat.gas_sidelength)/2

                gs00 = gs.GridSpecFromSubplotSpec(
                    2, 3, subplot_spec=gs0[ii%(NPlots/2), ii/(NPlots/2)],
                    wspace = 0.1
                    )
                ax_orig = plt.subplot(gs00[:,:2])
                ax_targ = plt.subplot(gs00[:1,-1])
                ax_pred = plt.subplot(gs00[1:,-1])
                fig.add_subplot(ax_orig)
                fig.add_subplot(ax_targ)
                fig.add_subplot(ax_pred)

#                if ARGS.scaling=='log' :
                if True :
                    vminmax = 3 # standard deviations
                    ax_orig.matshow(
                        orig[ii,0,plane_DM,:,:].numpy(),
                        extent=(0, globdat.DM_sidelength, 0, globdat.DM_sidelength),
#                        vmin = -vminmax, vmax = vminmax,
                        )
                    ax_targ.matshow(
                        targ[ii,0,plane_gas,:,:].numpy(),
#                        vmin = -vminmax, vmax = vminmax,
                        )
                    ax_pred.matshow(
                        pred[ii,0,plane_gas,:,:].numpy(),
#                        vmin = -vminmax, vmax = vminmax,
                        )
#                elif ARGS.scaling=='linear' :
#                    ax_orig.matshow(
#                        np.log(np.sum(np.exp(orig[ii,0,:,:,:].numpy()), axis=0)),
#                        extent=(0, globdat.DM_sidelength, 0, globdat.DM_sidelength),
#                        )
#                    ax_targ.matshow(
#                        np.log(np.sum(targ[ii,0,:,:,:].numpy(), axis=0)),
#                        )
#                    ax_pred.matshow(
#                        np.log(np.sum(pred[ii,0,:,:,:].numpy(), axis=0)),
#                        )

                rect = Rectangle(
                    (
                        (globdat.DM_sidelength-globdat.gas_sidelength)/2,
                        (globdat.DM_sidelength-globdat.gas_sidelength)/2,
                    ),
                    width=globdat.gas_sidelength, height=globdat.gas_sidelength,
                    axes = ax_orig,
                    edgecolor = 'red',
                    fill = False,
                    )
                ax_orig.add_patch(rect)

                ax_orig.set_xticks([])
                ax_targ.set_xticks([])
                ax_pred.set_xticks([])
                ax_orig.set_yticks([])
                ax_targ.set_yticks([])
                ax_pred.set_yticks([])

                ax_orig.set_ylabel('Input DM field')
                ax_targ.set_ylabel('Target')
                ax_pred.set_ylabel('Prediction')

            plt.suptitle('Output: %s, Scaling: %s'%(ARGS.output, ARGS.scaling), family='monospace')
            plt.show()
            #plt.savefig('./Outputs/comparison_target_predicted_%s_pg%d.pdf'%(ARGS.output,t))
            #plt.cla()
    #}}}

    # TRAINING
    if ARGS.mode=='train' :#{{{

        globdat = GlobalData(*CONFIGS)

        globdat.net = Network(import_module(ARGS.network).this_network)
        if ARGS.verbose :
            print 'Putting network in parallel mode.'
        globdat.net = nn.DataParallel(globdat.net)
        globdat.net.to(DEVICE)
        if ARGS.verbose :
            print 'Summary of %s'%ARGS.network
            summary(globdat.net, (1, globdat.DM_sidelength, globdat.DM_sidelength, globdat.DM_sidelength))

        loss_function = globdat.loss_function()
        optimizer = globdat.optimizer()
        lr_scheduler = globdat.lr_scheduler(optimizer)

        training_set   = InputData(globdat, 'training')
        validation_set = InputData(globdat, 'validation')

        # keep the validation data always the same
        # (reduces noise in the output and makes diagnostics easier)
        validation_set.generate_rnd_indices()
        validation_loader = globdat.data_loader(validation_set)

        while True : # train until time is up

            training_loader = globdat.data_loader(training_set)

            for t, data in enumerate(training_loader) :

                globdat.net.train()
                optimizer.zero_grad()
                loss = loss_function(
                    globdat.net(
                        torch.autograd.Variable(data[0].to(DEVICE), requires_grad=False)
                        ),
                    torch.autograd.Variable(data[1].to(DEVICE), requires_grad=False)
                    )
                
                if ARGS.verbose and not GPU_AVAIL :
                    print '\ttraining loss : %.3e'%loss.item()

                globdat.update_training_loss(loss.item())
                loss.backward()
                optimizer.step()
                
                if globdat.stop_training() :
                    globdat.save_loss('loss_%s.npz'%ARGS.output)
                    globdat.save_network('trained_network_%s.pt'%ARGS.output)
                    sys.exit(0)

            globdat.net.eval()
            _loss = 0.0
            for t_val, data_val in enumerate(validation_loader) :
                with torch.no_grad() :
                    _loss += loss_function(
                        globdat.net(
                            torch.autograd.Variable(data[0], requires_grad=False).to(DEVICE)
                            ),
                        torch.autograd.Variable(data[1], requires_grad=False).to(DEVICE)
                        ).item()

            if ARGS.verbose :
                print 'validation loss : %.6e'%(_loss/(t_val+1.0))
            globdat.update_validation_loss(_loss/(t_val+1.0))

            if lr_scheduler is not None :
                lr_scheduler.step(_loss)

            globdat.save_loss('loss_%s.npz'%ARGS.output)
            globdat.save_network('trained_network_%s.pt'%ARGS.output)
    #}}}
