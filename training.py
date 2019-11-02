import copy
from time import time
from os import system
import sys
import json
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

NETWOR_NR = int(sys.argv[1])
CONFIG_NR = int(sys.argv[2])
SIZE = 1024

START_TIME = time()

# TODO
MAX_TIME  = 350.0 # minutes
GPU_AVAIL = torch.cuda.is_available() # if TRUE, we're on a computing node, otherwise on head node
if GPU_AVAIL :
    print 'Found %d GPUs.'%torch.cuda.device_count()
    torch.set_num_threads(28)
else :
    print 'Could not find GPU.'
    torch.set_num_threads(2)


sys.path.append('/home/lthiele/DM_to_electrons/Networks')

"""
TODO
write code for testing
read from config file
"""

class GlobalData(object) :#{{{
    def __init__(self, configs) :#{{{
        # TODO read in some params file
        if GPU_AVAIL :
            print 'Starting to copy data to /tmp'
            system('cp /tigress/lthiele/boxes/hdf5_files/size_%d.hdf5 /tmp/'%SIZE)
            print 'Finished copying data to /tmp, took %.2e seconds'%(time()-START_TIME) # 4.06e+02 seconds for 2048
            self.data_path = '/tmp/size_%d.hdf5'%SIZE
        else :
            self.data_path = '/tigress/lthiele/boxes/hdf5_files/size_%d.hdf5'%SIZE # presumably move this data to local node /tmp
        self.__output_path = '/home/lthiele/DM_to_electrons/Outputs'

        self.global_dtype = np.float32
        self.block_shapes = {'training':   (SIZE, SIZE           , (1428*SIZE)/2048),
                             'validation': (SIZE,(1368*SIZE)/2048, ( 620*SIZE)/2048),
                             'testing':    (SIZE,( 680*SIZE)/2048, ( 620*SIZE)/2048),
                            }
        self.box_size = 205.0 # Mpc
        self.box_sidelength = SIZE
        self.gas_sidelength = (32*SIZE)/2048
        self.DM_sidelength  = (64*SIZE)/2048
        # these must both be divisible by 2!!!

        self.num_epochs = 10
        self.samples = {
            'training': 8192,
            'validation': 256 if GPU_AVAIL else 32,
            }
        self.global_dtype = np.float32

        # training hyperparameters
        self.__eval_period = 128 # after how many steps the validation loss is evaluated
        self.__loss_function = nn.MSELoss
        self.__optimizer = torch.optim.Adam
        self.__learning_rate = configs["learning_rate"]
        self.__betas = (0.9, 0.999)
        self.__eps = 1e-8
        self.__weight_decay = 1e-4 # L2 regularization
        self.__batch_size = (16*2048)/SIZE if GPU_AVAIL else (4*2048)/SIZE
        self.__num_workers = 1 # don't put this larger than 1 for single hdf5 file! (o/wise input corrupted)

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
            lr = self.__learning_rate,
            betas = self.__betas,
            eps = self.__eps,
            weight_decay = self.__weight_decay
            )
    #}}}
    def data_loader(self, data) :#{{{
        return DataLoader(
            data,
            batch_size = self.__batch_size,
            num_workers = self.__num_workers,
            shuffle = False
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
    def eval(self) :#{{{
        return not len(self.training_loss)%self.__eval_period
    #}}}
    def stop_training(self) :#{{{
        # TODO
        return (time()-START_TIME)/60. > MAX_TIME
    #}}}
    def save_network(self, name) :#{{{
        torch.save(self.net, self.__output_path+name)
    #}}}
    def load_network(self, name) :#{{{
        if GPU_AVAIL :
            self.net = torch.load(self.__output_path+name)
            self.net.cuda()
        else :
            self.net = torch.load(self.__output_path+name, map_location='cpu')
        self.net.eval()
    #}}}
    def clean(self) :#{{{
        if GPU_AVAIL :
            system('rm /tmp/size_%d.hdf5'%SIZE)
        else :
            pass
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

        self.file = h5py.File(self.globdat.data_path, 'r')
        self.DM_dataset = self.file['DM/'+self.mode]
        self.DM_min_log10 = self.file['DM'].attrs['min_log10']
        self.DM_max_log10 = self.file['DM'].attrs['max_log10']
        self.gas_dataset = self.file['gas/'+self.mode]
        self.gas_min_log10 = self.file['gas'].attrs['min_log10']
        self.gas_max_log10 = self.file['gas'].attrs['max_log10']
        
        self.xx_indices_rnd = None
        self.yy_indices_rnd = None
        self.zz_indices_rnd = None
    #}}}
    def generate_rnd_indices(self) :#{{{
        self.xx_indices_rnd = np.random.randint(0, high=self.xlength, size=self.globdat.samples[self.mode])
        self.yy_indices_rnd = np.random.randint(0, high=self.ylength, size=self.globdat.samples[self.mode])
        self.zz_indices_rnd = np.random.randint(0, high=self.zlength, size=self.globdat.samples[self.mode])
    #}}}
    @staticmethod
    def __randomly_transform(arr1, arr2) :#{{{
        # reflections
        if np.random.rand() < 0.5 :
            arr1 = arr1[::-1,:,:]
            arr1 = arr1[::-1,:,:]
        if np.random.rand() < 0.5 :
            arr1 = arr1[:,::-1,:]
            arr2 = arr2[:,::-1,:]
        if np.random.rand() < 0.5 :
            arr1 = arr1[:,::-1]
            arr2 = arr2[:,::-1]
        # transpositions
        prand = np.random.rand()
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
    def getitem_deterministic(self, xx, yy, zz) :#{{{
        assert xx < self.xlength
        assert yy < self.ylength
        assert zz < self.zlength
        DM = self.DM_dataset[
            xx : xx+self.globdat.DM_sidelength,
            yy : yy+self.globdat.DM_sidelength,
            zz : zz+self.globdat.DM_sidelength
            ]
        gas = self.gas_dataset[
            xx+(self.globdat.DM_sidelength-self.globdat.gas_sidelength)/2 : xx+(self.globdat.DM_sidelength+self.globdat.gas_sidelength)/2,
            yy+(self.globdat.DM_sidelength-self.globdat.gas_sidelength)/2 : yy+(self.globdat.DM_sidelength+self.globdat.gas_sidelength)/2,
            zz+(self.globdat.DM_sidelength-self.globdat.gas_sidelength)/2 : zz+(self.globdat.DM_sidelength+self.globdat.gas_sidelength)/2,
            ]

        # normalize input to (-1,1)
        DM *= 2.0
        DM -= 1.0
        return DM, gas
    #}}}
    def __getitem__(self, index) :#{{{
        DM, gas = self.getitem_deterministic(
            self.xx_indices_rnd[index],
            self.yy_indices_rnd[index],
            self.zz_indices_rnd[index]
            )
        DM, gas = InputData.__randomly_transform(DM, gas)
        assert DM.shape[0]==DM.shape[1]==DM.shape[2]==self.globdat.DM_sidelength, DM.shape
        assert gas.shape[0]==gas.shape[1]==gas.shape[2]==self.globdat.gas_sidelength, gas.shape
        return torch.from_numpy(DM.copy()).unsqueeze(0), torch.from_numpy(gas.copy()).unsqueeze(0)
        # add fake dimension
    #}}}
    def __len__(self) :#{{{
        return self.globdat.samples[self.mode]
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
        }
    #}}}
    __names = {#{{{
        'Conv': nn.Conv3d,
        'ConvTranspose': nn.ConvTranspose3d,
        'BatchNorm': nn.BatchNorm3d,
        'ReLU': nn.ReLU,
        }
    #}}}
    @staticmethod
    def __merge(source, destination):#{{{
        # overwrites field in destination if field exists in source, otherwise just merges
        for key, value in source.items():
            if isinstance(value, dict):
                node = destination.setdefault(key, {})
                BasicLayer.__merge(value, node)
            else:
                destination[key] = value
        return destination
    #}}}
    @staticmethod
    def __crop_tensor(x) :#{{{
        x = x.narrow(2,0,x.shape[2]-1).narrow(3,0,x.shape[3]-1).narrow(4,0,x.shape[4]-1).contiguous()
        return x
    #}}}
    def __init__(self, layer_dict) :#{{{
        super(BasicLayer, self).__init__()
        self.__merged_dict = BasicLayer.__merge(
            layer_dict,
            copy.deepcopy(BasicLayer.__default_param)
            )
        self.__crop_output = self.__merged_dict['crop_output'] if 'crop_output' in self.__merged_dict else False

        if self.__merged_dict['conv'] is not None :
            self.__conv_fct = BasicLayer.__names[self.__merged_dict['conv']](
                self.__merged_dict['inplane'], self.__merged_dict['outplane'],
                **self.__merged_dict['conv_kw']
                )
        else :
            self.__conv_fct = Identity()

        if self.__merged_dict['batch_norm'] is not None :
            self.__batch_norm_fct = BasicLayer.__names[self.__merged_dict['batch_norm']](
                self.__merged_dict['outplane'],
                **self.__merged_dict['batch_norm_kw']
                )
        else :
            self.__batch_norm_fct = Identity()
        
        if self.__merged_dict['activation'] is not None :
            self.__activation_fct = BasicLayer.__names[self.__merged_dict['activation']](
                **self.__merged_dict['activation_kw']
                )
        else :
            self.__activation_fct = Identity()
    #}}}
    def forward(self, x) :#{{{
        if self.__crop_output : 
            x = self.__activation_fct(self.__batch_norm_fct(BasicLayer.__crop_tensor(self.__conv_fct(x))))
        else :
            x = self.__activation_fct(self.__batch_norm_fct(self.__conv_fct(x)))
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
                BoxSize=(self.globdat.box_size/self.globdat.global_dtype(self.globdat.box_sidelength))*self.globdat.global_dtype(self.arr.shape)
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


# OUTPUT TESTING
if False :
    globdat = GlobalData()
    globdat.load_network('trained_network.pt')
    validation_set = InputData(globdat, 'validation')
    validation_loader = globdat.data_loader(validation_set)
    if True :

        for t, data in enumerate(validation_loader) :
            with torch.no_grad() :
                target = copy.deepcopy(data[1])
                orig   = copy.deepcopy(data[0])
                pred = globdat.net(
                    torch.autograd.Variable(data[0], requires_grad=False)
                    )
            for ii in xrange(4) :
                fig, ax = plt.subplots(ncols=3)
                ax[0].matshow(orig[ii,0,26,:,:].numpy())
                ax[1].matshow(target[ii,0,10,:,:].numpy())
                ax[2].matshow(pred[ii,0,10,:,:].numpy())
                ax[0].set_title('Original')
                ax[1].set_title('Target')
                ax[2].set_title('Prediction')
                plt.show()

# TRAINING
if True :

    with open('/home/lthiele/DM_to_electrons/Configs/config_%d.json'%CONFIG_NR) as f :
        configs = json.load(f)

    globdat = GlobalData(configs)

    globdat.net = Network(import_module('network_%d'%NETWOR_NR).this_network)
    if GPU_AVAIL :
        globdat.net.cuda()

# TODO
    summary(globdat.net, (1, 32, 32, 32))

    loss_function = globdat.loss_function()
    optimizer = globdat.optimizer()

    training_set   = InputData(globdat, 'training')
    validation_set = InputData(globdat, 'validation')

    while True : # train until time is up

        training_set.generate_rnd_indices()
        validation_set.generate_rnd_indices()
        training_loader   = globdat.data_loader(training_set)
        validation_loader = globdat.data_loader(validation_set)

        for t, data in enumerate(training_loader) :

            globdat.net.train()
            optimizer.zero_grad()
            if GPU_AVAIL :
                loss = loss_function(
                    globdat.net(
                        torch.autograd.Variable(data[0], requires_grad=False).cuda()
                        ),
                    torch.autograd.Variable(data[1], requires_grad=False).cuda()
                    )
            else :
                loss = loss_function(
                    globdat.net(
                        torch.autograd.Variable(data[0], requires_grad=False)
                        ),
                    torch.autograd.Variable(data[1], requires_grad=False)
                    )
            globdat.update_training_loss(loss.item())
            loss.backward()
            optimizer.step()
            
            if globdat.eval() :
                globdat.net.eval()
                _loss = 0.0
                for t_val, data_val in enumerate(validation_loader) :
                    with torch.no_grad() :
                        if GPU_AVAIL :
                            _loss += loss_function(
                                globdat.net(
                                    torch.autograd.Variable(data[0], requires_grad=False).cuda()
                                    ),
                                torch.autograd.Variable(data[1], requires_grad=False).cuda()
                                ).item()
                        else :
                            _loss += loss_function(
                                globdat.net(
                                    torch.autograd.Variable(data[0], requires_grad=False)
                                    ),
                                torch.autograd.Variable(data[1], requires_grad=False)
                                ).item()
                print 'validation loss : %.6e'%(_loss/(t_val+1.0))
                globdat.update_validation_loss(_loss/(t_val+1.0))

            if globdat.stop_training() :
                globdat.save_loss('loss_%d_%d.npz'%(NETWOR_NR,CONFIG_NR))
                globdat.save_network('trained_network_%d_%d.pt'%(NETWOR_NR,CONFIG_NR))
                sys.exit(0)

        globdat.save_loss('loss_%d_%d.npz'%(NETWOR_NR,CONFIG_NR))
        globdat.save_network('trained_network_%d_%d.pt'%(NETWOR_NR,CONFIG_NR))
