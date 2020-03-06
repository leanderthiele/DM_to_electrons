# general utilities
import copy
import datetime
from time import time, clock
from os import system
from os.path import isfile
import sys
from importlib import import_module
import argparse
import traceback

# numerical libraries
import numpy as np
from nbodykit.lab import ArrayMesh, FFTPower
from nbodykit.algorithms.fftpower import ProjectedFFTPower

import Pk_library as PKL

class _ArgParser(object) :#{{{
    def __init__(self) :#{{{
        self.__parser = argparse.ArgumentParser(
            description='Code to train on DM only sims and hydro sims to \
                         learn mapping between DM and electron pressure.'
            )

        self.__parser.add_argument(
            '-psfid', '--powerspectrumfid',
            nargs = '?',
            default = 'psfid',
            help = 'File to store fiducial powerspectrum.'
            )
        self.__parser.add_argument(
            '-psmdel', '--powerspectrummdel',
            nargs = '?',
            default = 'psmdel',
            help = 'File to store model powerspectrum.'
            )
        self.__parser.add_argument(
            '-pspred', '--powerspectrumpred',
            nargs = '?',
            default = 'pspred',
            help = 'File to store predicted powerspectrum.'
            )
        self.__parser.add_argument(
            '-bkfid', '--bispectrumfid',
            nargs = '?',
            default = 'bkfid',
            help = 'File to store fiducial bispectrum.'
            )
        self.__parser.add_argument(
            '-bkmdel', '--bispectrummdel',
            nargs = '?',
            default = 'bkmdel',
            help = 'File to store model bispectrum.'
            )
        self.__parser.add_argument(
            '-bkpred', '--bispectrumpred',
            nargs = '?',
            default = 'bkpred',
            help = 'File to store predicted bispectrum.'
            )
        self.__parser.add_argument(
            '-projpsfid', '--projpowerspectrumfid',
            nargs = '?',
            default = 'projpsfid',
            help = 'File to store fiducial projected powerspectrum.'
            )
        self.__parser.add_argument(
            '-projpsmdel', '--projpowerspectrummdel',
            nargs = '?',
            default = 'projpsmdel',
            help = 'File to store model projected powerspectrum.'
            )
        self.__parser.add_argument(
            '-projpspred', '--projpowerspectrumpred',
            nargs = '?',
            default = 'projpspred',
            help = 'File to store predicted projected powerspectrum.'
            )
        self.__parser.add_argument(
            '-crossmdel', '--crosspowermdel',
            nargs = '?',
            default = 'crossmdel',
            help = 'File to store correlation coefficient r(k) for the model (wrt ground truth)'
            )
        self.__parser.add_argument(
            '-crosspred', '--crosspowerpred',
            nargs = '?',
            default = 'crosspred',
            help = 'File to store correlation coefficient r(k) for the prediction (wrt ground truth)'
            )
        self.__parser.add_argument(
            '-opfid', '--onepointfid',
            nargs = '?',
            default = 'opfid',
            help = 'File to store fiducial one-point PDF.'
            )
        self.__parser.add_argument(
            '-opmdel', '--onepointmdel',
            nargs = '?',
            default = 'opmdel',
            help = 'File to store model one-point PDF.'
            )
        self.__parser.add_argument(
            '-oppred', '--onepointpred',
            nargs = '?',
            default = 'oppred',
            help = 'File to store predicted one-point PDF.'
            )
        self.__parser.add_argument(
            '-v', '--verbose',
            action = 'store_true',
            help = 'If you want some diagnostic outputs.',
            )
        self.__parser.add_argument(
            '-thr', '--threads',
            nargs = '?',
            type = int,
            default = 1,
            help = 'Number of OpenMP threads for power/bi-spectrum calculations.'
            )
        
        # parse now
        self.__args = self.__parser.parse_args()
#}}}
    def __getattr__(self, name) :#{{{
        return self.__args.__getattribute__(name)
    #}}}
#}}}

class Field(object) :#{{{
    def __init__(self, mode, source) :#{{{
        _modes = ['fiducial', 'model', 'predicted', ]
        self.box_sidelength = 1008
        self.summary_path = '/scratch/gpfs/lthiele/test_summaries/'
        self.mode = mode
        self.source = source
        assert self.mode in _modes

        self.data = np.fromfile(source, dtype=np.float32).reshape(self.box_sidelength,self.box_sidelength,self.box_sidelength)
        if ARGS.verbose :
            print 'Loaded data in Field(%s)'%self.mode

        self.BoxSize = 205.0*2016.0/2048.0
        self.MAS = None # mass assignment scheme for pylians

        self.onepointname = ARGS.onepointfid if self.mode == 'fiducial'        \
                            else ARGS.onepointmdel if self.mode == 'model'     \
                            else ARGS.onepointpred if self.mode == 'predicted' \
                            else None
        self.onepoint = None
        self.powerspectrumname = ARGS.powerspectrumfid if self.mode == 'fiducial'        \
                                 else ARGS.powerspectrummdel if self.mode == 'model'     \
                                 else ARGS.powerspectrumpred if self.mode == 'predicted' \
                                 else None
        self.powerspectrum = None
        self.crosspowername = ARGS.crosspowermdel if self.mode == 'model'          \
                              else ARGS.crosspowerpred if self.mode == 'predicted' \
                              else None
        self.crosspower = None
        self.bispectrumname = ARGS.bispectrumfid if self.mode == 'fiducial'        \
                              else ARGS.bispectrummdel if self.mode == 'model'     \
                              else ARGS.bispectrumpred if self.mode == 'predicted' \
                              else None
        self.bispectrum = None

        concat_names = lambda name : '%s_%s_%d'%(name, ARGS.output, self.box_sidelength) \
                                     if self.mode=='predicted'                              \
                                     else '%s_%d'%(name, self.box_sidelength)
        self.onepointname = concat_names(self.onepointname)
        self.powerspectrumname = concat_names(self.powerspectrumname)
        self.crosspowername = concat_names(self.crosspowername)
        self.bispectrumname = concat_names(self.bispectrumname)
    #}}}
    def compute_onepoint(self, save=True) :#{{{
        h, e = np.histogram(self.data, bins=np.linspace(1e-2, 1e1, num=101), density=False)
        h = h.astype(float)/float(self.data.size)
        self.onepoint = {'h': h, 'edges': e, }
        if save :
            np.savez(self.summary_path+'%s.npz'%self.onepointname,
                     **self.onepoint)

        if ARGS.verbose :
            print 'Computed onepoint in Field(%s)'%self.mode
    #}}}
    def __compute_powerspectrum(self) :#{{{
        Pk = PKL.Pk(self.data, self.BoxSize, 0, self.MAS, ARGS.threads)
        self.powerspectrum = {'k': Pk.k1D, 'P': Pk1D, }
    #}}}
    def _save_powerspectrum(self) :#{{{
        np.savez(self.summary_path+'%s.npz'%self.powerspectrumname,
                 **self.powerspectrum)
    #}}}
    def compute_powerspectrum(self, other, save=True) :#{{{
        if other is None :
            self.__compute_powerspectrum(save)
            if save :
                self._save_powerspectrum()
        else :
            assert isinstance(other, Field)
            assert np.allclose(self.BoxSize, other.BoxSize)
            Pk = PKL.XPk([self.data, other.data], self.BoxSize, 0, [self.MAS, other.MAS], ARGS.threads)
            self.powerspectrum = {'k': Pk.k1D, 'P': Pk.Pk1D[:,0], }
            other.powerspectrum = {'k': Pk.k1D, 'P': Pk.Pk1D[:,1], }
            self.crosspower = {'k': Pk.k1D, 'r': Pk.PkX1D[:,0]/np.sqrt(Pk.Pk1D[:,0]*Pk.Pk1D[:,1]), }
            if save :
                self._save_powerspectrum()
                other._save_powerspectrum()
                np.savez(self.summary_path+'%s.npz'%self.crosspowername,
                         **self.crosspower)

        if ARGS.verbose :
            print 'Computed powerspectrum in Field(%s)'%self.mode
    #}}}
    def compute_bispectrum(self, save=True) :#{{{
        # (1) triangle configurations k = 1, 3, varying angles
        _k1 = 1.0
        _k2 = 3.0
        _theta = np.linspace(0.0, np.pi, num=10)
        Bk1 = PKL.Bk(self.data, self.BoxSize, _k1, _k2, _theta, self.MAS, ARGS.threads).Q
        if ARGS.verbose :
            print 'Computed Bk1 in Field(%s)'%self.mode

        # (2) triangle configurations k = 0.1, 0.3, varying angles
        _k1 = 0.2
        _k2 = 0.3
        _theta = np.linspace(0.0, np.pi, num=10)
        Bk2 = PKL.Bk(self.data, self.BoxSize, _k1, _k2, _theta, self.MAS, ARGS.threads).Q
        if ARGS.verbose :
            print 'Computed Bk2 in Field(%s)'%self.mode

        # (3) equilateral triangle configurations
        _k = 10.0**np.linspace(np.log10(0.2), np.log10(3.0), num=10)
        _theta = np.array([np.pi/3.0])
        Bk3 = []
        for k in _k :
            Bk3.append(PKL.Bk(self.data, self.BoxSize, k, k, _theta, self.MAS, ARGS.threads).Q)
        if ARGS.verbose :
            print 'Computed Bk3 in Field(%s)'%self.mode

        self.bispectrum = {'Bk1': Bk1, 'Bk2': Bk2, 'Bk3': Bk3, }
        if save :
            np.savez(self.summary_path+'%s.npz'%self.bispectrumname,
                     **self.bispectrum)

        if ARGS.verbose :
            print 'Computed bispectrum in Field(%s)'%self.mode
    #}}}
#}}}

if __name__ == '__main__' :
    global ARGS
    ARGS = _ArgParser()

    fmdel = Field('model', '/scratch/gpfs/lthiele/boxes/testbox_gas_model_1024.bin')
    ffid  = Field('fiducial', '/scratch/gpfs/lthiele/boxes/testbox_gas_1024.bin')

#    try :
#        ffid.compute_onepoint()
#        fmdel.compute_onepoint()
#    except Exception as e :
#        print 'Failed to compute onepoint.'
#        print traceback.print_exc()
#        print e.__doc__
#        print e.message

    try :
        fmdel.compute_powerspectrum(ffid)
    except Exception as e :
        print 'Failed to compute powerspectrum.'
        print traceback.print_exc()
        print e.__doc__
        print e.message

    try :
        ffid.compute_bispectrum()
    except Exception as e :
        print 'Failed to compute fiducial bispectrum.'
        print traceback.print_exc()
        print e.__doc__
        print e.message

    try :
        fmdel.compute_bispectrum()
    except Exception as e :
        print 'Failed to compute model bispectrum.'
        print traceback.print_exc()
        print e.__doc__
        print e.message
