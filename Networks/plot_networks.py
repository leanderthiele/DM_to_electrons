import numpy as np
import collections
import copy
from importlib import import_module

DIM_IN = 64 # current DM box size
DIM_OUT = 32 # current gas box size

# First determine all the plane coordinates
HORI = 0.75 # units
VERT = 1.0 # units

# Pretty Layers
T = 0.1*HORI
A = 0.1*abs(VERT)
B = 0.1*HORI


class Point(object) :#{{{
    def __init__(self, xx, yy) :
        self.xx = xx
        self.yy = yy
#}}}

class LayerMode(object) :#{{{
    def __init__(self, Nchannels, dim) :
        self.Nchannels = Nchannels
        self.dim = dim
        self.xx  = 0
    def __add__(self, other) :
        assert self.dim == other.dim
        return LayerMode(
            self.Nchannels + other.Nchannels,
            self.dim
            )
#}}}

class ArrowMode(object) :#{{{
    def __init__(self, layer_dict) :
        self.conv = layer_dict['conv']
        if self.conv is not 'Copy' :
            self.padding = layer_dict['conv_kw']['padding']
            self.stride = layer_dict['conv_kw']['stride']
            self.kernel_size = layer_dict['conv_kw']['kernel_size']
            self.crop_output = layer_dict['crop_output']
            self.inplane = layer_dict['inplane']
            self.outplane = layer_dict['outplane']
    def out_layer(self, in_layer) :
        if self.conv is 'Copy' :
            return in_layer
        elif self.conv is 'Conv' :
            assert self.inplane == in_layer.Nchannels
            return LayerMode(
                self.outplane,
                (in_layer.dim+2*self.padding-(self.kernel_size-1)-1+self.stride)/self.stride-int(self.crop_output)
                )
        elif self.conv is 'ConvTranspose' :
            assert self.inplane == in_layer.Nchannels
            return LayerMode(
                self.outplane,
                (in_layer.dim-1)*self.stride-2*self.padding+(self.kernel_size-1)+1-int(self.crop_output)
                )
        else :
            raise RuntimeError('Unknown convolution')
#}}}

class Layer(object) :#{{{
    def __init__(self, point, mode) :
        self.xx = point.xx
        self.yy = point.yy
        self.mode = mode
        self.description = ''
        
        self.add_text(
            r'$%d\times%d^3$'%(self.mode.Nchannels,self.mode.dim)
            )
    def draw(self) :
#        FILE.write(
#            r'\draw[thick] (%.2f,%.2f) -- (%.2f,%.2f);'%(self.xx,self.yy-0.5*VERT,self.xx,self.yy+0.5*VERT)
#            )
#        FILE.write('\n')
        FILE.write(
            r'\draw (%.2f,%.2f) -- (%.2f,%.2f);'%(
                self.xx-T, self.yy-0.5*abs(VERT)-0.5*A, self.xx-T, self.yy+0.5*abs(VERT)-0.5*A
                )
            )
        FILE.write('\n')
        FILE.write(
            r'\draw (%.2f,%.2f) -- (%.2f,%.2f);'%(
                self.xx, self.yy-0.5*abs(VERT)-0.5*A, self.xx, self.yy+0.5*abs(VERT)-0.5*A
                )
            )
        FILE.write('\n')
        FILE.write(
            r'\draw (%.2f,%.2f) -- (%.2f,%.2f);'%(
                self.xx-T, self.yy-0.5*abs(VERT)-0.5*A, self.xx, self.yy-0.5*abs(VERT)-0.5*A
                )
            )
        FILE.write('\n')
        FILE.write(
            r'\draw (%.2f,%.2f) -- (%.2f,%.2f);'%(
                self.xx-T, self.yy+0.5*abs(VERT)-0.5*A, self.xx, self.yy+0.5*abs(VERT)-0.5*A
                )
            )
        FILE.write('\n')
        FILE.write(
            r'\draw (%.2f,%.2f) -- (%.2f,%.2f);'%(
                self.xx+B, self.yy-0.5*abs(VERT)+0.5*A, self.xx+B, self.yy+0.5*abs(VERT)+0.5*A
                )
            )
        FILE.write('\n')
        FILE.write(
            r'\draw (%.2f,%.2f) -- (%.2f,%.2f);'%(
                self.xx, self.yy-0.5*abs(VERT)-0.5*A, self.xx+B, self.yy-0.5*abs(VERT)+0.5*A
                )
            )
        FILE.write('\n')
        FILE.write(
            r'\draw (%.2f,%.2f) -- (%.2f,%.2f);'%(
                self.xx, self.yy+0.5*abs(VERT)-0.5*A, self.xx+B, self.yy+0.5*abs(VERT)+0.5*A
                )
            )
        FILE.write('\n')
        FILE.write(
            r'\draw (%.2f,%.2f) -- (%.2f,%.2f);'%(
                self.xx-T, self.yy+0.5*abs(VERT)-0.5*A, self.xx+B-T, self.yy+0.5*abs(VERT)+0.5*A
                )
            )
        FILE.write('\n')
        FILE.write(
            r'\draw (%.2f,%.2f) -- (%.2f,%.2f);'%(
                self.xx-T+B, self.yy+0.5*abs(VERT)+0.5*A, self.xx+B, self.yy+0.5*abs(VERT)+0.5*A
                )
            )
        FILE.write('\n')
        self.set_text()
    def add_text(self, text) :
        self.description += r'%s '%text
    def set_text(self) :
        FILE.write(
            r'\node [%s, rotate=90] at (%.2f,%.2f) {%s};'%(
                'right' if VERT>0 else 'left',
                self.xx, self.yy+0.5*VERT,
                self.description
                )
            )
        FILE.write('\n')
#}}}

class Arrow(object) :#{{{
    def __init__(self, in_layer, point_out, mode) :
        self.xin = in_layer.xx
        self.xout = point_out.xx
        self.yin = in_layer.yy
        self.yout = point_out.yy
        self.mode = mode
        self.description = ''
    def draw(self, pos='non_defaults') :
        self.set_text(pos)
        if self.mode.conv is 'Copy' :
            ls = 'dashed'
        else :
            ls = 'thick'
        if self.yin == self.yout :
            FILE.write(
                r'\draw[->,%s] (%.2f,%.2f) -- (%.2f,%.2f);'%(
                    ls,
                    self.xin+0.5*B,self.yin,self.xout-T,self.yout
                    )
                )
            FILE.write('\n')
        else :
            FILE.write(
                r'\draw[%s] (%.2f,%.2f) -- (%.2f,%.2f);'%(
                    ls,
                    self.xin+0.5*B, self.yin+np.sign(self.yout-self.yin)*0.25*VERT*np.sign(VERT),
                    0.5*(self.xin+0.5*B+self.xout-T), self.yin+np.sign(self.yout-self.yin)*0.25*VERT*np.sign(VERT)
                    )
                )
            FILE.write('\n')
            FILE.write(
                r'\draw[%s] (%.2f,%.2f) -- (%.2f,%.2f);'%(
                    ls,
                    0.5*(self.xin+0.5*B+self.xout-T), self.yin+np.sign(self.yout-self.yin)*0.25*VERT*np.sign(VERT),
                    0.5*(self.xin+0.5*B+self.xout-T), self.yout-np.sign(self.yout-self.yin)*0.25*VERT*np.sign(VERT)
                    )
                )
            FILE.write('\n')
            FILE.write(
                r'\draw [->,%s] (%.2f,%.2f) -- (%.2f,%.2f);'%(
                    ls,
                    0.5*(self.xin+0.5*B+self.xout-T), self.yout-np.sign(self.yout-self.yin)*0.25*VERT*np.sign(VERT),
                    self.xout-T, self.yout-np.sign(self.yout-self.yin)*0.25*VERT*np.sign(VERT)
                    )
                )
            FILE.write('\n')
    def add_text(self, text) :
        self.description += r'%s\\'%text
    def set_text(self, pos='non_defaults') :
        # TODO
        if self.description is not '' :
            if pos is 'non_defaults' :
                FILE.write(
                    r'\draw[ultra thin] (%.2f,%.2f) -- (%.2f,%.2f);'%(
                        0.5*(self.xin+0.5*B+self.xout-T), 0.5*(self.yin+self.yout),
                        0.5*(self.xin+0.5*B+self.xout-T), (NLevels+5)*VERT,
                        )
                    )
            elif pos is 'defaults' :
                FILE.write(
                    r'\draw[ultra thin] (%.2f,%.2f) -- (%.2f,%.2f);'%(
                        0.5*(self.xin+0.5*B+self.xout-T), 0.5*(self.yin+self.yout),
                        0.5*(self.xin+0.5*B+self.xout-T), (-4)*VERT,
                        )
                    )
            else :
                raise RuntimeError('pos must be either defaults or non_defaults.')
            FILE.write('\n')
            if pos is 'non_defaults' :
                FILE.write(
                    r'\node[align=left,rotate=90,anchor=%s] at (%.2f,%.2f) {%s};'%(
                        'north east' if VERT>0 else 'north west',
                        0.5*(self.xin+0.5*B+self.xout-T), (NLevels+5)*VERT,
                        self.description
                        )
                    )
            elif pos is 'defaults' :
                FILE.write(
                    r'\node[align=left,anchor=%s,text width=%.2f cm] at (%.2f,%.2f) {%s};'%(
                        'south west' if VERT>0 else 'north west',
                        8,
                        0.5*(self.xin+0.5*B+self.xout-T), (-4)*VERT,
                        self.description
                        )
                    )
            FILE.write('\n')
#}}}

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
def __merge(source, destination):#{{{
    # overwrites field in destination if field exists in source, otherwise just merges
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            __merge(value, node)
        else:
            destination[key] = value
    return destination
#}}}
def __flatten_dict(d, parent_key='', sep='_'):#{{{
    items = []
    for k, v in d.items():
#        new_key = parent_key + sep + k if parent_key else k
        new_key = k
        if isinstance(v, collections.MutableMapping):
            items.extend(__flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
#}}}
__default_param_flat = __flatten_dict(__default_param)
__default_param_text = ''
for key, value in __default_param_flat.items() :
    if key is 'inplane' or key is 'outplane': continue
    __default_param_text += r'\mbox{\texttt{%s:%s}}, '%(key,value)

def draw_network(index) :#{{{
    global FILE
    global NLevels

    FILE = open('./tex_files/network%d.tex'%index, 'w')

    this_network = import_module('network_%d'%index).this_network
    NLevels = this_network['NLevels']
    
    xx = 0

    level_states = []
    level_states.append(LayerMode(1, DIM_IN)) # input size
    l = Layer(Point(0,0), level_states[0])
    l.add_text(r'\textbf{Input}')
    l.draw()
    l.set_text()
    xx += 1

    gave_defaults = False

    # contracting path
    for ii in xrange(NLevels) :
        if ii == len(level_states) : # need to copy from higher level
            amode = ArrowMode({'conv': 'Copy'})
            level_states.append(amode.out_layer(level_states[ii-1]))
            a = Arrow(l, Point(xx*HORI, ii*VERT), amode)
            a.draw()
            l = Layer(Point(xx*HORI, ii*VERT), level_states[ii])
            l.draw()
            xx += 1
        for jj,layer_dict in enumerate(this_network['Level_%d'%ii]['through' if ii==NLevels-1 else 'in']) :
            amode = ArrowMode(__merge(layer_dict, copy.deepcopy(__default_param)))
            level_states[ii] = amode.out_layer(level_states[ii])
            level_states[ii].xx = xx
            a = Arrow(l, Point(xx*HORI, ii*VERT), amode)
            # check is non-default settings apply
            layer_dict_flat = __flatten_dict(layer_dict)
            for key, value in layer_dict_flat.items() :
                if key is 'inplane' or key is 'outplane' :
                    continue
                if __default_param_flat[key] != value :
                    a.add_text(r'\texttt{%s:%s}'%(key, value))
            if a.description is '' and not gave_defaults :
                a.add_text(r'default parameters:')
                a.add_text(__default_param_text)
                gave_defaults = True
                a.draw('defaults')
            else :
                a.draw('non_defaults')
            l = Layer(Point(xx*HORI, ii*VERT), level_states[ii])
            l.draw()
            xx += 1

    # expanding path
    for ii in xrange(NLevels-2, -1, -1) :
        amode = ArrowMode({'conv': 'Copy'})
        a = Arrow(l, Point(xx*HORI, ii*VERT), amode)
        a.draw()
        if this_network['Level_%d'%ii]['concat'] :
            amode_cat = ArrowMode({'conv': 'Copy'})
            a = Arrow(Layer(Point(level_states[ii].xx*HORI, ii*VERT), level_states[ii]), Point(xx*HORI, ii*VERT), amode_cat)
            a.draw()
            level_states[ii] = level_states[ii] + level_states[ii+1]
        else :
            level_states[ii] = level_states[ii+1]
        l = Layer(Point(xx*HORI, ii*VERT), level_states[ii])
        l.draw()
        xx += 1
        for jj,layer_dict in enumerate(this_network['Level_%d'%ii]['out']) :
            amode = ArrowMode(__merge(layer_dict, copy.deepcopy(__default_param)))
            level_states[ii] = amode.out_layer(level_states[ii])
            a = Arrow(l, Point(xx*HORI, ii*VERT), amode)
            # check is non-default settings apply
            layer_dict_flat = __flatten_dict(layer_dict)
            for key, value in layer_dict_flat.items() :
                if key is 'inplane' or key is 'outplane' :
                    continue
                if __default_param_flat[key] != value :
                    a.add_text(r'\texttt{%s:%s}'%(key, value))
            a.draw()
            l = Layer(Point(xx*HORI, ii*VERT), level_states[ii])
            if ii == 0 and jj == len(this_network['Level_0']['out'])-1 :
                assert l.mode.dim == DIM_OUT
                assert l.mode.Nchannels == 1
                l.add_text(r'\textbf{Output}')
            l.draw()
            xx += 1

    FILE.close()
#}}}

index = 0
while True :
    try :
        print index
        draw_network(index)
        index += 1
    except ImportError :
        print 'loaded %d networks'%index
        break
