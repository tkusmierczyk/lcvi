# -*- coding: utf-8 -*-
""" Auxiliary functions mostly for printing and processing PyTorch tensors. """

import sys
import re
import numpy as np
import torch
import traceback
import json


def tonumpy(tensor):
    return tensor.cpu().detach().numpy()


def tonumpy2(tensor):
    return np.array( tensor.cpu().detach().numpy() )


def is_valid(w):
    if w is None: return False
    if torch.isnan(w).any().item(): return False
    if (w==float('inf')).any().item(): return False
    if (w==float('-inf')).any().item(): return False
    return True
    

def assert_valid(w, msg=""):
    assert is_valid(w), msg
    

def flatten_first_two_dims(t):
    return t.view( (t.shape[0]*t.shape[1], ) + t.shape[2: ] )

    
def print2(txt):
    sys.stdout.write(txt+"\n")    
    sys.stdout.flush()
    

def sel_matching(dct, key_regex="^.*"):
    return dict((k,v) for k, v in dct.items() if re.match(key_regex, k) is not None)


def sel_with_prefix(dct, prefix=""):
    six = len(prefix) #skip prefix
    return dict((k[six: ],v) for k, v in dct.items() if k.startswith(prefix))


def sel_without_prefix(dct, prefix=""):    
    return dict((k,v) for k, v in dct.items() if not k.startswith(prefix))


def get_traceback():
    return ''.join(l for l in traceback.format_stack() if "anaconda" not in l)


_annouced = {}
def print_numtimes(method, msg, num_times=1):
    if _annouced.get(method, 0)>=num_times: return
    _annouced[method] = _annouced.get(method, 0) + 1
    print2("%s" % (method))
announce_yourself = print_numtimes


_print_counter = {}
def sparse_print(method, msg, num_times=10, first_times=10):
    if num_times<=1:
        print2(msg)
        return
    
    _print_counter[method] = _print_counter.get(method, 0) + 1
    if _print_counter[method]%num_times==0 or _print_counter[method]<=first_times: 
        print2("[exec no:%i]%s" % (_print_counter[method], msg))


def retrieve_param(name, kwargs):
    if name not in kwargs or kwargs[name] is None:
        raise NameError("Param <%s> is not set!" % name)
    return kwargs[name]


def retrieve_param_opt(name, kwargs, default=None):
    if name not in kwargs or kwargs[name] is None: 
        return default
    return kwargs[name]  
    

def format_value(v):
    if hasattr(v, "__name__"): return v.__name__
    if hasattr(v, "shape"): return v.shape
    return str(v)[:15]       


def parse_args(s):    
    if s is None or s=="" or s=="-f": return {}
    s = s.replace("=", ":")
    s = s.replace("," , ",\"")
    s = s.replace(":" , "\":")
    s = s.replace("[" , "\"").replace("]" , "\"")
    s = "{\""+s+"}"
    return json.loads(s)


def parse_script_args():
    args_str = sys.argv[1] if len(sys.argv)>1 else "" 
    print2("parsing: <%s>" % args_str)
    return parse_args(args_str)


def dict2str(variables, SKIP_KEYS = ["OUT", "DESCRIPTION", "RED", "GREEN", "BLUE"]): 
    """ Try passing variables = globals(). """    
    params = sorted((k for k in variables.keys() if k==k.upper() and k[0]!="_" and k not in SKIP_KEYS), 
                    key=lambda v: (v, len(v)) )
    return " ".join( ("%s=%s" % (k, variables[k])) for k in params if type(variables[k]) in [int, float, str] )



