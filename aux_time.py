# -*- coding: utf-8 -*-
""" Auxiliary functions for measuring time. """

import time


class Timer():
    
    last_measurement = {}
    
    def __init__(self, timer_name="Timer"):
        self.timer_name = timer_name
                        
    def __enter__(self):
        self.start = time.time()                
        
    def __exit__(self, type, value, traceback):
        elapsed = (time.time()-self.start)
        Timer.last_measurement[self.timer_name] = elapsed
        
    @classmethod
    def get_elapsed(Timer, timer_name="Timer"):
        return Timer.last_measurement.get(timer_name, None)
    
    @classmethod
    def get_measurements(Timer):
        return Timer.last_measurement   


class AccumulatingTimer():
    
    shared_dict = {}
    enabled = True;
        
    def __init__(self, timer_name="Timer", save_to_dict=None):
        if not self.enabled: return
        if save_to_dict is None: save_to_dict = AccumulatingTimer.shared_dict
        self.save_to_dict = save_to_dict
        self.timer_name = timer_name
                        
    def __enter__(self):
        if not self.enabled: return
        self.start = time.time()                
        
    def __exit__(self, type, value, traceback):
        if not self.enabled: return
        elapsed = (time.time()-self.start)
        self.save_to_dict[self.timer_name] = self.save_to_dict.get(self.timer_name, 0.) + elapsed
        self.save_to_dict[self.timer_name+"-count"] = self.save_to_dict.get(self.timer_name+"-count", 0) + 1
        
    @classmethod
    def get_elapsed(AccumulatingTimer, timer_name="Timer"):
        return AccumulatingTimer.shared_dict.get(timer_name, 0.0)
        
    @classmethod            
    def get_report(AccumulatingTimer):
        return AccumulatingTimer.shared_dict
                           
    @classmethod            
    def get_report_pd(AccumulatingTimer):
        import pandas as pd
        keys = sorted(AccumulatingTimer.shared_dict.keys())
        return pd.DataFrame([[AccumulatingTimer.shared_dict[k] for k in keys]]).rename(columns=dict(enumerate(keys)))
                           
    @classmethod            
    def enable(AccumulatingTimer):
        AccumulatingTimer.enabled = True
        
    @classmethod            
    def disable(AccumulatingTimer):
        AccumulatingTimer.enabled = False


def timing(f):
    def measured_function(*args, **kwargs): 
        with AccumulatingTimer(f.__name__):
            results = f(*args, **kwargs)   
            #from aux import assert_valid    
            #assert_valid(results, f.__name__)
            return results
        
    #measured_function.__name__ = "<time measuring wrapper of %s>" % f.__name__
    measured_function.__name__ = f.__name__
    return measured_function        
