# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 13:53:15 2016

@author: zellinger
"""

from keras import backend as K
from keras.regularizers import Regularizer
#import theano.tensor as T
import theano as th
import numpy as np

class DomainRegularizer(Regularizer):
    def __init__(self,l=1,name='mmd',beta=1.0):
        self.uses_learning_phase = 1
        self.l=l
        self.name=name
        self.beta=beta

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        if not hasattr(self, 'layer'):
            raise Exception('Need to call `set_layer` on '
                            'ActivityRegularizer instance '
                            'before calling the instance.')
        regularizer_loss = loss
        sim = 0
        if len(self.layer.inbound_nodes)>1:
            if self.name=='mmd':
                sim = self.mmd(self.layer.get_output_at(0),
                               self.layer.get_output_at(1),
                               self.beta)
            elif self.name=='mmd5':
                sim = self.mmdK(self.layer.get_output_at(0),
                                self.layer.get_output_at(1),
                                5)
            elif self.name=='mmatch':
                sim = self.mmatch(self.layer.get_output_at(0),
                                  self.layer.get_output_at(1),
                                  5)
            elif self.name=='acmd':
                sim = self.acmd(self.layer.get_output_at(0),
                                self.layer.get_output_at(1),
                                5)
            elif self.name=='mmatchK':
                sim = self.mmatch(self.layer.get_output_at(0),
                                  self.layer.get_output_at(1),
                                  self.beta)
            elif self.name=='ecmd':
                sim = self.ecmd(self.layer.get_output_at(0),
                                self.layer.get_output_at(1),
                                5)
            else:
                print('ERROR: Regularizer not supported.')
            
        add_loss = K.switch(K.equal(len(self.layer.inbound_nodes),2),sim,0)
        regularizer_loss += self.l*add_loss
            
        return K.in_train_phase(regularizer_loss, loss)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'l': float(self.l)}
        
    def mmd(self, x1, x2, beta):
        x1x1 = self.gaussian_kernel(x1, x1, beta)
        x1x2 = self.gaussian_kernel(x1, x2, beta)
        x2x2 = self.gaussian_kernel(x2, x2, beta)
        diff = x1x1.mean() - 2 * x1x2.mean() + x2x2.mean()
        return diff
        
    def mmatch(self, x1, x2, n_moments):
        mx1 = x1.mean(0)
        mx2 = x2.mean(0)
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        dm = self.matchnorm(mx1,mx2)
        scms = dm
        for i in range(n_moments-1):
            scms+=self.scm(sx1,sx2,i+2)
        return scms
        
    def mmdK(self, x1, x2, n_moments):
        mx1 = x1.mean(0)
        mx2 = x2.mean(0)
        s1=mx1
        s2=mx2
        for i in range(n_moments-1):
            s1+=(x1**K.cast(i+2,'int32')).mean(0)
            s2+=(x2**K.cast(i+2,'int32')).mean(0)
        return ((s1-s2)**2).sum().sqrt()
        
    def gaussian_kernel(self, x1, x2, beta = 1.0):
        r = x1.dimshuffle(0,'x',1)
        return T.exp( -beta * T.sqr(r - x2).sum(axis=-1))
        
    def scm(self, sx1, sx2, k):
        ss1 = (sx1**K.cast(k,'int32')).mean(0)
        ss2 = (sx2**K.cast(k,'int32')).mean(0)
        return self.matchnorm(ss1,ss2)
        
    def matchnorm(self, x1, x2):
        return ((x1-x2)**2).sum().sqrt()# euclidean
    #    return T.abs_(x1 - x2).sum()# maximum
    #    return 1-T.minimum(x1,x2).sum()/T.maximum(x1,x2).sum()# ruzicka
    #    return kl_divergence(x1,x2)# KL-divergence
    
    def acmd(self, x1, x2, n_moments):
        mx1 = x1.mean(0)
        mx2 = x2.mean(0)
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        dm = self.matchnorm(mx1,mx2)
        acms = dm
        for i in range(n_moments-1):
            acms+=self.acm(sx1,sx2,i+2)
        return acms
        
    def acm(self, sx1, sx2, k):
        ss1 = (T.abs_(sx1)**K.cast(k,'int32')).mean(0)
        ss2 = (T.abs_(sx2)**K.cast(k,'int32')).mean(0)
        return self.matchnorm(ss1,ss2)
        
#    def ecmd(self, x1, x2, n_moments):
#        # x-dim: n_samples x n_features
#        # E[\sup_k|c_k(F_i)-c_k(G_i)|]
#        n_features = 256
#        expected_cmd = 0
#        for i in range(n_features):
#            mom_diff = self.centr_moments(x1[:,i],n_moments)-self.centr_moments(x2[:,i],n_moments)
#            cmd = T.max(mom_diff)
#            expected_cmd+=cmd
#        expected_cmd=expected_cmd/n_features
#        return expected_cmd
#            
#    def centr_moments(self,x,n_moments):
#        # x-dim: n_samples x 1
#        cmoms = TT.shared(np.full((n_moments,1),np.nan))
#        
#        cmoms.get_value()[0,0] = x.mean(0)
#        x_cent = x-cmoms[0]
#        for k in range(n_moments-1):
#            cmoms.get_value()[k,0]=(x_cent**T.cast(k+2,'int32')).mean(0)
#        return cmoms.get_value()

    def centr_moments(self,x,n_moments):
        # returns n_moments X n_features
        x_cent = x-x.mean(0)
        result, updates = th.scan(fn=lambda prior_result, x_cent: prior_result * x_cent,
                                  outputs_info=T.ones_like(x_cent),
                                  non_sequences=x_cent,
                                  n_steps=n_moments)
        cmoms = result.mean(1)
        # add mean as first moment
        cmoms = T.set_subtensor(cmoms[0,:],x.mean(0))
        return cmoms

    def ecmd(self, x1, x2, n_moments):
        cmoms_x1 = self.centr_moments(x1, n_moments)
        cmoms_x2 = self.centr_moments(x2, n_moments)
#        # old cmd
#        mom_res = (cmoms_x1-cmoms_x2)**2
#        cmd_old = T.sqrt(mom_res.sum(1)).sum()
#        return cmd_old
#        # expected central moment discrepancy (~same accuracy)
#        mom_res = (cmoms_x1-cmoms_x2)**2
#        sup_diff = 256*T.max(mom_res,axis=0)
#        marginal_expected_cmd = sup_diff.sum()
#        return marginal_expected_cmd
        # expected central moment discrepancy
        mom_res = K.abs_(cmoms_x1-cmoms_x2)
        sup_diff = K.max(mom_res,axis=0)
        marginal_expected_cmd = sup_diff.sum()
        return marginal_expected_cmd
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
        
        
        
        
        
        