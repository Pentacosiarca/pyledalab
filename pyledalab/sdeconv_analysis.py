# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:28:25 2017

@author: Alberto Gonzalez Olmos, alberto.gonzalezolmos@glasgow.ac.uk

Python version of the sdeconv_analysis function from ledalab

Define outside of the class

         cond_data = cond_data
         time_data = time_data
         sampling_rate = sampling_rate
         smoothwin = smoothwin
        
"""

##########################################
## test data

#import os
#os.chdir('C:/Users/alber/OneDrive - University of Glasgow/PhD/Software/Python/EDA/cvxEDA-master/src/')     
#from numpy import genfromtxt
#cond_data = genfromtxt('testEDA.csv', delimiter=',')
#sampling_rate = 1000

##########################################

import numpy as np
import math
# from scipy import ndimage
from scipy import signal
from scipy import interpolate
#from sklearn.linear_model import Ridge
#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt


class sdeconv_analysis_shell():
    def __init__(self, cond_data, sampling_rate):
        
        self.cond_data = cond_data
        self.sampling_rate = sampling_rate
        # Variables
        
        self.tauMin = 1.0000e-03
        self.tauMinDiff = 0.01
        self.sigPeak = 1.0000e-03
        self.segmWidth = 12
        x = [1.,3.75] # see Benedek & Kaernbach, 2010, J Neurosc Meth # after optimization(cgt) in matlab[0.267,2.3136] # 
        self.tonicGridSize_sdeco = 10
        self.dist0_min = 0
        
        
        self.smoothwin = 0.2 # smoothwin, or input param ?
        
        cond_datad, sampling_rated = self.downsampling(cond_data, sampling_rate) # check if it is necessary to downsample the data before calculations
        
        self.cond_data = cond_datad
        self.sampling_rate = sampling_rated
        self.time_data = np.linspace(0,(1. /self.sampling_rate) * (len(self.cond_data)-1),len(self.cond_data))
        
        self.tonicData, err, x = self.sdeconv_analysis(x)
        
        
        self.countElseMF = []
        self.countElseMF.append(0)
        xopt = self.cgd(x, self.sdeconv_analysis, [.3,2], .01, 20, .05)
#        print('error pre opt' + str(x))
#        print('error after opt' + str(xopt))
        
        self.tonicData2, err, x = self.sdeconv_analysis(xopt)
        
    
    def sdeconv_analysis(self, x):
        # Check Limits
        x[0] = self.withinlimits(  x[0], self.tauMin, 10) # withinlimits(x[0], self.tauMin, 10)
        x[1] = self.withinlimits(  x[1], self.tauMin, 20) # withinlimits(x[1], self.tauMin, 20)
        if x[1] < x[0]:  # tau1 < tau2
            x[0:2] = np.flipud(x[0:2])

        if np.abs(x[0]-x[1]) < self.tauMinDiff:
            x[1] = x[1] + self.tauMinDiff
        
        tau = [x[0],x[1]]

        data = self.cond_data #  cond_data # conductance data
        t = self.time_data #  time_data  # time data
        sr = self.sampling_rate #  sampling_rate  # samplingrate
        smoothwin = self.smoothwin * 8 #  smoothwin * 8 # Gauss 8 SD
        dt = 1./sr
        winwidth_max = 3. # sec
        swin = np.round(np.min((smoothwin, winwidth_max)) * sr)
        
        d = data
        
        # Data preparation
        tb = np.zeros((1,len(data)))
        tb[0,:] = t - t[0] + dt
        
        bg = np.zeros((1,len(data)))
        bg[0,:] = self.bateman_gauss(  tb, 5, 1, 2, 40, .4)
        mx = np.max((bg))
        idx = list(np.where(bg[0,:] == mx)[0])
        
        prefix = bg[0,0:(idx[0]+2)] / bg[0,idx[0]+2] * d[0] # +10
        prefix = [i for i in prefix if i != 0]
        n_prefix = len(prefix)
        d_ext = np.zeros((1,len(np.concatenate((prefix,d)))))
        d_ext[0,:] = np.concatenate((prefix,d))
        
        t_ext = np.linspace(t[0]-n_prefix*dt,t[0]-dt,n_prefix)
        t_ext = np.concatenate((t_ext,t))
        
        tb = np.zeros((1,len(t_ext)))
        tb[0,:] = t_ext - t_ext[0] + dt
        
        kernel = self.bateman_gauss(tb, 0, 0, tau[0], tau[1], 0)
        
        # Adaptive kernel size
        mx = np.max((kernel))
        midx = list(np.where(kernel[0,:] == mx)[0])
        
        kernelaftermx = kernel[0,midx[0]+1:]
        kernelaftermx = [i for i in kernelaftermx if i > 10**-5]
        kernel_pre = kernel
        kernel = np.zeros((1,len(np.concatenate((kernel[0,0:midx[0]+1], kernelaftermx)))))
        kernel[0,:] = np.concatenate((kernel_pre[0,0:midx[0]+1], kernelaftermx))
        kernel = kernel / np.sum(kernel) # normalize to sum = 1
        
        sigc = np.max((.1, self.sigPeak/np.max((kernel))*10))   # threshold for burst peak
        
        # ESTIMATE TONIC
        
        kernel = np.zeros((1,len(np.concatenate((kernel[0,0:midx[0]+1], kernelaftermx)))))
        kernel[0,:] = np.concatenate((kernel_pre[0,0:midx[0]+1], kernelaftermx))
        
        signal_toDeconvolve = np.zeros((1,np.concatenate((d_ext, d_ext[0,-1]*np.ones((1,len(kernel[0,:])-1))),axis = 1).shape[1]))
        signal_toDeconvolve[0,:] = np.concatenate((d_ext, d_ext[0,-1]*np.ones((1,len(kernel[0,:])-1))),axis = 1)
        
        driverSC, remainderSC = signal.deconvolve(signal_toDeconvolve[0,:], kernel[0,:]) # signal.deconvolve()
    
        driverSC_smooth = self.smooth(driverSC, swin, 'gauss')
        # Shorten to data range
        driverSC = driverSC[n_prefix:]
        driverSC_smooth = driverSC_smooth[n_prefix:]
        remainderSC = remainderSC[n_prefix:len(d)+n_prefix]

        # Inter-impulse fit
        onset_idx, impulse, overshoot, impMin, impMax = self.segment_driver(driverSC_smooth, np.zeros((len(driverSC_smooth))), sigc, np.round(sr * self.segmWidth)) # Segmentation of non-extended data!
        #if estim_tonic
        tonicDriver, tonicData = self.sdeco_interimpulsefit(driverSC_smooth, kernel, impMin, impMax)
        #else
            #tonicDriver = leda2.analysis0.target.tonicDriver
            #nKernel = length(kernel)
            #tonicData = conv([tonicDriver(1)*ones(1,nKernel), tonicDriver], kernel)
            #tonicData = tonicData(nKernel:length(tonicData) - nKernel)
        
        # Build tonic and phasic data
        #phasicData = d - tonicData
        phasicDriverRaw = driverSC[0:min(len(driverSC),len(tonicDriver))] - tonicDriver[0:min(len(driverSC),len(tonicDriver))]
        phasicDriver = self.smooth(phasicDriverRaw, swin, 'gauss')
     
        # Compute model error
        #err_MSE = fiterror(data, tonicData+phasicData, 0, 'MSE')
        #err_RMSE = sqrt(err_MSE)
        #err_chi2 = err_RMSE / leda2.data.conductance.error
        #err1d = deverror(phasicDriver, .2)
        err_discreteness = self.succnz(phasicDriver, max(.01, max(phasicDriver)/20), 2, sr)
        phasicDriverNeg = np.array(phasicDriver)
        wP = list([map(lambda x: x > 0, phasicDriver)][0])
        phasicDriverNeg[wP] = 0
        err_negativity = np.sqrt(np.mean(phasicDriverNeg**2))
        
        # CRITERION
        alpha = 5
        err = err_discreteness + err_negativity * alpha
  
        return tonicData, err, x
    
    def bateman_gauss(self, time, onset, amp, tau1, tau2, sigma):

        component =  self.bateman(time,onset,0,tau1,tau2)
        
        if sigma > 0:
            sr = np.round(1/np.mean(np.diff(time)))
            winwidth2 = int(np.ceil(sr*sigma*4)) # round half winwidth: 4 SD to each side
            t = np.zeros((1,winwidth2*2+1))
            t[0,:] = np.linspace(1,winwidth2*2+1,winwidth2*2+1) # odd number (2*winwidth-half+1)
            g =  self.normpdf(t, winwidth2+1, sigma*sr)
            g = np.squeeze((g / np.max(g)) * amp)

            comp = np.concatenate((np.ones((1,winwidth2))*component[0,0], component, np.ones((1,winwidth2))*component[0,-1]),axis=1)
            comp = np.squeeze(comp)
            
            bg = np.convolve(comp, g)
            
            component = bg[winwidth2*2 : -winwidth2*2]
            
        return component
    
    def bateman(self, time, onset, amp, tau1, tau2):

        if tau1 < 0 or tau2 < 0:
            print("error tau1 or tau2 < 0: " + str(tau1) + ", " +str(tau2))
        
        if tau1 == tau2:
            print("error tau1 == tau2 == " + str(tau1))
        
        conductance = np.zeros(time.shape)
        time_range = np.where(time[0,:] > onset)
        if not list(time_range):
            raise Exception("in bateman, time range == []!")

        xr = time[0,time_range] - onset
        
        if amp > 0:
            maxx = tau1 * tau2 * np.log(tau1/tau2) / (tau1 - tau2)  # b' = 0
            maxamp = np.abs(np.exp(-maxx/tau2) - np.exp(-maxx/tau1))
            c =  amp/maxamp
        
        else: # if amp == 0: normalized bateman, area(bateman) = 1/sr
            sr = np.round(1/np.mean(np.diff(time)))
            c = 1/((tau2 - tau1) * sr)
        
        if tau1 > 0:
            conductance[0,time_range] = c * (np.exp(-xr/tau2) - np.exp(-xr/tau1))
        else:
            conductance[0,time_range] = c * np.exp(-xr/tau2)

        return conductance
    
    def normpdf(self, x, mu, sigma):

        y = np.divide(np.exp(-0.5 * np.power(np.divide((x - mu),sigma),2)),(np.multiply(np.sqrt(2*math.pi),sigma)))
        
        return y

    def withinlimits(self, w_in, lowerlimit, upperlimit):

        w_out = np.max((np.min((w_in, upperlimit)),lowerlimit))

        return w_out
    
    def downsampling(self, data, sampling_rate):
        Fs_min = 4
        N_max = 3000
        Fs = int(sampling_rate)
        N = len(data)
        
        if N > N_max:
                factorL = self.factors(Fs)
                FsL = np.divide(Fs,factorL)
                
                idx = list(np.where(FsL >= Fs_min)[0])

                factorL = [factorL[i] for i in idx]
                FsL = [FsL[i] for i in idx]
                
                if not list(factorL):
                    raise Exception("in downsampling, no factors found for the sample frequency, downsampling not possible!")
                else:

                    N_new = np.divide(N,factorL)
                    idx = list(np.where(N_new < N_max)[0])
                    if not idx:
                        idx = len(factorL)  # if no factor meets criterium, take largest factor
                    else:
                        idx = idx[0]

                    fac =  factorL[idx]
                    time = np.linspace(0,1./Fs * len(data),len(data))
                    td, scd =  self.downsamp(time, data, fac, 'step') # difference between downsampling and decimate
                    time = td
                    data = scd
                    sampling_rate = FsL[idx]
        
        return data, sampling_rate
    
    def factors(self, n):  
        
        return [x for x in range(1, int(n+1)) if n % x == 0]
    
    def downsamp(self, t, data, fac, downsampling_method):

        # N = len(data) # #samples
        if downsampling_method =='step':
            t = t[0::fac]
            data = data[0::fac]
            
            '''
            ## NOT finished
        elif downsampling_method == 'mean':
            t = t(1:end-mod(N, fac))
            t = np.mean(reshape(t, fac, [])).T
            data = data[0:N % fac] # reduce samples to match a multiple of <factor>
            data = np.mean(reshape(data, fac, [])).T # Mean of <factor> succeeding samples
            
        else:
            t = t[0::fac]
            data = smooth(data, 2^fac, downsampling_method) # in ledalab it is preset to Gauss
            data = data[0::fac]
            '''

        return t, data
    
    def smooth(self, data, winwidth, smooth_type):

        if winwidth < 1:
            sdata = data
            
        else:
            if smooth_type == None:
                smooth_type = 'gauss'
            
            data = list(data) # data is a list
            data.insert(0,data[0])
            data.append(data[-1])
    
            winwidth = np.floor(winwidth/2)*2 # force even winsize for odd window
                  
            if smooth_type == 'hann':
                    window = 0.5*(1 - np.cos(2*math.pi*np.linspace(0,1,winwidth+1))) # hanning window
            elif smooth_type == 'mean':
                    window = np.ones((1,int(winwidth+1))) # moving average
            elif smooth_type == 'gauss':
                    window = self.normpdf(np.linspace(1,(winwidth+1),(winwidth+1)), winwidth/2+1, winwidth/8)
            elif smooth_type == 'expl':
                    window = np.append(np.zeros(np.round(winwidth/2)), np.exp(-4*(np.linspace(0,1,1+np.round(1/(2/winwidth))))))
            else:
                    print("Unknown smoothing type")
                    
            window = window / np.sum(window) # normalize window
            winwidth = int(winwidth)
            
            data_ext = list((np.ones((1,winwidth/2))*data[0])[0])
            data_ext.extend(data)
            data_ext.extend(list((np.ones((1,winwidth/2))*data[-1])[0]))
    
            sdata_ext = list(np.convolve(data_ext, window)) # convolute with window
            sdata = sdata_ext[1+winwidth : -winwidth-1] # cut to data length
            
        return sdata
    
    def segment_driver(self, data, remd, sigc, segmWidth):
        
        segmOnset = []
        segmImpulse = []
        segmOversh = []
        impMin = []
        impMax = []
        
        cccrimin, cccrimax = self.get_peaks(data)
        if list(cccrimax) == []:
            segmOnset = []
            segmImpulse = [] 
            segmOversh = []
            impMin = []
            impMax = []
            
        else:
            # sigc = max(data(cccrimax(2:end)))/100;  %relative criterium for sigc
            minL, maxL = self.signpeak(data, cccrimin, cccrimax, sigc)
            
            '''
            %for i = 1:length(cccrimax)
            %s(i) = sum(data(cccrimin(i):cccrimin(i+1))) * dt # area of possible segment
            %maximum difference of min-max or max-min
            %end
            %maxL = cccrimax(data(cccrimax) - data(cccrimin(1:end-1)) > sigc)
            %maxL = cccrimax(s > sigc)
            '''
    
            rmdimin, rmdimax = self.get_peaks(remd) # get peaks of remainder
            rmdimins, rmdimaxs = self.signpeak(remd, rmdimin, rmdimax, .005) # get remainder segments
            
            segmOnset = []
            segmImpulse = []
            # Segments: 12 sec, max 3 sec preceding maximum
            for i in range(0,len(maxL)):
                segm_start = int(max(minL[i][0], maxL[i] - np.round(self.segmWidth/2)))
                segm_end   = int(min(segm_start + self.segmWidth - 1, len(data) - 1))
            
                # impulse
                segm_idx = range(segm_start,segm_end+1)
                segm_data = np.array(data)[segm_idx]
                segm_data[segm_idx >= minL[i][1]] = 0
                segmOnset.append(segm_start)
                segmImpulse.append(segm_data)
            
                # overshoot
                oversh_data = np.zeros((1,len(segm_idx)))
                if i < len(maxL):
                    rmi = list([rmdimaxs > maxL[i] and rmdimaxs < maxL[i+1]][0])
                else:
                    rmi = list([rmdimaxs > maxL[i]][0])
    
    
                # no zero overshoots
                if rmi == []:
                    if i < len(maxL):
                        rmi = list([rmdimaxs > maxL[i] and rmdimaxs < maxL[i+1]][0])
                    else:
                        rmi = list([rmdimaxs > maxL[i]][0])
    
                    rmdimaxs = rmdimax
                    rmdimins = [rmdimin[0:-2] ,rmdimin[1:]]
            
                if rmi != []:
                    rmi = rmi[0]
                    oversh_start = max(rmdimins[rmi[0],0], segm_start)
                    oversh_end = min(rmdimins[rmi[0],1], segm_end) # min(rmdimins(rmi+1), segm_end)
                    oversh_data[oversh_start - segm_start : - (segm_end - oversh_end)] = remd[oversh_start:oversh_end]
    
                '''
                %     %     if mean(oversh_data) < 2*leda2.data.conductance_error
                %     %         oversh_data = zeros(size(segm_idx));
                %     %     end
            
                %     oversh_data = remd(segm_idx);
                %     oversh_data(segm_idx < maxL(i)) = 0;
                %     if i < length(maxL)
                %         oversh_data(segm_idx >= maxL(i+1)) = 0;
                %     end
                %
                '''
                segmOversh.append(oversh_data)
    
            
            impMin = minL
            impMax = maxL
        
        return segmOnset, segmImpulse, segmOversh, impMin, impMax
    
    def get_peaks(self, data):

        ccd = np.subtract(data[1:],data[0:-1]) # Differential
        '''
        Search for signum changes in first differntial:
        slower but safer method to determine extrema than looking for zeros (taking into account
        plateaus where ccd does not immediatly change the signum at extrema)
        
        '''
        
        ccd_filt = ccd
        ccd_filt = list(filter(lambda x: x!=0, ccd_filt))

        if ccd_filt == []: # data == zeros(1,n)
            cccrimin = []
            cccrimax = []
        else:
        
            start_idx = list(ccd).index(ccd_filt[0])
            
            cccri = np.zeros((1, len(ccd)))
            cccriidx = 1
            csi = np.sign(ccd[start_idx]) # currentsignum = current slope
            signvec = np.sign(ccd)
            for i in range(start_idx+1,len(ccd)):
                if signvec[i] != csi:
                    cccri[0,cccriidx] = i
                    cccriidx += 1
                    csi = -csi
            
            if cccriidx == 1: # no peak as data is increasing only
               raise Exception("in get_peaks, cccriidx == 2:, no peak as data is increasing only!")
            
            # if first extrema = maximum, insert minimum before
            if np.sign(ccd[start_idx]) == 1:
               predataidx = range(start_idx,int(cccri[0,1]))
               mn = min(np.array(data)[predataidx])
               idx = data.index(mn)
               cccri[0] =  predataidx[idx]
            
            # if last extremum is maximum add minimum after it
            if (cccriidx - (cccri[0,0]==0))% 2:
                cccri[0,cccriidx] = len(data)
                cccriidx = cccriidx + 1
            
            # crop cccri from the first minimum to the last written index
            cccri = cccri[0,(cccri[0,0]==0):cccriidx]
            
            cccri = np.sort(cccri)
            
            cccrimin = cccri[::2] # list of minima
            cccrimax = cccri[1::2] # list of maxima

        return cccrimin, cccrimax
    
    def signpeak(self, data, cccrimin, cccrimax, sigc):
        
        if list(cccrimax) == []:
            minL = []
            maxL = []
            
        else:
            dmm = np.zeros((2,max(len(cccrimin),len(cccrimax))-1))
            
            dmm[0,:] = np.array(data)[cccrimax.astype(int)[:-1]] - np.array(data)[cccrimin.astype(int)[:-1]]
            dmm[1,:] = np.array(data)[cccrimax.astype(int)[:-1]] - np.array(data)[cccrimin.astype(int)[1:]]           
            maxL = list(cccrimax[list(np.where(np.max(dmm,axis=0) > sigc))])
            
            # keep only minima right before and after sign maxima
            
            minL = []
            for i in range(0,len(maxL)):
                minm1_idx = list(list(np.where(list(filter(lambda x: x < maxL[i], cccrimin))))[0])
                minL.append([cccrimin[minm1_idx[-1]], cccrimin[minm1_idx[-1]+1]])

        return minL, maxL
    
    def sdeco_interimpulsefit(self, driver, kernel, minL, maxL):

        t = self.time_data
        d =  self.cond_data
        sr = self.sampling_rate
        tonicGridSize = self.tonicGridSize_sdeco
        nKernel = len(kernel[0])
        
        
        # Get inter-impulse data index
        gap_idx = []
        iif_idx = []
        if len(maxL) > 2:
            for i in range(0,len(maxL)-1):
                gap_idx = range(int(minL[i][1]),int(minL[i+1][0])+1) # +1: removed otherwise no inter-impulse points may be available at highly smoothed data
                iif_idx += gap_idx[:]

            iif_idx.insert(0,int(minL[1][0]))
            iif_idx.extend(range(int(minL[-1][1]),len(driver)-sr+1))

        else:  # no peaks (exept for pre-peak and may last peak) so data represents tonic only, so ise all data for tonic estimation
            iif_idx = list(np.where(t > 0)[0])

        iif_t = t[iif_idx]
        iif_data = np.array(driver)[iif_idx]
        
        groundtime = []
        groundtime.extend(range(0,int(t[-2]),tonicGridSize))
        groundtime.append(t[-1])
        
        if tonicGridSize < 30:
            tonicGridSize = tonicGridSize*2
        
        groundlevel = []
        for i in range(0,len(groundtime)):
            # Select relevant interimpulse time points for tonic estimate at groundtime
            t_idx = []
            grid_idx = []
            xt = []
            yt = []
            xg = []
            yg = []
            
            if i == 0:
                
                xt = [iif_t <= (groundtime[i] + tonicGridSize)][0]
                yt = [iif_t > 1][0]
                t_idx = np.array([xt[j] and yt[j] for j in range(len(xt))])
                
                xg = [t <= (groundtime[i] + tonicGridSize)][0]
                yg = [t > 1][0]
                grid_idx = np.array([xg[j] and yg[j] for j in range(len(xt))])
                
            elif i == len(groundtime):
                
                xt = [iif_t > (groundtime[i] - tonicGridSize)][0]
                yt = [iif_t < (t[-1] - 1)][0]
                t_idx = np.array([xt[j] and yt[j] for j in range(len(xt))])
                
                xg = [t > (groundtime[i] - tonicGridSize)][0]
                yg = [t < (t[-1] - 1)][0]
                grid_idx = np.array([xg[j] and yg[j] for j in range(len(xt))])
                
            else:
                
                xt = [iif_t > (groundtime[i] - tonicGridSize/2)][0]
                yt = [iif_t <= (groundtime[i] + tonicGridSize/2)][0]
                t_idx = np.array([xt[j] and yt[j] for j in range(len(xt))])
                
                xg = [t > (groundtime[i] - tonicGridSize/2)][0]
                yg = [t <= (groundtime[i] + tonicGridSize/2)][0]
                grid_idx = np.array([xg[j] and yg[j] for j in range(len(xt))])
                
            # Estimate groundlevel at groundtime
            if len(list(np.where(t_idx == True))[0]) > 2:
                groundlevel.append(min(np.mean(iif_data[t_idx]),  d[int(self.time_idx(t, groundtime[i]))]))
            else:  # if no inter-impulses data is available
                groundlevel.append(min(np.median(np.array(driver)[grid_idx]),  d[int(self.time_idx(t, groundtime[i]))]))

        tonicDriver_obj = interpolate.PchipInterpolator(groundtime, groundlevel)
        xi = np.arange(groundtime[0],groundtime[-1]+(t[1]-t[0]),(t[1]-t[0]))
        tonicDriver = tonicDriver_obj(xi)
        
        convTonicDriver = list((tonicDriver[0]*np.ones((1,nKernel)))[0])
        convTonicDriver.extend(tonicDriver)
        tonicData = np.convolve(convTonicDriver, list(kernel)[0])
        tonicData = tonicData[nKernel:len(tonicData)-nKernel+1]
        
        # Correction for tonic sections still higher than raw data
        # Move closest groundtime at time of maximum difference of tonic surpassing data
        for i in range(len(groundtime)-2,-1,-1) :
        
            t_idx = self.subrange_idx(t, groundtime[i], groundtime[i+1])
            ddd = max((tonicData[t_idx] + self.dist0_min) - d[t_idx])
        
            if ddd > 2.2204e-16:
                # Move closest groundtime to maxmimum difference position and level

                groundlevel[i] -= ddd
                groundlevel[i+1] -= ddd
        
                tonicDriver_obj = interpolate.PchipInterpolator(groundtime, groundlevel)
                xi = np.arange(groundtime[0],groundtime[-1]+(t[1]-t[0]),(t[1]-t[0]))
                tonicDriver = tonicDriver_obj(xi)
                
                convTonicDriver = list((tonicDriver[0]*np.ones((1,nKernel)))[0])
                convTonicDriver.extend(tonicDriver)
                tonicData = np.convolve(convTonicDriver, list(kernel)[0])
                tonicData = tonicData[nKernel:len(tonicData)-nKernel+1]
        
        return tonicDriver, tonicData
    
    def time_idx(self, time, time0):

        idx = list(np.where(time >= time0)[0])
        if idx == []:
            idx = min(idx)
            time0_adj = time[idx]
            
            # check if there is a closer idex before
            if time0_adj != time[0]:
                time0_adjbefore = time[idx-1]
                if abs(time0 - time0_adjbefore) < abs(time0 - time0_adj):
                    idx = idx - 1
                    time0_adj = time0_adjbefore
            
        else:
            idx = list(np.where(time <= time0)[0])
            idx = max(idx)
            time0_adj = time[idx]
            
        return idx # , time0_adj # 
    
    def subrange_idx(self, t, t1, t2):
        
        t1_idx = []
        t1_idx.append(self.time_idx(t, t1))
        t2_idx = []
        t2_idx.append(self.time_idx(t, t2))
        
        if (t1_idx != []) and (t2_idx != []):
            idx = range(t1_idx[0],t2_idx[0]+1)
        else:
            idx = []
        
        return idx
    
    def succnz(self, data, crit, fac, sr):
        
        '''
        succnz calculates an index of how many successive values are
        above the parameter crit as described in section 4.3 of
        Benedek, M. & Kaernbach, C. (2010). A continuous measure of phasic
        electrodermal activity. J. Neurosci. Methods, 190, 80â€“91.
        '''
        
        n = len(data)
        
        abovecrit = np.array(map(lambda x: abs(x) > crit, data))
        nzidx = list(np.array(np.where(np.diff(abovecrit))[0]) + 1)
        
        if nzidx == []:
            snz = 0
            return
        
        
        # if the sequence begins with a value above crit prepend 1
        if abovecrit[0] == 1:
            nzidx.insert(0,0)
  
        # if the sequence ends with a value above crit append the length
        if abovecrit[-1] == 1:
            nzidx.append(n+1)

        '''
        now nzidx contains every position where data rises above crit (odd
        indices) or dips below crit (even indices).
        The lengths of spans above crit is the difference between the start index
        and the end index
        '''
        
        nzL = np.subtract(nzidx[1::2],nzidx[0::2])
        
        snz = sum(np.power((nzL/float(sr)),fac)/(float(n)/float(sr)))

        return snz

    def cgd(self, start_val, error_fcn, h, crit_error, crit_iter, crit_h):
        
        ### TEST
#        crit_error = [.01]
#        crit_iter = 20
#        crit_h = [.05]
#        h = [.3, 2.]
        ###
        
        
        
        x = start_val[:]
        a,newerror,b = error_fcn(x)
#        starterror = newerror

#        history.x = x
#        history.direction = zeros(size(x))
#        history.step = -1
#        history.h = -ones(size(h))
#        history.error = newerror

        iterV = 0
        
        while 1:
            iterV += 1
            olderror = newerror
            
            # GET GRADIENT
            if iterV == 1:
                
                gradient = self.cgd_get_gradient(x, olderror, error_fcn, h)
                direction = list(-gradient.T[0])
                if gradient == []:
                    break
            
            else:
                
                new_gradient = self.cgd_get_gradient(x, olderror, error_fcn, h)
#                old_direction = direction[:]
#                old_gradient = gradient[:]
        
                method = 1
                if method == 1:
                        # no conjugation
                    direction = list(-new_gradient.T[0])

#                elif method == 2:
#                    # Fletcher-Reeves
#                    beta = np.linalg.norm(new_gradient, 2) / np.linalg.norm(old_gradient, 2)
#                    direction = -new_gradient + beta * old_direction
#                elif method == 3:
#                    # Polak-Ribiere
#                    a = (new_gradient - old_gradient) * new_gradient'
#                    b = old_gradient * old_gradient'
#                    beta = max(a / b, 0)
#                    direction = -new_gradient + beta * old_direction
#                elif method == 4:
#                    # Hestenes-Stiefel
#                    a = (new_gradient - old_gradient) * new_gradient'
#                    b = old_direction * (new_gradient - old_gradient)'                        
#                    beta = a / b
#                    direction = -new_gradient + beta * old_direction     

        
            if any(direction):
                # LINESEARCH
                
                x, newerror, step = self.cgd_linesearch(x, olderror, direction, error_fcn, h)
                error_diff = newerror - olderror
            
            else:
                error_diff = 0 # empty gradient
                step = 0
        
            # history
#            history.x(iterV + 1, :) = x
#            history.direction(iterV + 1, :) = direction
#            history.step(iterV + 1) = step
#            history.h(iterV + 1, :) = h
#            history.error(iterV + 1) = newerror

        
            if iterV > crit_iter:
                break

            if error_diff > -crit_error: # no improvement
        
                h = np.divide(h,2)
                if all(h < crit_h):
                    break
        
        return x #, history
    
    def cgd_get_gradient(self, x, error0, error_fcn, h):
        
        
        Npars = len(x)
        gradient = np.zeros((Npars,1))
        
        for i in range(0,Npars):
        
            xc = x[:] # x_copy
            xc[i] = xc[i] + h[i]
        
            a,error1,b = error_fcn(xc)
        
            if error1 < error0:
                gradient[i,0] = (error1 - error0)
        
            else: # try opposite direction
                xc = x[:]
                xc[i] = xc[i] - h[i]
                a,error1,b = error_fcn(xc)
        
                if error1 < error0:
                    gradient[i,0] = -(error1 - error0)
        
                else:
                    gradient[i,0] = 0
        
        return gradient
    
    def cgd_linesearch(self, x, error0, direction, error_fcn, h):
        '''
        There are numerical differences with polyfit and polyval
        '''
        ### TEST ,x, olderror, direction, error_fcn, h
#        x = x # [1.,3.75] # 
#        error0 = olderror # 10.02 # 
#        direction = direction # [-2.4046,-4.6201] # 
#        h = h # [.3,2.] # 
        ###
        
        direction_n = direction / np.linalg.norm(direction,2)
        error_list = []
        error_list.append(error0)
        factor = []
        factor.append(0)
        stepsize = h
        maxSteps = 6
        
        endofdecline = 0
        #model = make_pipeline(PolynomialFeatures(5), Ridge())
        
        

        for iStep in range(1,maxSteps):
        
            factor.append(2**(iStep-1))
            xc = (x + direction_n.T * stepsize * factor[iStep])
            a,error_list_n, xc = error_fcn(xc) # xc may be changed due to limits
            error_list.append(error_list_n)
        
            if error_list[-1] >= error_list[-2]: # end of decline
                endofdecline = 1
                
                if iStep == 1: # no success
                    step = 0
                    error1 = error0
                    
                else: # parabolic
                    
                    p = np.polyfit(factor, (error_list - np.mean(error_list)), 2)
                    fx = np.linspace(factor[0],factor[-1],np.round(len(range(factor[0],factor[-1]))/.1)+1)
                    fy = np.polyval(p, fx)
#                    fy = model.predict(fx)
                    idx = np.where(fy == min(fy))[0][0]
                    fxm = fx[idx]
                    xcm = (x + direction_n.T * stepsize * fxm)
                    a, error1, xcm = error_fcn(xcm) # xc may be changed due to limits
                    
        
                    if error1 < error_list[iStep - 1]:
                        xc = xcm
                        step = fxm
                        
                    else: # finding Minimum did not work
                        xc = (x + direction_n.T * stepsize * factor[iStep-2])[0] # before last point
                        a, error1, xc = error_fcn(xc) # recalculate error in order to check for limits again
                        step = factor[iStep-2]
                        
                break 
        
        if endofdecline:
            
            return xc, error1, step
        
        step = factor[iStep]
        error1 = error_list[iStep]
        
        # Taylor-Check??
        
        return xc, error1, step
    
    def plot_EDAprepostOPT(self):
        
                # Three subplots sharing both x/y axes
        f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
        ax1.plot(self.time_data,self.tonicData,'r--',label='tonic data pre optimization')
        ax1.plot(self.time_data,self.cond_data,'b.',label='raw data')
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        ax2.plot(self.time_data,self.tonicData2,'g--',label='tonic data after optimization')
        ax2.plot(self.time_data,self.cond_data,'b.',label='raw data')

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        # Fine-tune figure; make subplots close to each other and hide x ticks for
        # all but bottom plot.
        f.subplots_adjust(hspace=0)
        plt.show()
        pass

    
if __name__ == '__main__':
    import os
    from numpy import genfromtxt
    
#    os.chdir('C:/Users/alber/OneDrive - University of Glasgow/PhD/Software/Python/EDA/cvxEDA-master/src/') 
#    EDAcsv = 'testEDA10.csv'    
    
    os.chdir('C:/Users/alber/OneDrive - University of Glasgow/PhD\/database_OWN/PT/E4/1528900404_A0118A/')
    EDAcsv = 'EDA.csv'
    
    cond_data = genfromtxt(EDAcsv, delimiter=',')
    cond_data = cond_data[1:]
    sampling_rate = 64 # 64 # for Empatica E4, this is 64, other EDA 1000
    al=sdeconv_analysis_shell(cond_data,sampling_rate)
    al.plot_EDAprepostOPT()
    
    
    