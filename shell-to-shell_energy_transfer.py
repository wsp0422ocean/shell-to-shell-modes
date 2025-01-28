# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 12:14:28 2024

@author: Shengpeng Wang
"""

import numpy as np
from netCDF4 import Dataset
import math
import time
import scipy.io as io
from scipy import signal
import scipy.fft as fft
import os
import cmaps
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib as mpl



# directory containing SWOT level-3 orbit dataset
file_dir=r'l3_karin_nadir/1day_orbit/expert/alpha_v0_3'



reso=0.02
inter_type='linear'

g=9.81     
a=6378.137*1000.0 
omega=7.292e-5 


dt=105


#load the maximum and minimum latitude for the ACC front regions
A=io.loadmat('latdot_swot.mat')
latdot=A['latdot']


#width selected for the shell-to-shell model
sca=1.2



for number in range(1,29): 
    
    if number>28:
        continue
        
    ny=9860
    nx=69

    L=[]
    for root, dirs, files in os.walk(file_dir):
        files.sort()
        break
    for file in files:
        if os.path.splitext(file)[1] == '.nc':
            if(int(file[26+1:28+1])==number):   #should modified based on the length of directory name
                L.append(os.path.join(root, file))

    #load the dx,dy 
    data=io.loadmat(r'dist_angle/SWOT_dist_angle'+str(number)+'.mat')    
    Distx=data['Distx']
    Disty=data['Disty']

    nL=np.size(L)
    ssh_m=np.zeros((nx,ny), float)
    ssh_n=np.zeros((nx,ny), float)
   
    for jj in range(nL):
        file=L[jj]
        with Dataset(file, 'r') as dset:
            ssh0 = dset['ssha_noiseless'][:,:] #L3:ssha_noiseless 
            ssh0=ssh0.T    
            ssh0=ssh0.data   
            ssh0[ssh0<-1e8]=np.nan  
        mask=np.ones((nx,ny), float)
        mask[np.isnan(ssh0)]=0
        ssh_n=ssh_n+mask
        ssh0[np.isnan(ssh0)]=0
        ssh_m=ssh_m+ssh0
        
    ssh_m[ssh_m==0]=np.nan
    ssh_n[ssh_n==0]=np.nan
    ssh_m=ssh_m/ssh_n




    for day in range(473,473+dt+1):
   
        path=r'SWOT_'+str(sca)           
        path=str(path)+'/'+str(number)
        orbitname='noiseless_withsmoothy_nodetrend' #L3:ssha_noiseless L2:sha


        L=[]
        for root, dirs, files in os.walk(file_dir):
            files.sort()
            break
        for file in files:
            if os.path.splitext(file)[1] == '.nc':
                if int(file[21+1:24+1])==day and (int(file[26+1:28+1])==number):
                    L.append(os.path.join(root, file))

        nL=np.size(L)

        if nL<1:
            continue
     
        nL=np.size(L)
        ny=9860
        nx=69
        lat = np.zeros((nx,ny), float)
        lon = np.zeros((nx,ny), float)
        SSH = np.zeros((nx,ny), float)  
        nL=np.size(L)    
        for jj in range(nL):
            file=L[jj]
            
            (file)
            with Dataset(file, 'r') as dset:
                latitude  = dset['latitude'][:,:]
                longitude = dset['longitude'][:,:]
                ssh1 = dset['ssha_noiseless'][:,:]
        
                latitude=latitude.T
                longitude=longitude.T
                ssh1=ssh1.T    
                lat1 =latitude .data
                lon1 =longitude.data
                ssh1=ssh1.data   
                ssh1[ssh1<-1e8]=np.nan   

        ssh1=ssh1-ssh_m
        
        ny=np.size(ssh1,1)
        nx=np.size(ssh1,0)
        
        if(lat1[0,0]>lat1[0,ny-1]):
            ssh1=ssh1[:,::-1]
            lat1=lat1[:,::-1]
            lon1=lon1[:,::-1]
            
        if(lon1[0,0]>lon1[nx-1,0]):
            ssh1=ssh1[::-1,:]
            lat1=lat1[::-1,:]
            lon1=lon1[::-1,:]   
        

        ## smooth data along tracks         
        window_size=3
        window=np.array([0.5,1,0.5])/2           
        ssh0=np.zeros((nx,ny-2),float)*np.nan
        for i in range(nx):
            ssh0[i,:]=np.convolve(ssh1[i,:],window,mode='valid')                

    
    
        lon1[lon1>180]=lon1[lon1>180]-360      
             
        ssh1=ssh0[1:nx-1,:]   
        lat1=lat1[1:nx-1,1:ny-1]
        lon1=lon1[1:nx-1,1:ny-1]    
        
        coslat=np.cos(lat1*np.pi/180)
        coslon=np.cos(lon1*np.pi/180)
        sinlat=np.sin(lat1*np.pi/180)
         
        f=2*omega*sinlat
        
  
        Dy=Disty[1:nx-1,1:ny-1]
        Dx=Distx[1:nx-1,1:ny-1]    

         
        dslady=np.gradient(ssh1,axis=1)/Dy
        dsladx=np.gradient(ssh1,axis=0)/Dx

        #get the geostrophic velocity from the SLA
        ug=-1*(g/f)*dslady
        vg=(g/f)*dsladx       
        
        lat=lat1
        lon=lon1
        ssh=ssh1

        lat_m=lat[34,:]
        
        latmax=latdot[number-1,0]
        latmin=latdot[number-1,1]
            
        
        #slice tracks in the ACC frontal regions
        pos=np.where((lat_m>=latmin)&(lat_m<=latmax))
        pos=pos[0]

        ug=ug[:,pos]
        vg=vg[:,pos]
        lon1=lon[:,pos]
        lat1=lat[:,pos]
        ssh=ssh[:,pos]
        Dx=Dx[:,pos]
        Dy=Dy[:,pos]
        ########################################
        
        ny=np.size(ug,1)
        nx=np.size(ug,0)

        ug1=ug.ravel() 
        #vg1=vg.ravel()
        pos=np.where(abs(ug1)>0)
        pos=pos[0]

        

        ############################Window function###########################################
        Ly=np.nanmean(np.nansum(Dy,1))
        Lx=np.nanmean(np.nansum(Dx,0))
       
        if round(nx/2)*2 == nx:
            Nx = nx/2;
            wn_x = 2*math.pi*np.append(np.arange(0,Nx+1),np.arange(-(Nx-1),-1+1)) / Lx;
        else:
            Nx = (nx-1)/2;
            wn_x = 2*math.pi*np.append(np.arange(0,Nx+1),np.arange(-(Nx),-1+1)) / Lx;      
            
        if round(ny/2)*2 == ny:
            Ny = ny/2;
            wn_y = 2*math.pi*np.append(np.arange(0,Ny+1),np.arange(-(Ny-1),-1+1)) / Ly;
        else:
            Ny = (ny-1)/2;
            wn_y = 2*math.pi*np.append(np.arange(0,Ny+1),np.arange(-(Ny),-1+1)) / Ly;        
        
        [ky,kx] = np.meshgrid(wn_y,wn_x);
        wvsq=ky*ky
        
        dy=Ly/np.size(Dy,1)
        dx=Lx/np.size(Dx,0)
        
        dk = (2*math.pi/Ly);
        KN = 2*math.pi/(2*dy);
        KL = 2*math.pi/(2*Ly);
        step_max = int(KN/dk);          
        
        ny=np.size(ug,1)
        n_width=5
        nn=9
        y=np.arange(1,ny+1,1)
        
        window_dy=np.floor(ny/n_width)
        window_y=np.round(np.linspace(window_dy/2.0,ny-window_dy/2.0-1,nn))\
                -np.round(window_dy/2.0)+1
        window=np.hanning(window_dy)
        
        WINDOW=np.zeros((ny,),float)          
        for m in range(nn):
            A=np.zeros((ny,),float)     
                   
            A[int(window_y[m])-1:int(window_y[m]+np.size(window))-1]=window
            
            WINDOW=WINDOW+A
      
       ############################################################################################
       
       
        ugw=np.zeros(np.shape(ug),float)
        vgw=np.zeros(np.shape(vg),float)
        sshw=np.zeros(np.shape(ssh),float)
        
        for ii in range(nx):
            ug1=ug[ii,:]
            vg1=vg[ii,:]
            ssh1=ssh[ii,:]
            pos=np.where((abs(ug1)>=0)&(abs(vg1)>=0))
            pos=pos[0]
            nn=np.size(pos)         
            
            if nn<ny:
                ugw[ii,:]=np.nan
                vgw[ii,:]=np.nan   
                sshw[ii,:]=np.nan                    
            else:
                ug1[np.isnan(ug1)]=0
                vg1[np.isnan(vg1)]=0
                ug1=ug1-np.nanmean(ug1)
                vg1=vg1-np.nanmean(vg1)
                ssh1=ssh1-np.nanmean(ssh1)
                #detrend
                # ug1 = signal.nodetrend(ug1, axis=-1, type='linear', 
                                # bp=0, overwrite_data=False)        
                # vg1 = signal.nodetrend(vg1, axis=-1, type='linear', 
                                # bp=0, overwrite_data=False)   
                # ssh1 = signal.nodetrend(ssh1, axis=-1, type='linear', 
                                # bp=0, overwrite_data=False)     
                #tapering by using window function                                
                ugw[ii,:]=ug1*WINDOW;
                vgw[ii,:]=vg1*WINDOW;
                sshw[ii,:]=ssh1*WINDOW;
           

    
        ufft=fft.fft(ugw,axis=1)
        vfft=fft.fft(vgw,axis=1)
        uufft=fft.fft(ugw*ugw,axis=1)
        vvfft=fft.fft(vgw*vgw,axis=1)
        uvfft=fft.fft(ugw*vgw,axis=1)               
        sshfft=fft.fft(sshw,axis=1)  

        
        nscale=int(np.log(10000*1e3/1e3)/np.log(sca))
         
        scale=np.zeros((nscale,),float)
        scale[0]=1000
        for i in range(1,nscale):
            scale[i]=scale[i-1]*sca        
    
        Lscale=np.sort(2*math.pi/scale)
    
        scale0=(scale[0:nscale-1]+scale[1:nscale])/2.0
        Lscale0=(Lscale[0:nscale-1]+Lscale[1:nscale])/2.0            
        

        u0_scale=np.zeros((nx,ny,nscale),'float')*np.nan
        v0_scale=np.zeros((nx,ny,nscale),'float')*np.nan
        uu0_scale=np.zeros((nx,ny,nscale),'float')*np.nan
        vv0_scale=np.zeros((nx,ny,nscale),'float')*np.nan
        uv0_scale=np.zeros((nx,ny,nscale),'float')*np.nan
        
    
        
        for step in range(nscale):
   

            k=Lscale[step]
            wvsq0=np.ones(np.shape(wvsq),float)
            wvsq0[wvsq>=k*k]=0
            u_l=fft.ifft(ufft*wvsq0,axis=1).real
            v_l=fft.ifft(vfft*wvsq0,axis=1).real
            uu_l=fft.ifft(uufft*wvsq0,axis=1).real
            vv_l=fft.ifft(vvfft*wvsq0,axis=1).real
            uv_l=fft.ifft(uvfft*wvsq0,axis=1).real

    
            u0_scale[:,:,step]=u_l
            v0_scale[:,:,step]=v_l
            uu0_scale[:,:,step]=uu_l
            vv0_scale[:,:,step]=vv_l
            uv0_scale[:,:,step]=uv_l




        Flux2_2D_=np.zeros((nscale-1,nscale-1,nx),float)*np.nan      

        for kk1 in range(nscale-1):
            # print(kk1,nscale)
            u1=u0_scale[:,:,kk1+1]-u0_scale[:,:,kk1]
            v1=v0_scale[:,:,kk1+1]-v0_scale[:,:,kk1]       
            for kk2 in range(nscale-1):
                u0=u0_scale[:,:,kk2+1]-u0_scale[:,:,kk2]
                v0=v0_scale[:,:,kk2+1]-v0_scale[:,:,kk2]                 
                u_x=np.gradient(u0,axis=0)/Dx
                v_x=np.gradient(v0,axis=0)/Dx
                v_y=np.gradient(v0,axis=1)/Dy
                u_y=np.gradient(u0,axis=1)/Dy           
    
                flux=-(u1*(ugw*u_x+vgw*u_y)+v1*(ugw*v_x+vgw*v_y))   
                flux1=flux.ravel()        
                Flux2_2D_[kk1,kk2,:]=np.nanmean(flux,axis=1)   
                
                
        path=r'SWOT_'+str(sca)
        if(not(os.path.exists(path))):
            os.mkdir(path)
           
        path=str(path)+'/'+str(number)
        if(not(os.path.exists(path))):
            os.mkdir(path)          
        
        io.savemat(path+'/Flux_'+orbitname+'_'+str(day)+'.mat', {'Flux2_2D_x':Flux2_2D_,'Lscale':Lscale,'scale':scale},do_compression=True) 


# average shell-to-shell flux among all tracks 
file_dir=r'SWOT_'+str(sca)+'/'
L=[]
headname='Flux_noiseless_withsmoothy_nodetrend'
for orbit in range(1,28+1): 
    print(orbit)
    for root, dirs, files in os.walk(file_dir+str(orbit)):
        files.sort()
        for file in files:
            if os.path.splitext(file)[1] == ".mat" and os.path.basename(file)[0:len(headname)] == headname:
                L.append(os.path.join(root, file))
    
    
nL=np.size(L)
Flux_2D=[]
for ii in range(nL):
    file=L[ii]
    data=io.loadmat(file)
    Flux2_2D_x=data['Flux2_2D_x'];
    Lscale=data['Lscale'];
    if np.size(Flux2_2D_x)==0:
        continue
    for jj in range(np.size(Flux2_2D_x,2)):
        Flux_2D.append(Flux2_2D_x[:,:,jj])

Flux_2D=np.array(Flux_2D)   
flux_2D=np.nanmean(Flux_2D,axis=0)

ratio=2.0

pos=np.where(1/Lscale/1000>1);
nL=np.size(pos[1])
[L1,L2]=np.meshgrid(Lscale[0,0:nL],Lscale[0,0:nL]);


###################calculte the local and nonlocal cross-scale EKE transfer##################
mask=np.ones((nL,nL),float)
mask[(L1/L2>ratio)|(L1/L2<1/ratio)]=0    
fluxx11=np.zeros((nL-1,),float)
for k in range(nL-1):
    fluxx11[k]=np.nansum(flux_2D[k:nL,0:k]*mask[k:nL,0:k])    

mask=np.ones((nL,nL),float)
mask[(L1/L2<=ratio)&(L1/L2>=1/ratio)]=0    
fluxx12=np.zeros((nL-1,),float)
for k in range(nL-1):
    fluxx12[k]=np.nansum(flux_2D[k:nL,0:k]*mask[k:nL,0:k])    
############################################################################################

Lscale1=Lscale/(2*math.pi) 
Lscale1=Lscale1[0,:]
nscale=np.size(flux_2D,0)
Lscale10=(Lscale1[0:nscale-1]+Lscale1[1:nscale])/2.0


############################Show the result################################################
fig=plt.figure(figsize=(16,8))

ax1 = fig.add_axes([0.00, 0.1, 0.45, 0.9])

nscale=np.size(Lscale10)
x1=Lscale1
cmap1=cmaps.temp_diff_18lev      
 
blim=0.4
c1=ax1.pcolor(Lscale10*1000,Lscale10*1000,flux_2D[0:nscale,0:nscale]*1e9,cmap=cmap1,vmin=-blim,vmax=blim) 
plt.plot(x1*1000,x1*1000/ratio,linewidth=2,color='g',linestyle='--')
plt.plot(x1*1000,x1*1000*ratio,linewidth=2,color='g',linestyle='--')
plt.plot(x1*1000,x1*1000,linewidth=2,color='k',linestyle='--')
cbar=plt.colorbar(c1)
cbar.set_label('Unit:10$^{-6}$ W/m$^3$')
plt.xscale('log')
plt.yscale('log')
plt.xlim(x1[0]*1000,x1[-1]*1000)
plt.ylim(x1[0]*1000,x1[-1]*1000)
label_fontdict = {
    'fontsize': 16,
    'fontweight': 'bold',
    'color': 'black',
    'verticalalignment': 'baseline',
    'horizontalalignment': 'center' 
    }
plt.title(label='Shell-to-shell EKE transfer (SWOT)',fontdict=label_fontdict)
plt.text(5e-3/(2*math.pi) ,2e-1/(2*math.pi) ,'Nonlocal Transfer',fontsize=12,rotation=45,fontweight='bold')
plt.text(2e-1/(2*math.pi) ,3e-1/(2*math.pi) ,'Local Transfer',fontsize=12,rotation=45,fontweight='bold')

font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}
plt.xlabel('q (cpkm)',font2,)
plt.ylabel('k (cpkm)',font2,)
ax1.set_aspect("equal") 
ax1.set_xlim(3e-4,3e-1)
ax1.set_ylim(3e-4,3e-1)




ax1 = fig.add_axes([0.5, 0.2, 0.45, 0.7])

ax1.plot(Lscale1[0:nL-1]*1000,fluxx11.T*1000*1e6,linewidth=3,color='r',label='Local transfer')
ax1.plot(Lscale1[0:nL-1]*1000,fluxx12.T*1000*1e6,linewidth=3,color='b',label='Nonlocal transfer')

ax1.plot([1e-4,1e0],[0,0],linewidth=1,color='black',linestyle='--')
plt.xscale('log')

plt.legend(fontsize=16)
label_fontdict = {
    'fontsize': 16,
    'fontweight': 'bold',
    'color': 'black',
    'verticalalignment': 'baseline',
    'horizontalalignment': 'center' 
    }
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}

plt.xlabel('Wavenumber (cpkm)',font2,)
plt.ylabel('Transfer rate (10$^{-6}$ W/m$^3$)',font2,)
ax1.set_title('Cross-scale EKE transfer',fontdict=label_fontdict)

ax1.set_xlim(1e-4,1e0)
ylim=ax1.set_ylim()
xlim=ax1.set_xlim()

ax2=ax1.twiny()
plt.xscale('log')
ax2.tick_params(labelbottom=False, labeltop=True)
ax2.set_xlim(xlim)
xtick=[1/1500,1/600,1/350,1/200,1/100,1/50,1/30,1/10]
top_label=['1500 km','600 km','350 km','200 km','100 km','50 km','30 km','10 km']

ax2.set_xticks(xtick)
ax2.set_xticklabels(top_label)
plt.xticks(rotation=45)
plt.grid()


plt.show()

