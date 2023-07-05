import uproot
import matplotlib.pyplot as plt
import vector
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import seaborn as sns
import matplotx
import math as math
from IPython.core.pylabtools import figsize
import sys, os, glob, pathlib
import numpy as np
import h5py as h5
import glob
import os
#import ROOT as rt # used for plotting
import matplotx
from IPython.core.pylabtools import figsize

class JetPlotter:
    exten = '.png'
    def __init__(self, file,):
        self.ofile = uproot.open(file)
        self.lctuple = self.ofile['MyLCTuple']
        self.tree = self.ofile['MyLCTuple;1']
        self.branches = self.tree.arrays()
        self.px_branch = self.branches['jmox']
        self.py_branch = self.branches['jmoy']
        self.pz_branch = self.branches['jmoz']
        self.p = self.branches['jmom']
        self.E_branch  = self.branches['jene']
        self.nevents = len(self.branches['evene']) 
        self.jet_4p = vector.zip({'px': self.px_branch, 'py': self.py_branch, 'pz': self.pz_branch, 'E': self.E_branch})
        self.file_title = file
    

    def label_format(self):
        tokens = self.file_title.split("_")
        if "bib" in tokens:
            return "BIB 10%"
        else:
            return "BIB 0%"
        
    def file_name(self): 
        check =  self.file_title.split("_")
        if "bib" in check:
                sim_type = "bib/"
        else:
            sim_type = "nbib/"
        tokens = self.file_title.split("/")
        folder_name = tokens[0]
        new_name = "{0}{1}{2}".format(folder_name,'/plots/',sim_type)
        return new_name

    def saver(self, plot_type):
        plt.savefig(self.file_name()+plot_type+self.exten)
    
    # Determing an array of max jet momentum  
    def maxj_p(self):
        caller_name = self.maxj_p.__name__
        jet_p_max = np.arange(self.nevents)
        for i in range(0, self.nevents):
            jet_p_max[i]=  max(self.p[i])
        return jet_p_max, caller_name

    # Determing an array of invariant mass of all jets per event
    def invmass_jets(self): 
        caller_name = self.invmass_jets.__name__
        j_invmass = np.empty(len(self.jet_4p), np.float64)
        for i, event in enumerate(self.jet_4p):
            total = vector.obj(px=0.0, py=0.0, pz=0.0, E=0.0)
            for jet in event:
                total = total + jet
            j_invmass[i] = total.mass
        return j_invmass, caller_name
    
    # Determining an array of invariant mass of two leading jets per event
    def invmass_lead_jets(self):
        caller_name = self.invmass_lead_jets.__name__
        j_invmass = np.empty(len(self.jet_4p), np.float64)
        # momentum filter 
        jet_pmasked = ak.sort(self.jet_4p.p, ascending=False)[:,0:2]
        for i, event in enumerate(self.jet_4p):
            total = vector.obj(px=0.0, py=0.0, pz=0.0, E=0.0)
            for jet in event:
                p = math.trunc(jet.p)
                if p == math.trunc(int(jet_pmasked[i,0])) or p == math.trunc(int(jet_pmasked[i,1])):
                    total = total + jet
                else:
                    continue
            j_invmass[i] = total.mass
        return j_invmass, caller_name
    
    # Determining an array of pt values of jets per event 
    def pt_jet(self):
        caller_name = self.pt_jet.__name__
        bg_token =  self.label_format()
        pt_j_store = np.empty(len(self.jet_4p), np.float64)
        pt_j = self.jet_4p.pt
        pt_j_flatten = ak.to_numpy(ak.flatten(pt_j))
        return pt_j_flatten, caller_name, bg_token
    
    # Determining an array of theta values of jets per event 
    def theta_jet(self):
        caller_name = self.theta_jet.__name__
        bg_token = self.label_format()
        theta_j = self.jet_4p.theta
        theta_j_flatten= ak.to_numpy(ak.flatten(theta_j))
        return theta_j_flatten, caller_name, bg_token
        
    
    # Plotter of invariant mass of all jets per event
    def invmj_plotter(self, showlegend = True):
        plt.hist(self.invmass_jets()[0],  bins =40, range=(0, 1000), alpha =0.6, label = self.label_format()+ '\n$\overline{{m}}$ = {0} \n$\sigma_{{m}}$ = {1}'.format(round(np.mean(self.inmass_jets()[0]),2),round(np.std(self.invmass_jets()[0]),2)))
        plt.legend(loc='upper right')
        plt.xticks(np.arange(0, 1000, 100))
        plt.yticks(np.arange(0, 60, 10))
        plt.xlabel("mass [Gev]")
        plt.ylabel('Number of events')
        plt.title('Invariant $m$ distribution jets, 0 MeV threshold')
      
    
    # Plotter of invariant mass of two leading jets per event
    def invm2j_plotter(self, showlegend = True):
        plt.hist(self.invmass_lead_jets()[0],  bins =40, range=(0, 1000), alpha =0.6, label = self.label_format()+ '\n$\overline{{m}}$ = {0} \n$\sigma_{{m}}$ = {1}'.format(round(np.mean(self.invmass_lead_jets()[0]),2),round(np.std(self.invmass_lead_jets()[0]),2)))
     
        plt.legend(loc='upper right')
        plt.xticks(np.arange(0, 1000, 100))
        plt.yticks(np.arange(0, 60, 10))
        plt.xlabel("mass [Gev]")
        plt.ylabel('Number of events')
        plt.title('Invariant $m$ distribution for two leading jets, 0 MeV threshold')
        

    # Plotter of max jet per event
    def maxj_p_plotter(self):
        
        plt.hist(self.maxj_p()[0], bins =30, range=(0,850), alpha =0.5, label = self.label_format()+'\n$\overline{{m}}$ = {0} \n$\sigma_{{m}}$ = {1}'.format(round(np.mean(self.maxj_p()[0]),2),  round(np.std(self.maxj_p()[0]),2)))                                                                   
        plt.legend(loc='upper right')
        plt.xticks(np.arange(0, 850, 100))
        plt.yticks(np.arange(0, 31, 5))
        plt.xlabel("$p_{\mathrm{J}}$ [GeV]")
        plt.ylabel('Number of events')
        plt.title('Max Jet momentum per event, 2 MeV threshold')
        #plt.savefig(self.file_name()+"invmass_all_jets"+self.exten)
        
    #To-Do's : IMPLEMENT the saving function for the plots
    @staticmethod
    def overlayed2h(data1, data2):
        n_rows = 1
        n_cols = 1
        fig, ax = plt.subplots() #move labels into conditionals
        fig.set_size_inches((12, 8))
        label_1 = 'BIB 10% \n$\overline{{m}}$ = {0} \n$\sigma_{{m}}$ = {1}'.format(round(np.mean(data1[0]),2),round(np.std(data1[0]),2))
        label_2 = 'BIB 0% \n$\overline{{m}}$ = {0} \n$\sigma_{{m}}$ = {1}'.format(round(np.mean(data2[0]),2),round(np.std(data2[0]),2))
        ax.hist(data1[0], bins=50, range=(0,1000), alpha=0.6, label=label_1)
        ax.hist(data2[0], bins=50, range=(0,1000), alpha=0.6, label=label_2)  
        ax.legend(loc='upper right')
        if data1[1]=="invmass_lead_jets":
            ax.set_xlabel(r"$m^{H \rightarrow \mu \mu +bb}_{2l2q}$" + "[Gev]", loc="right", fontsize = 18)
            ax.set_ylabel('Number of events',fontsize = 18)
            ax.set_title('Invariant $m$ distribution for two leading jets, 2 MeV threshold')
            ax.set_xticks(np.arange(0, 1100, 100))
            ax.set_yticks(np.arange(0, 70, 10))
            ax.set_xticks(np.arange(0, 990, 10), minor=True)
            ax.set_yticks(np.arange(0, 59, 1), minor=True)
            ax.text(50, 55, "Muon Collider", weight="bold", fontsize=18, fontstyle ='italic')
            ax.text(240, 55, "Simulation Data")
            ax.text(50, 52, "TRIUMF Collaboration")
        elif data1[1]=="maxj_p":
            ax.set_xticks(np.arange(0, 1100, 100))
            ax.set_yticks(np.arange(0, 31, 5))
            ax.set_xlabel("$p_{\mathrm{J}}$ [GeV]")
            ax.set_ylabel('Number of events')
            ax.set_title('Max Jet momentum per event, 2 MeV threshold')
            ax.text(29, 28, "Muon Collider", weight="bold", fontsize=18, fontstyle ='italic')
            ax.text(210, 28, "Simulation Data")
            ax.text(29, 27, "TRIUMF Collaboration")
        elif data1[1]=="theta_j":
            ax.set_xticks(np.arange(0, 3.5,  0.5))
            ax.set_yticks(np.arange(0, 310, 20))
            ax.set_xlabel("$\Theta$ [rad]")
            ax.set_ylabel('Number of events')
            ax.set_title('$\Theta$ distribution, with BIB, 0 MeV threshold')
        
        ax.tick_params(which='both', direction='in', top=True, right=True)
              
     
    # 2D histogram plotter
    @staticmethod
    def histo2d(data1, data2):
        fig, ax = plt.subplots() 
        fig.set_size_inches((10, 8))#move labels into conditionals 
        if (data1[1]=="pt_jet" and data2[1]=="theta_jet"):
            range1= [[0,200], [0,3.5]]
            hist  = ax.hist2d(data1[0], data2[0], bins=(50, 50), range =range1 , cmap = "RdYlGn_r")
            ax.set_xlabel('$p_{T} [GeV]$', fontsize=18)
            ax.set_ylabel("$\Theta$ [rad]",fontsize=18)
            ax.set_xticks(np.arange(0, 220,  20)) # add conditional fro the title 
            if (data1[2] == 'BIB 10%'):
                ax.set_title('$p_{T}$ vs $\Theta$ of a jet, BIB 10%', fontsize= 18)
            else: ax.set_title('$p_{T}$ vs $\Theta$ of a jet, BIB 0%, 2 MeV threshol', fontsize= 18)
            ax.text(15, 3.3, "Muon Collider", weight="bold", fontsize=18, fontstyle ='italic')
            ax.text(70, 3.3, "Simulation Data")
            ax.text(15, 3.1, "TRIUMF Collaboration")
            cbar3= plt.colorbar(hist[3])
            cbar3.set_label('Number of jets', rotation=270, loc ='center',labelpad= 15)
            
     