import uproot
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
from matplotlib import colors


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
        self.njets = self.branches['nj']
        self.pfopx_branch = self.branches['rcmox']
        self.pfopy_branch = self.branches['rcmoy']
        self.pfopz_branch = self.branches['rcmoz']
        self.pfoE_branch = self.branches['rcene']
        self.npfojets = self.branches['nrec']
        self.jet_4p = vector.zip({'px': self.px_branch, 'py': self.py_branch, 'pz': self.pz_branch, 'E': self.E_branch})
        self.pfos_4p =  vector.zip({'px': self.pfopx_branch, 'py': self.pfopy_branch, 'pz': self.pfopz_branch, 'E': self.pfoE_branch})
        self.file_title = file
    
    # Return label for BIB/no BIB to be inserted in the title 
    def label_format(self):
        tokens = self.file_title.split("_")
        if "bib" in tokens:
            return "BIB 10%"
        else:
            return "BIB 0%"
    # Asembles the name of the plot from the constituents of the file name     
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
        jet_pmasked = ak.sort(self.jet_4p.pt, ascending=False)[:,0:2]
        for i, event in enumerate(self.jet_4p):
            total = vector.obj(px=0.0, py=0.0, pz=0.0, E=0.0)
            for jet in event:
                p = math.trunc(jet.pt)
                if p == math.trunc(int(jet_pmasked[i,0])) or p == math.trunc(int(jet_pmasked[i,1])):
                    total = total + jet
                else:
                    continue
            j_invmass[i] = total.mass
        return j_invmass, caller_name
    
    
    def invmass_lead_pfojets(self):
        caller_name = self.invmass_lead_pfojets.__name__
        pfoj_invmass = np.empty(len(self.pfos_4p), np.float64)
        # momentum filter 
        jet_pmasked = ak.sort(self.pfos_4p.pt, ascending=False)[:,0:2]
        for i, event in enumerate(self.pfos_4p):
            total = vector.obj(px=0.0, py=0.0, pz=0.0, E=0.0)
            for jet in event:
                p = math.trunc(jet.pt)
                if p == math.trunc(int(jet_pmasked[i,0])) or p == math.trunc(int(jet_pmasked[i,1])):
                    total = total + jet
                else:
                    continue
            pfoj_invmass[i] = total.mass
        return pfoj_invmass, caller_name
    
    # Determining an array of pt values of jets per event 
    def pt_jet(self):
        caller_name = self.pt_jet.__name__
        bg_token =  self.label_format()
        pt_j_store = np.empty(len(self.jet_4p), np.float64)
        pt_j = self.jet_4p.pt
        pt_j_flatten = ak.to_numpy(ak.flatten(pt_j))
        return pt_j_flatten, caller_name, bg_token
    
    def pt_pfojet(self):
        caller_name = self.pt_pfojet.__name__
        bg_token =  self.label_format()
        pt_pfoj_store = np.empty(len(self.pfos_4p), np.float64)
        pt_pfoj = self.pfos_4p.pt
        pt_pfoj_flatten = ak.to_numpy(ak.flatten(pt_pfoj))
        return pt_pfoj_flatten, caller_name, bg_token
    
    
    def pt_lead_jet(self):
        caller_name = self.pt_lead_jet.__name__
        j_pt_lead = np.empty(len(self.jet_4p), np.float64)
        # momentum filter 
        jet_pmasked = ak.sort(self.jet_4p.p, ascending=False)[:,0:2]
        for i, event in enumerate(self.jet_4p):
            total = vector.obj(px=0.0, py=0.0, pz=0.0, E=0.0)
            for jet in event:
                p = math.trunc(jet.p)
                if p == math.trunc(int(jet_pmasked[i,0])) or p == math.trunc(int(jet_pmasked[i,1])):
                     j_pt_lead[i] = jet.pt
                else:
                    continue
        return j_pt_lead, caller_name
    
    # Determining an array of theta values of jets per event 
    def theta_jet(self):
        caller_name = self.theta_jet.__name__
        bg_token = self.label_format()
        theta_j = self.jet_4p.theta
        theta_j_flatten= ak.to_numpy(ak.flatten(theta_j))
        return theta_j_flatten, caller_name, bg_token
    
    def eta_jet(self):
        caller_name = self.eta_jet.__name__
        bg_token = self.label_format()
        eta_j = self.jet_4p.eta
        eta_j_flatten= ak.to_numpy(ak.flatten(eta_j))
        return eta_j_flatten, caller_name, bg_token
    
    def phi_jet(self):
        caller_name = self.phi_jet.__name__
        bg_token = self.label_format()
        phi_j = self.jet_4p.phi
        phi_j_flatten= ak.to_numpy(ak.flatten(phi_j))
        return phi_j_flatten, caller_name, bg_token
    
    
    def theta_pfojet(self):
        caller_name = self.theta_pfojet.__name__
        bg_token = self.label_format()
        theta_pfoj = self.pfos_4p.theta
        theta_pfoj_flatten= ak.to_numpy(ak.flatten(theta_pfoj))
        return theta_pfoj_flatten, caller_name, bg_token
    
    def eta_pfojet(self):
        caller_name = self.eta_pfojet.__name__
        bg_token = self.label_format()
        eta_pfoj = self.pfos_4p.eta
        eta_pfoj_flatten= ak.to_numpy(ak.flatten(eta_pfoj))
        return eta_pfoj_flatten, caller_name, bg_token
        
    def phi_pfojet(self):
        caller_name = self.phi_pfojet.__name__
        bg_token = self.label_format()
        phi_pfoj = self.pfos_4p.phi
        phi_pfoj_flatten= ak.to_numpy(ak.flatten(phi_pfoj))
        return phi_pfoj_flatten, caller_name, bg_token
    
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
    
    
    # Plotter of pt for jets per event
    def pt_jet_plotter(self, thres = None, range_histo = (0,100), x_ticks =np.arange(0,110, 10), y_ticks = np.arange(0, 110, 10), x_1 = 5, y_1 = 90, x_2 = 5, y_2 = 85, x_3=30, y_3 =85, x_4 =5, y_4 =80, x_5=5, y_5=75, method=None, a = None):
        N = len(self.jet_4p)
        mean_pt_jets = np.mean(self.pt_jet()[0])
        std_pt_jets = np.std(self.pt_jet()[0])
        fig, ax = plt.subplots()
        ax.hist(self.pt_jet()[0], bins=50, range=range_histo, alpha = 0.5, color = 'purple', label = f'{JetPlotter.legend_label(thres)}\n{self.pt_jet()[2]}\n$\overline{{p_{{J}}}}$ = {str(round(np.mean(self.pt_jet()[0]),2))} \n$\sigma_{{p_{{J}}}}$ = {str(round(np.std(self.pt_jet()[0]),2))}\n$N_{{events}}$ = {N}')
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)       
#plt.title('Momentum of a jet, $p_{J}$ BIB, 0 MeV threshold,(flattened)',fontsize = 13, fontweight ='bold')
        ax.set_xlabel('$p_{\mathrm{t}}$ [GeV]')
        ax.set_ylabel('Number of Jets')
        ax.legend(loc='upper right')
        ax.text(x_1, y_1, "Muon Collider", weight="bold", fontsize=18, fontstyle ='italic')
        ax.text(x_2, y_2, "Simulation Data")
        ax.text(x_3, y_3, "$\sqrt{s} = 3 TeV$")
        if method == 'sk':
            ax.text(x_4, y_4, f'SoftKiller(a={a})')
        else: 
            ax.text(x_4, y_4, "Calo Baseline Cut")
        ax.text(x_5, y_5, "TRIUMF Collaboration")
    
    def pt_pfojet_plotter(self, thres = None, range_histo = (0,100), x_1 = 5, y_1 = 90, x_2 = 5, y_2 = 85, x_3=30, y_3 =85, x_4 =5, y_4 =80, x_5=5, y_5=75, e = 1.5):
        N = len(self.pfos_4p)
        mean_pt_jets = np.mean(self.pt_pfojet()[0])
        std_pt_jets = np.std(self.pt_pfojet()[0])
        fig, ax = plt.subplots()
        ax.hist(self.pt_pfojet()[0], bins=50, range=range_histo, alpha = 0.5, color = 'purple', label = f'{JetPlotter.legend_label(thres)}\n{self.pt_pfojet()[2]}\n$\overline{{p_{{pfos}}}}$ = {str(round(np.mean(self.pt_pfojet()[0]),2))} \n$\sigma_{{p_{{pfos}}}}$ = {str(round(np.std(self.pt_pfojet()[0]),2))}\n$N_{{events}}$ = {N}')
          
#plt.title('Momentum of a jet, $p_{J}$ BIB, 0 MeV threshold,(flattened)',fontsize = 13, fontweight ='bold')
        ax.set_xlabel('$p_{\mathrm{t}}$ [GeV]')
        ax.set_ylabel('Number of PFOs')
        ax.legend(loc='upper right')
        ax.text(x_1, y_1, "Muon Collider", weight="bold", fontsize=18, fontstyle ='italic')
        ax.text(x_2, y_2, "Simulation Data")
        ax.text(x_3, y_3, f'$\sqrt{{s}} = {e} TeV$')
        ax.text(x_4, y_4, "PFOs")
        ax.text(x_5, y_5, "TRIUMF Collaboration")
    
    
    #Plotter of pt of two leading jets per event 
    def pt_2jet_plotter(self, thres = None, range_histo = (0,100), x_ticks =np.arange(0,100, 10), y_ticks = np.arange(0, 31, 5), x_1 = 5, y_1 = 40, x_2 = 5, y_2 = 35, x_3=30, y_3 =35, x_4 =150, y_4 =24, x_5=5, y_5=30, method=None, a=None):
        N = len(self.jet_4p)
        mean_pt_jets = np.mean(self.pt_lead_jet()[0])
        std_pt_jets = np.std(self.pt_lead_jet()[0])
        fig, ax = plt.subplots()
        ax.hist(self.pt_lead_jet()[0], bins=50, range=range_histo, alpha = 0.5, color = 'purple', label = f'{JetPlotter.legend_label(thres)}\n$\overline{{p_{{J}}}}$ = {round(mean_pt_jets,2)}\n$\sigma_{{p_{{J}}}}$ = {round(std_pt_jets,2)}\nN = {N}')
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
                        
       # ax.set_title('$p_{\mathrm{t}}$ of two leading jets, (flattened)',fontsize = 13, fontweight ='bold')
        ax.set_xlabel('$p_{\mathrm{t}}$ [GeV]')
        ax.set_ylabel('Number of Jets')
        ax.legend(loc='upper right')
        ax.text(x_1, y_1, "Muon Collider", weight="bold", fontsize=18, fontstyle ='italic')
        ax.text(x_2, y_2, "Simulation Data")
        ax.text(x_3, y_3, "$\sqrt{s} = 3 TeV$")
        if method == 'sk':
            ax.text(x_4, y_4, f'SoftKiller(a={a})')
        else: 
            ax.text(x_4, y_4, "Calo Baseline Cut")
        ax.text(x_5, y_5, "TRIUMF Collaboration")
        
           
    

    # Plotter of max jet per event
    def maxj_p_plotter(self, thres = None, range_histo = (0,850), x_ticks =np.arange(0,850, 100), y_ticks = np.arange(0, 31, 5), x_1 = 150, y_1 = 28, x_2 = 150, y_2 = 26, x_3=400, y_3 =26, x_4 =150, y_4 =24, x_5=150, y_5=22, method=None, a=None):
        N = len(self.maxj_p()[0])
        fig, ax = plt.subplots()
        ax.hist(self.maxj_p()[0], bins =30, range=range_histo, alpha =0.5, label=f'{JetPlotter.legend_label(thres)}\n {self.label_format()}\n$\overline{{p_{{max}}}}$ = {round(np.mean(self.maxj_p()[0]),2)} \n$\sigma_{{p_{{max}}}}$ = {round(np.std(self.maxj_p()[0]),2)} \n$N_{{events}}$= {N}')
        
        ax.legend(loc='upper right')
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xlabel("$p_{\mathrm{J}}$ [GeV]")
        ax.set_ylabel('Number of events')
        ax.text(x_1, y_1, "Muon Collider", weight="bold", fontsize=18, fontstyle ='italic')
        ax.text(x_2, y_2, "Simulation Data")
        ax.text(x_3, y_3, "$\sqrt{s} = 3 TeV$")
        if method == 'sk':
            ax.text(x_4, y_4, f'SoftKiller(a={a})')
        else: 
            ax.text(x_4, y_4, "Calo Baseline Cut")
        ax.text(x_5, y_5, "TRIUMF Collaboration")
    
        #ax.set_title('Max Jet momentum per event, 2 MeV threshold', fontsize = 10)
        #plt.savefig(self.file_name()+"invmass_all_jets"+self.exten)
        
    def num_jets_plotter(self, thres = None, range_histo = (0,50), x_ticks =np.arange(0,60, 5), y_ticks = np.arange(0, 110, 10), x_1 = 5, y_1 = 90, x_2 = 5, y_2 = 85, x_3=19, y_3 =85, x_4 =5, y_4 =80, x_5=5, y_5=75, method = None, a=None):
        N = len(self.njets)
        mean_num_jets = np.mean(self.njets)
        std_num_jets = np.std(self.njets)
        
        fig, ax = plt.subplots()
        ax.hist(self.njets, bins=50, range=range_histo, alpha = 0.5, color = 'purple', label = f'{JetPlotter.legend_label(thres)}\n {self.label_format()}\n$\overline{{n_{{J}}}}$ = {round(mean_num_jets,2)} \n$\sigma_{{n_{{J}}}}$ = {round(std_num_jets,2)} \n$N_{{events}}$= {N}')
        ax.set_xticks(np.arange(0, 60,  5))
        ax.set_yticks(np.arange(0, 110, 10))
                        
#plt.title('Momentum of a jet, $p_{J}$ BIB, 0 MeV threshold,(flattened)',fontsize = 13, fontweight ='bold')
        ax.set_xlabel('Number of Jets')
        ax.set_ylabel('Frequency of Jets per event')
        ax.legend(loc='upper right')
        ax.text(x_1, y_1, "Muon Collider", weight="bold", fontsize=18, fontstyle ='italic')
        ax.text(x_2, y_2, "Simulation Data")
        ax.text(x_3, y_3, "$\sqrt{s} = 3 TeV$")
        if method == 'sk':
            ax.text(x_4, y_4, f'SoftKiller(a={a})')
        else: 
            ax.text(x_4, y_4, "Calo Baseline Cut")
        ax.text(x_5, y_5, "TRIUMF Collaboration")
        
        
        
    def num_pfojets_plotter(self, thres = None, range_histo = (0,10000), x_1 = 5, y_1 = 90, x_2 = 5, y_2 = 85, x_3=19, y_3 =85, x_4 =5, y_4 =80, x_5=5, y_5=75):
        N = len(self.npfojets)
        mean_num_jets = np.mean(self.npfojets)
        std_num_jets = np.std(self.npfojets)
        
        fig, ax = plt.subplots()
        ax.hist(self.npfojets, bins=50, range=range_histo, alpha = 0.5, color = 'purple', label = f'{JetPlotter.legend_label(thres)}\n {self.label_format()}\n$\overline{{n_{{pfos}}}}$ = {round(mean_num_jets,2)} \n$\sigma_{{n_{{pfos}}}}$ = {round(std_num_jets,2)} \n$N_{{events}}$= {N}')
      
                        
#plt.title('Momentum of a jet, $p_{J}$ BIB, 0 MeV threshold,(flattened)',fontsize = 13, fontweight ='bold')
        ax.set_xlabel('Number of PFOs')
        ax.set_ylabel('Frequency of PFOs per event')
        ax.legend(loc='upper right')
        ax.text(x_1, y_1, "Muon Collider", weight="bold", fontsize=18, fontstyle ='italic')
        ax.text(x_2, y_2, "Simulation Data")
        ax.text(x_3, y_3, "$\sqrt{s} = 1.5 TeV$")
        ax.text(x_4, y_4, f'PFOs')
        ax.text(x_5, y_5, "TRIUMF Collaboration")
      
         
    @staticmethod  
    def legend_label(x):
        if x == "2 MeV":
            return "$E_{{calo\;thres}}$ = " + x
        elif x =="O Mev": 
            return "$E_{{calo\;thres}}$ = " + x
        else:
            return x
        
    
    #To-Do's : IMPLEMENT the saving function for the plots
    @staticmethod
    def overlayed2h(data1, data2,  thres = None, range_histo = (0,1000), x_ticks =np.arange(0,1100, 100), y_ticks = np.arange(0, 70, 10), x_1 = 50, y_1 = 55, x_2 = 50, y_2 = 52, x_3=200, y_3 =52, x_4 =50, y_4 =49, x_5=50, y_5=46, energy = '1.5',  method = None, a = None):
        n_rows = 1
        n_cols = 1
        fig, ax = plt.subplots() #move labels into conditionals
        fig.set_size_inches((12, 8))
        label_1 = '{0}\nBIB 10% \n$\overline{{m}}$ = {1} \n$\sigma_{{m}}$ = {2}'.format(JetPlotter.legend_label(thres),round(np.mean(data1[0]),2),round(np.std(data1[0]),2))
        label_2 = '\nBIB 0% \n$\overline{{m}}$ = {0} \n$\sigma_{{m}}$ = {1}'.format(round(np.mean(data2[0]),2),round(np.std(data2[0]),2))
        ax.hist(data1[0], bins=50, range=range_histo, alpha=0.6, label=label_1)
        ax.hist(data2[0], bins=50, range=range_histo, alpha=0.6, label=label_2)  
        ax.legend(loc='upper right', fontsize = 18)
        if data1[1]=="invmass_lead_jets":
            ax.set_xlabel(r"$m_{2j}$" + "[Gev]", loc="right", fontsize = 18)
            ax.set_ylabel('Number of events',fontsize = 18)
            ax.set_title('Invariant $m$ distribution for two leading jets')
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
           # ax.set_xticks(np.arange(0, 990, 10), minor=True)
          #  ax.set_yticks(np.arange(0, 59, 1), minor=True)
            ax.text(x_1, y_1, "Muon Collider", weight="bold", fontsize=18, fontstyle ='italic')
            ax.text(x_2, y_2, "Simulation Data")
            ax.text(x_3, y_3, f"$\sqrt{{s}}={energy} TeV$")
            if method == 'sk':
                ax.text(x_4, y_4, f'SoftKiller(a={a})')
            else: 
                ax.text(x_4, y_4, "Calo Baseline Cut")
            ax.text(x_5, y_5, "TRIUMF Collaboration")
        elif data1[1]=="maxj_p":
            ax.set_xticks(np.arange(0, 1100, 100))
            ax.set_yticks(np.arange(0, 31, 5))
            ax.set_xlabel("$p_{\mathrm{J}}$ [GeV]",fontsize = 18)
            ax.set_ylabel('Number of events',fontsize = 18)
            ax.set_title('Max Jet momentum per event')
            ax.text(x_1, y_1, "Muon Collider", weight="bold", fontsize=18, fontstyle ='italic')
            ax.text(x_2, y_2, "Simulation Data")
            ax.text(x_3, y_3, "$\sqrt{s} = 3 TeV$")
            if method == 'sk':
                ax.text(x_4, y_4, f'SoftKiller(a={a})')
            else: 
                ax.text(x_4, y_4, "Calo Baseline Cut")
            ax.text(x_5, y_5, "TRIUMF Collaboration")
        elif data1[1]=="theta_j":
            ax.set_xticks(np.arange(0, 3.5,  0.5))
            ax.set_yticks(np.arange(0, 310, 20))
            ax.set_xlabel("$\Theta$ [rad]")
            ax.set_ylabel('Number of events')
            ax.set_title('$\Theta$ distribution, with BIB')
        
        ax.tick_params(which='both', direction='in', top=True, right=True)
              
     
    # 2D histogram plotter
    @staticmethod
    def histo2d(data1, data2, thres=None, range_histo = [[0,200], [0,3.5]], x_ticks =np.arange(0,220, 20), y_ticks = np.arange(0, 70, 10),x_0=140, y_0=3.1,  x_1 = 15, y_1 = 3.3, x_2 = 80, y_2 = 3.3, x_3=140, y_3 =3.3, x_4 =80, y_4 =3.1, x_5=15, y_5=3.1, method = None, a=None, energy = '1.5'):
        fig, ax = plt.subplots() 
        fig.set_size_inches((10, 8))#move labels into conditionals
        if (data1[1]=="pt_jet" and data2[1]=="theta_jet"):
            hist  = ax.hist2d(data1[0], data2[0], norm = colors.LogNorm(), bins=(50, 50), range =range_histo, cmap=plt.cm.Reds)
            ax.set_xlabel('$p_{T} [GeV]$', fontsize=18)
            ax.set_ylabel("$\Theta$ [rad]",fontsize=18)
           # ax.set_xticks(np.arange(0, 220,  20)) # add conditional fro the title 
            ax.text(x_0, y_0, f'{JetPlotter.legend_label(thres)}')
            ax.text(x_1, y_1, "Muon Collider", weight="bold", fontsize=18, fontstyle ='italic')
            ax.text(x_2, y_2, "Simulation Data")
            ax.text(x_3, y_3, f"$\sqrt{{s}}={energy} TeV$")
            ax.text(2, 3.18, "$\Theta = \pi$", fontsize=18,color='b', ha='center')
            ax.text(2, 0, '$\Theta = 0$', fontsize=18,color='b',ha='center')
            if method == 'sk':
                ax.text(x_4, y_4, f'SoftKiller(a={a})')
            if method =='calo': 
                ax.text(x_4, y_4, "Calo Baseline Cut")
            if method == None:
                ax.text(x_4, y_4, f'No Selection')
            ax.text(x_5, y_5, "TRIUMF Collaboration")
            if (data1[2] == 'BIB 10%'):
                #ax.set_title('$p_{T}$ vs $\Theta$ of a jet, BIB 10%', fontsize= 18)
                text = "{}\n BIB 10%".format(JetPlotter.legend_label(thres))
            else:
                text = "{}\n BIB 10%".format(JetPlotter.legend_label(thres))
                #ax.set_title('$p_{T}$ vs $\Theta$ of a jet, BIB 0%, 2 MeV threshold', fontsize= 18)
            ax.text = (100, 3.3, text)
            cbar3= plt.colorbar(hist[3])
            cbar3.set_label('Number of jets', rotation=270, loc ='center',labelpad= 15)
            plt.axhline(y = 0, color = 'b', linestyle = '--')
            plt.axhline(y = np.round(np.pi, 5), color = 'b', linestyle = '--')
            
            
           
            
        if (data1[1]=="pt_pfojet" and data2[1]=="theta_pfojet"):
            hist  = ax.hist2d(data1[0], data2[0], norm = colors.LogNorm(), bins=(50, 50), range =range_histo, cmap=plt.cm.Reds)
            ax.set_xlabel('$p_{T} [GeV]$', fontsize=18)
            ax.set_ylabel("$\Theta$ [rad]",fontsize=18)
            
         #   ax.grid(True, linestyle='--', color='gray', linewidth=0.5)  # Turn on the grid, set linestyle, color, and linewidth

           # Customize the grid
           # ax.grid(False)  # Turn off the default grid

# Set the size of grid patches
           # patch_length_pT = 0.081
           # patch_width_theta = 0.772

# Loop to draw the grid patches
           # for i in range(0, int(max(data1[0])) + 1):
           #     for j in range(0,4):
           #         rect = patches.Rectangle((i , j), patch_length_pT, patch_width_theta, linewidth=1, edgecolor='gray', facecolor='none')
           #         ax.add_patch(rect)

            ax.text(x_0, y_0, f'{JetPlotter.legend_label(thres)}')
            ax.text(x_1, y_1, "Muon Collider", weight="bold", fontsize=18, fontstyle ='italic')
            ax.text(x_2, y_2, "Simulation Data")
            ax.text(x_3, y_3, f"$\sqrt{{s}}={energy} TeV$")
            ax.text(x_4, y_4, f'No Selection')
            ax.text(x_5, y_5, "TRIUMF Collaboration")
            ax.text(2, 3.18, "$\Theta = \pi$", fontsize=17, color='b', ha='center')
            ax.text(2, 0, '$\Theta = 0$', fontsize=17, color='b', ha='center')
            if (data1[2] == 'BIB 10%'):
                #ax.set_title('$p_{T}$ vs $\Theta$ of a jet, BIB 10%', fontsize= 18)
                text = "{}\n BIB 10%".format(JetPlotter.legend_label(thres))
            else:
                text = "{}\n BIB 10%".format(JetPlotter.legend_label(thres))
                #ax.set_title('$p_{T}$ vs $\Theta$ of a jet, BIB 0%, 2 MeV threshold', fontsize= 18)
            ax.text = (100, 3.3, text)
           # ax.text(15, 3.3, "Muon Collider", weight="bold", fontsize=18, fontstyle ='italic')
            #ax.text(70, 3.3, "Simulation Data")
            #ax.text(15, 3.1, "TRIUMF Collaboration")
            cbar3= plt.colorbar(hist[3])
            cbar3.set_label('Number of PFOs', rotation=270, loc ='center',labelpad= 15)
            ax.axhline(y = 0, color = 'b', linestyle = '--')
            ax.axhline(y = np.round(np.pi, 5), color = 'b', linestyle = '--')
           # ax.text(2, 3, "\u03C0", ha='center')
           # ax.text(2, 0, '0', fontsize=12)
            
            
        if (data1[1]=="eta_jet" and data2[1]=="phi_jet"):
            hist  = ax.hist2d(data1[0], data2[0], norm = colors.LogNorm(), bins=(50, 50), range =range_histo, cmap='BuPu')
            ax.set_xlabel('$\eta$', fontsize=18)
            ax.set_ylabel("$\Phi$ [rad]",fontsize=18)
               
           # Customize the grid
            ax.grid(False)  # Turn off the default grid

           # Set the size of grid patches
            patch_length_pT = 1.0
            patch_width_theta =1.0

           # Loop to draw the grid patches
            for i in range(0, int(max(data1[0])) + 1):
                for j in range(0,4):
                    rect = patches.Rectangle((i , j), patch_length_pT, patch_width_theta, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
        
            
            ax.text(x_0, y_0, f'{JetPlotter.legend_label(thres)}')
            ax.text(x_1, y_1, "Muon Collider", weight="bold", fontsize=18, fontstyle ='italic')
            ax.text(x_2, y_2, "Simulation Data")
            ax.text(x_3, y_3, f"$\sqrt{{s}}={energy} TeV$")
            if method == 'sk':
                ax.text(x_4, y_4, f'SoftKiller(a={a})')
            if method =='calo': 
                ax.text(x_4, y_4, "Calo Baseline Cut")
            if method == None:
                ax.text(x_4, y_4, f'No Selection')
            ax.text(x_5, y_5, "TRIUMF Collaboration")
            ax.text(0.2, 3.18, "$\Theta = \pi$", fontsize=17, color='b', ha='center')
            ax.text(0.2, 0, '$\Theta = 0$', fontsize=17, color='b', ha='center')
        
            cbar3= plt.colorbar(hist[3])
            cbar3.set_label('Number of Pfos', rotation=270, loc ='center',labelpad= 15)
            plt.axhline(y = 0, color = 'b', linestyle = '--')
            plt.axhline(y = np.round(np.pi, 6), color = 'b', linestyle = '--')
            
        if (data1[1]=="eta_pfojet" and data2[1]=="phi_pfojet"):
            hist  = ax.hist2d(data1[0], data2[0], norm = colors.LogNorm(), bins=(50, 50), range =range_histo, cmap='BuPu')
            ax.set_xlabel('$\eta$', fontsize=18)
            ax.set_ylabel("$\Phi$ [rad]",fontsize=18)
            
            
            
           # Customize the grid
            ax.grid(False)  # Turn off the default grid

           # Set the size of grid patches
            patch_length_pT = 1.0
            patch_width_theta =1.0

           # Loop to draw the grid patches
            for i in range(0, int(max(data1[0])) + 1):
                for j in range(0,4):
                    rect = patches.Rectangle((i , j), patch_length_pT, patch_width_theta, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
        
            
            ax.text(x_0, y_0, f'{JetPlotter.legend_label(thres)}')
            ax.text(x_1, y_1, "Muon Collider", weight="bold", fontsize=18, fontstyle ='italic')
            ax.text(x_2, y_2, "Simulation Data")
            ax.text(x_3, y_3, f"$\sqrt{{s}}={energy} TeV$")
            ax.text(x_4, y_4, f'No Selection')
            ax.text(x_5, y_5, "TRIUMF Collaboration")
            ax.text(0.2, 3.18, "$\Theta = \pi$", fontsize=17, color='b', ha='center')
            ax.text(0.2, 0, '$\Theta = 0$', fontsize=17, color='b', ha='center')
        
            cbar3= plt.colorbar(hist[3])
            cbar3.set_label('Number of Pfos', rotation=270, loc ='center',labelpad= 15)
            plt.axhline(y = 0, color = 'b', linestyle = '--')
            plt.axhline(y = np.round(np.pi, 6), color = 'b', linestyle = '--')
            
            
    
            
           
            
     # to do : function that acts on max and min of the data array and outputs proper x, yticks range 
     #         function that gives a customized legend specifies: sk, cal threshold, and BIB 
     #.         function for the title 