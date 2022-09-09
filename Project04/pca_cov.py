'''pca_cov.py
Performs principal component analysis using the covariance matrix approach
YOUR NAME HERE
CS 251 Data Analysis Visualization
Spring 2021
'''
from matplotlib import markers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class PCA_COV:
    '''
    Perform and store principal component analysis results
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: pandas DataFrame. shape=(num_samps, num_vars)
            Contains all the data samples and variables in a dataset.

        (No changes should be needed)
        '''
        self.data = data

        # vars: Python list. len(vars) = num_selected_vars
        #   String variable names selected from the DataFrame to run PCA on.
        #   num_selected_vars <= num_vars
        self.vars = None

        # A: ndarray. shape=(num_samps, num_selected_vars)
        #   Matrix of data selected for PCA
        self.A = None

        # normalized: boolean.
        #   Whether data matrix (A) is normalized by self.pca
        self.normalized = None

        # A_proj: ndarray. shape=(num_samps, num_pcs_to_keep)
        #   Matrix of PCA projected data
        self.A_proj = None

        # e_vals: ndarray. shape=(num_pcs,)
        #   Full set of eigenvalues (ordered large-to-small)
        self.e_vals = None
        # e_vecs: ndarray. shape=(num_selected_vars, num_pcs)
        #   Full set of eigenvectors, corresponding to eigenvalues ordered large-to-small
        self.e_vecs = None

        # prop_var: Python list. len(prop_var) = num_pcs
        #   Proportion variance accounted for by the PCs (ordered large-to-small)
        self.prop_var = None

        # cum_var: Python list. len(cum_var) = num_pcs
        #   Cumulative proportion variance accounted for by the PCs (ordered large-to-small)
        self.cum_var = None
        
        self.min = None
        self.max = None

    def get_prop_var(self):
        '''(No changes should be needed)'''
        return self.prop_var

    def get_cum_var(self):
        '''(No changes should be needed)'''
        return self.cum_var

    def get_eigenvalues(self):
        '''(No changes should be needed)'''
        return self.e_vals

    def get_eigenvectors(self):
        '''(No changes should be needed)'''
        return self.e_vecs

    def covariance_matrix(self, data):
        '''Computes the covariance matrix of `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_vars)
            `data` is NOT centered coming in, you should do that here.

        Returns:
        -----------
        ndarray. shape=(num_vars, num_vars)
            The covariance matrix of centered `data`

        NOTE: You should do this wihout any loops
        NOTE: np.cov is off-limits here — compute it from "scratch"!
        '''
        r,c = data.shape
        p1 = 1/(r-1)
        
        Ac = data - data.mean(axis = 0)
        c = Ac.T@Ac
        c = c*p1
        
        return c
        
    def compute_prop_var(self, e_vals):
        '''Computes the proportion variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        e_vals: ndarray. shape=(num_pcs,)

        Returns:
        -----------
        Python list. len = num_pcs
            Proportion variance accounted for by the PCs
        '''
            
        sum = np.sum(e_vals)
        
        p = []
       
        for val in e_vals:
           p.append(val/sum)
        
        return p

    def compute_cum_var(self, prop_var):
        '''Computes the cumulative variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        prop_var: Python list. len(prop_var) = num_pcs
            Proportion variance accounted for by the PCs, ordered largest-to-smallest
            [Output of self.compute_prop_var()]

        Returns:
        -----------
        Python list. len = num_pcs
            Cumulative variance accounted for by the PCs
        '''
        cum_var = []
        for i in range(len(prop_var)):
            cum_var.append(np.sum(prop_var[:i+1]))

        return(cum_var)

    def pca(self, vars, normalize=False):
        '''Performs PCA on the data variables `vars`

        Parameters:
        -----------
        vars: Python list of strings. len(vars) = num_selected_vars
            1+ variable names selected to perform PCA on.
            Variable names must match those used in the `self.data` DataFrame.
        normalize: boolean.
            If True, normalize each data variable so that the values range from 0 to 1.

        NOTE: Leverage other methods in this class as much as possible to do computations.

        TODO:
        - Select the relevant data (corresponding to `vars`) from the data pandas DataFrame
        then convert to numpy ndarray for forthcoming calculations.
        - If `normalize` is True, normalize the selected data so that each variable (column)
        ranges from 0 to 1 (i.e. normalize based on the dynamic range of each variable).
            - Before normalizing, create instance variables containing information that would be
            needed to "undo" or reverse the normalization on the selected data.
        - Make sure to compute everything needed to set all instance variables defined in constructor,
        except for self.A_proj (this will happen later).
        '''
        #get data set from desired variables
        ds = self.data[vars]
        self.A = ds
        
        #normalize if true
        if normalize:
            min_data = ds.min()
            self.min = min_data
            max_data = ds.max()
            self.max = max_data
            min_max = max_data-min_data
            
            ds = ds - min_data
            ds = ds/min_max
            self.normalized = ds
         
        #center the data and get covariance matrix
        c = self.covariance_matrix(ds)
        
        #get eighvectors and eighvalues
        (evals, v) = np.linalg.eig(c)
        
        #sort evals
        sortInd = np.argsort(evals)[::-1]
        sortedEvals = [x for _,x in sorted(zip(sortInd,evals))]
        sortedV = [x for _,x in sorted(zip(sortInd,v))]
        
        sortedV=np.array(sortedV)
        sortedEvals=np.array(sortedEvals)
        
        #compute prop variance
        propVar = self.compute_prop_var(sortedEvals)
        #compute cumulative variance
        cumVar = self.compute_cum_var(propVar)
        
        #set variables to the init function
        self.vars = vars
        self.e_vals = sortedEvals
        self.e_vecs = sortedV
        self.prop_var = propVar
        self.cum_var = cumVar  
        self.normalized = normalize      

    def elbow_plot(self, num_pcs_to_keep=None):
        '''Plots a curve of the cumulative variance accounted for by the top `num_pcs_to_keep` PCs.
        x axis corresponds to top PCs included (large-to-small order)
        y axis corresponds to proportion variance accounted for

        Parameters:
        -----------
        num_pcs_to_keep: int. Show the variance accounted for by this many top PCs.
            If num_pcs_to_keep is None, show variance accounted for by ALL the PCs (the default).

        NOTE: Make plot markers at each point. Enlarge them so that they look obvious.
        NOTE: Reminder to create useful x and y axis labels.
        NOTE: Don't write plt.show() in this method
        '''
        
        if num_pcs_to_keep == None:
            x = [0]
            for i in range(len(self.cum_var)):
                x.append(i+1)
                
            y = [0]
            for i in self.cum_var:
                y.append(i)
        else:
            
            x = [0]
            y = [0]
            for i in range(num_pcs_to_keep):
                x.append(i+1)
                y.append(self.cum_var[i])
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        
        plt.plot(x,y, marker = 'o')
        
        plt.xlabel("Number of eigenvectors")
        plt.ylabel("Cumulative variance")

    def pca_project(self, pcs_to_keep):
        '''Project the data onto `pcs_to_keep` PCs (not necessarily contiguous)

        Parameters:
        -----------
        pcs_to_keep: Python list of ints. len(pcs_to_keep) = num_pcs_to_keep
            Project the data onto these PCs.
            NOTE: This LIST contains indices of PCs to project the data onto, they are NOT necessarily
            contiguous.
            Example 1: [0, 2] would mean project on the 1st and 3rd largest PCs.
            Example 2: [0, 1] would mean project on the two largest PCs.

        Returns
        -----------
        pca_proj: ndarray. shape=(num_samps, num_pcs_to_keep).
            e.g. if pcs_to_keep = [0, 1],
            then pca_proj[:, 0] are x values, pca_proj[:, 1] are y values.

        NOTE: This method should set the variable `self.A_proj`
        '''
        Ac = self.A - np.mean(self.A, axis = 0)
        v = self.e_vecs[:, pcs_to_keep]

        self.A_proj = Ac@v

        return self.A_proj

    def pca_then_project_back(self, top_k):
        '''Project the data into PCA space (on `top_k` PCs) then project it back to the data space

        Parameters:
        -----------
        top_k: int. Project the data onto this many top PCs.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_selected_vars)

        TODO:
        - Project the data on the `top_k` PCs (assume PCA has already been performed).
        - Project this PCA-transformed data back to the original data space
        - If you normalized, remember to rescale the data projected back to the original data space.
        '''
        pcs_to_keep = np.arange(top_k)
        vhat = self.e_vecs[:, :top_k]
        self.A_proj = self.pca_project(pcs_to_keep)
        Ar = (self.A_proj @ vhat.T)
        Acr = Ar + Ar.mean(axis=0)

        if self.normalized:
            range = self.max-self.min
            r = pd.DataFrame(range)
            r =r.to_numpy()
            r = r.reshape((4,))
            ds = Acr * r
            
            m = pd.DataFrame(self.min)
            m = m.to_numpy()
            m = m.reshape((4,))
            ds = ds + m
            
        ds = Acr

        return ds
        
        
        