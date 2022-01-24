# -*- coding: utf-8 -*-
"""   
@author: Nikolaos Giakoumoglou
@date: Sun Aug 22 12:21:44 2021
"""

from __future__ import division
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import cv2
from scipy.spatial.distance import cdist

class Plaque:
    '''
    Parameters
    ----------
    ultrasound : ndarray
        Ultrasound that contrains a plaque.
    points : list
        List of points definined the plaque in the ultrasound
    points_lumen : list
        List of points above that are near lumen (subset of points).
    name : str
        Name of the given plaque.
    pad : int, optional
        Pad the plaque, Default value for padding is 2.
        
    Attributes
    -------
    plaque : ndarray
        Plaque as an image. 
    mask : ndarray
        Mask as an image. 
    perimeter : ndarray
        Perimeter as an image. 
    perimeter_lumen : ndarray
        Perimeter near lumen as an image.  
    points : list
        List of points definined the plaque in the ultrasound
    points_lumen : list
        List of points above that are near lumen (subset of points).
    name : str
        Name of the given plaque.
    '''
    
    def __init__(self, ultrasound, points, points_lumen, name, pad=2):
        
        self.name = name
        self.ultrasound = ultrasound
        self.points = points
        self.points_lumen = points_lumen
                
        # Find mask, and perimeter
        mask = cv2.fillPoly(np.zeros(ultrasound.shape,np.double), [points.reshape((-1,1,2))], color=1).astype('i')
        plaque = np.multiply(ultrasound.astype(np.double), mask).astype(np.uint8) # img with 0 outside mask
        perimeter = cv2.polylines(np.zeros(ultrasound.shape), [points.reshape((-1,1,2))],isClosed=True,color=1,thickness=1).astype('i')        
        perimeter_lumen = cv2.polylines(np.zeros(ultrasound.shape), [points_lumen.reshape((-1,1,2))],isClosed=False,color=1,thickness=1).astype('i')
    
        # Crop to plaques' dimensions
        max_point_x = max(points[:,0])
        max_point_y = max(points[:,1])
        min_point_x = min(points[:,0])
        min_point_y = min(points[:,1])  
        a,b,c,d = min_point_y, (max_point_y+1), min_point_x, (max_point_x+1)
        plaque = plaque[a:b,c:d]
        mask = mask[a:b,c:d]
        perimeter = perimeter[a:b,c:d]
        perimeter_lumen = perimeter_lumen[a:b,c:d] 
        
        self.plaque = Plaque.padImage(plaque, pad, 0)
        self.mask = Plaque.padImage(mask, pad, 0)
        self.perimeter = Plaque.padImage(perimeter, pad, 0)
        self.perimeter_lumen = Plaque.padImage(perimeter_lumen, pad, 0)

    def plotUnderlinedPlaque(self):
        '''
        Returns
        -------
        Plots ultrasound with the plaque underlined in yellow color.
        '''
        img = self.ultrasound
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        img = cv2.polylines(img, [self.points.reshape((-1,1,2))], True, (255,255,0), 3)
        plt.imshow(img)
        plt.title('Underlined Plaque in Ultrasound Image' + ' (' + self.name + ')')
        plt.show()
   
    def plotPlaque(self):
        '''
        Returns
        -------
        Plots plaque with white background (turn 0 to 255).
        '''
        img = self.plaque.copy()
        img[self.mask==0] = 255
        plt.imshow(img, cmap='gray')     
        plt.title('Plaque' + ' (' + self.name + ')')       
        plt.show()
        
    def plotPerimeter(self):
        '''
        Returns
        -------
        Plots plaque's perimeter.
        '''
        plt.imshow(Plaque.imageXOR(self.perimeter), cmap='gray')     
        plt.title("Plaque's Perimeter" + ' (' + self.name + ')')       
        plt.show()
        
    def plotMask(self):
        '''
        Returns
        -------
        Plots plaque's mask.
        '''
        plt.imshow(Plaque.imageXOR(self.mask), cmap='gray')     
        plt.title("Plaque's Mask" + ' (' + self.name + ')')       
        plt.show()        

    def plotHistogram(self, bins=32):
        '''
        Parameters
        ----------
        bins : TYPE, optional
            Number of bins to plot histogram. The default is 32.

        Returns
        -------
        Plots plaque's histogram.
        '''
        
        level_min = 0
        level_max = 255
        #n_level = (level_max - level_min) + 1
        #bins = n_level
    
        img_ravel = self.plaque.ravel() 
        mask_ravel = self.mask.ravel()
        roi = img_ravel[mask_ravel.astype(bool)] 
        plt.hist(roi, bins=bins, range=[level_min, level_max], density=False)     
        plt.title("Plaque's Histogram" + ' (' + self.name + ')')       
        plt.show()
        
    def plotGeroulakosClassification(self):
        '''
        Returns
        -------
        Plots Geroulakos Classification in plaques' components.
        '''
        cutoffs = np.array([[0,25],[26,50],[51,75],[76,100],[101,125],[126,255]])
        colors = [(0,0,0), (0,0,255),(0,255,0),(255,255,0),(255,165,0),(255,0,0)]
        colors2 = ['black','blue','green','yellow','orange','red']
        labels = ['(0-25)', '(26-50)', '(51-75)','(76-100)','(101-125)','(125-255)']
        
        c = np.zeros((self.plaque.shape[0],self.plaque.shape[1],3),np.uint32)
        h = np.zeros(len(colors),np.int32)
        
        for i in range(len(colors)):
            low = cutoffs[i,0]
            up = cutoffs[i,1]
            c[((self.plaque>=low) & (self.plaque<=up))] = colors[i]
            h[i] = np.sum(((self.plaque>=low) & (self.plaque<=up)))
        h[0] = h[0] - (self.mask==0).sum() # substract RONI
        c[self.mask==0] = (255,255,255) # substract RONI
        
        # Plot Contour - Geroulakos Classification
        custom_lines = [Line2D([0], [0], color=(0,0,0), lw=4),
                        Line2D([0], [0], color=(0,0,1), lw=4),
                        Line2D([0], [0], color=(0,1,0), lw=4),
                        Line2D([0], [0], color=(1,1,0), lw=4),
                        Line2D([0], [0], color=(1,165/255,0), lw=4),
                        Line2D([0], [0], color=(1,0,0), lw=4)]
        fig, ax = plt.subplots(1)
        plt.imshow(c)  
        ax.legend(custom_lines, labels)
        plt.title('Plaque Contour - Geroulakos Classification' + ' (' + self.name + ')')       
        plt.show()

        # Plot Histogram
        plt.bar(labels,h, color=colors2)
        plt.title('Plaque Contour - Histogram' + ' (' + self.name + ')')
        plt.show()
        
    @staticmethod
    def imageXOR(im):
        '''
        Parameters
        ----------
        im : ndarray
            Image input.

        Returns
        -------
        out : ndarray
            Turn "0" to "1" and vice versa: XOR with image consisting of "1".
        '''
        im = im.astype(np.uint8)
        mask = np.ones(im.shape, np.uint8)
        out = np.zeros(im.shape, np.uint8)
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                out[i,j] = im[i,j] ^ mask[i,j]
        return out    
    
    @staticmethod
    def padImage(im, pad=2, value=0):
        '''
        Parameters
        ----------
        im : ndarray
            Image input.
        pad : int, optional
            How many pixels to pad in each direction. Default is 2.
        value : int, optional
            Value of padded pixels

        Returns
        -------
        out : ndarray
            Padded image.
        '''
        TDLU=[1, 1, 1, 1]  #top, down, left, right pad
        out = im.copy()
        for _ in range(pad):
            out = cv2.copyMakeBorder(out, TDLU[0], TDLU[1], TDLU[2], TDLU[3],\
                                     cv2.BORDER_CONSTANT, None, value)
        return out
    
    @staticmethod
    def get_perc_ROI(mask, perimeter_lumen, perc=0.25):
        '''
        Parameters
        ----------
        mask : ndarray
            Mask input.
        perimeter_lumen : int
            Perimeter of the plaques' lumen
        perc : int, optional
            Percentage of plaque (as mask) to keep near lumen. Default is 25%.

        Returns
        -------
        area_lumen : ndarray
            Mask of the <perc> plaque that is near lumen.
        '''  
        dist = np.empty(mask.shape)
        dist[:] = np.inf
        II = np.argwhere(mask)
        JJ = np.argwhere(perimeter_lumen)
        K = tuple(II.T)
        dist[K] = cdist(II, JJ).min(axis=1, initial=np.inf)     
        percPixels = np.fix(perc * np.count_nonzero(mask) ).astype('i')
        def get_indices_of_k_smallest(arr, k):
            idx = np.argpartition(arr.ravel(), k)
            return tuple(np.array(np.unravel_index(idx, arr.shape))[:, range(min(k, 0), max(k, 0))])
        idx = get_indices_of_k_smallest(dist, percPixels)
        idx = np.array(idx,dtype=np.int32).T          
        area_lumen = np.zeros(mask.shape, dtype=np.int32)
        for i in range(idx.shape[0]):
            area_lumen[idx[i,0],idx[i,1]] = 1 
        return area_lumen