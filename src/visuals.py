import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib
import math

font = {'weight': 'normal', 'size': 12}
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rc('font', **font)

def newlegend(col=1, fsize=12):
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels))
              if l not in labels[:i]]
    plt.gca().legend(*zip(*unique), fontsize=fsize, ncol=int(col))

def get_cmap(n, name='brg'):
    return plt.cm.get_cmap(name, n)

def lighten_color(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


class Visual:
    def __init__(self, nu, nLST, ant, save=False):
        """Initialize to generate the plots.

        Args:
            nu (array): Frequency range
            nLST (int): Number of time bins to fit
            ant (list): List of antenna designs
            save (bool, optional): Option to save the plot. Defaults to False.
        """
        self.nu = nu
        self.nLST = nLST
        self.ant = ant
        self.save = save
        if self.save:
            if not os.path.exists('FigsOutput'): os.mkdir('FigsOutput')
        
    def plotModset(self, set, opt, n_curves=100):
        """Plots the modeling set.

        Args:
            set (array): Modeling set
            opt (string): '21' or 'FG'
            n_curves (int, optional): Number of curves to plot. Defaults to 100.
        """
        plt.figure(figsize=(6, 4))
        if opt == '21':
            plt.plot(self.nu, set.T[:, :n_curves], lw=0.2)
        if opt == 'FG':
            cmap = get_cmap(self.nLST)
            for i in range(self.nLST):
                plt.plot(self.nu, set.T[i*len(self.nu):(i+1)*len(self.nu), :n_curves],
                         c=cmap(i), label=r'$t_{%d}$'%(i+1), alpha=0.2, lw=0.8)
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            newlegend()
        plt.xlim(50, 200)
        plt.xlabel(r'$\nu\ ({\rm MHz})$')
        plt.ylabel(r'$T_{\rm b}\ ({\rm K})$')
        if self.save: plt.savefig('FigsOutput/modeling%sSet.pdf'%opt, bbox_inches='tight')
        else: plt.show()

    def plotMockObs(self, y21, yFg, noise):
        """Plots the mock observation.

        Args:
            y21 (array): Input 21cm component
            yFg (array): Input beam-weighted foreground component
            noise (array): Noise realization
        """
        plt.figure(figsize=(13, 3.5))
        plt.subplot(131)
        plt.plot(self.nu, y21, label=r'$y_{21}$', c='k')
        plt.legend()
        plt.xlim(50, 200)
        plt.ylabel(r'$T_{\rm b}\ ({\rm K})$')
        
        plt.subplot(132)
        cmap = get_cmap(self.nLST)
        lsty = ['-', '--', '-.']
        for j in range(len(self.ant)):
            for i in range(self.nLST):
                plt.plot(self.nu, yFg[(i+self.nLST*j)*len(self.nu)
                                      :(i+self.nLST*j+1)*len(self.nu)],
                         c=cmap(i), ls=lsty[j],
                         label=r'$y_{\rm FG}\ (t_{%d})$ %s'%(i+1, self.ant[j]))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        plt.xlabel(r'$\nu\ ({\rm MHz})$')
        plt.xlim(50, 200)
        newlegend(fsize=6, col=len(self.ant))
        
        plt.subplot(133)
        for j in range(len(self.ant)):
            for i in range(self.nLST):
                plt.plot(self.nu, noise[(i+self.nLST*j)*len(self.nu)
                                        :(i+self.nLST*j+1)*len(self.nu)],
                        c=cmap(i), ls=lsty[j],
                        label=r'$n\ (t_{%d})$ %s'%(i+1, self.ant[j]))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        newlegend(fsize=6, col=len(self.ant))
        plt.xlim(50, 200)
        if self.save: plt.savefig('FigsOutput/mockObservation.pdf', bbox_inches='tight')
        else: plt.show()

    def plotBasis(self, basis, opt, n_curves=5):
        """Plots the basis functions.

        Args:
            basis (array): Basis functions (or modes)
            opt (string): '21' or 'FG'
            n_curves (int, optional): Number of modes to plot. Defaults to 5.
        """
        plt.figure(figsize=(6, 4))
        cmap = get_cmap(n_curves)
        if opt == '21':
            for i in range(n_curves):
                plt.plot(self.nu, basis.T[i], c=cmap(i),
                         label=r'$F^{21}_{%d}$'%(i+1))
        if opt == 'FG':
            lsty = ['-', '--', '-.']
            for i in range(n_curves):
                for j in range(self.nLST):
                    plt.plot(self.nu, basis.T[i, j*len(self.nu):(j+1)*len(self.nu)],
                             c=cmap(i), ls=lsty[j], label=r'$F^{\rm FG}_{%d\ (t_{%d})}$'%(i+1, j+1))
        plt.legend(fontsize=9, ncol=n_curves)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        plt.xlim(50, 200)
        plt.xlabel(r'$\nu\ ({\rm MHz})$')
        plt.ylabel(r'$T_{\rm b}\ ({\rm K})$')
        if self.save: plt.savefig('FigsOutput/basis%s.pdf'%opt, bbox_inches='tight')
        else: plt.show()

    def plotInfoGrid(self, file, modesFg, modes21):
        """Plots the information criteria estimated on a grid.

        Args:
            file (string): Filename that contains the gridded information
            modesFg (int): Number of foreground modes 
            modes21 (int): Number of 21cm modes
        """
        modesFg = np.linspace(1, modesFg, num=modesFg)
        modes21 = np.linspace(1, modes21, num=modes21)

        fp = np.loadtxt(file)
        dicVals = fp[:, 2]
        X, Y = np.meshgrid(modesFg, modes21)
        DIC_array = dicVals.reshape(len(modesFg), len(modes21))

        plt.figure(figsize=(5.7, 5))
        plt.pcolormesh(Y, X, DIC_array.T, cmap="jet",
                       shading='nearest', rasterized=True,
                       norm=colors.LogNorm(vmin=min(dicVals), vmax=min(dicVals)*1.2))
        plt.xlabel(r'21cm modes $(n_{b}^{21})$')
        plt.ylabel(r'Foreground modes $(n_{b}^{\rm FG})$')
        fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        cbar = plt.colorbar(aspect=30, format=fmt, extend='max', pad=0.02)
        cbar.set_label(r"$\delta^{T} C^{-1} \delta\ +\ 2\, (n_{b}^{\rm FG} + n_{b}^{21})$")
        if self.save: plt.savefig('FigsOutput/DICinfoGrid.pdf', bbox_inches='tight')
        else: plt.show()

    def plotExtSignal(self, y21, recons21, sigma21):
        """Plots the extracted 21cm signal with the input signal.

        Args:
            y21 (array): Input 21cm signal
            recons21 (array): Reconstructed signal
            sigma21 (array): 1sigma interval
        """
        plt.figure(figsize=(6, 4))
        plt.plot(self.nu, recons21[0:len(self.nu)], c='darkcyan',
                 label='Extracted 21cm signal')
        plt.fill_between(self.nu, recons21[0:len(self.nu)] - (1*sigma21),
                        recons21[0:len(self.nu)] + (1*sigma21),
                        color="darkcyan", alpha=0.4)
        plt.plot(self.nu, y21, c='k', label='Input 21cm signal')
        plt.legend()
        plt.xlim(50, 200)
        plt.xlabel(r'$\nu\ ({\rm MHz})$')
        plt.ylabel(r'$T_{\rm b}\ ({\rm K})$')
        if self.save: plt.savefig('FigsOutput/extracted21.pdf', bbox_inches='tight')
        else: plt.show()

    def plotBiasCDF(self, antNames, fnames, save=False):
        """Plots the CDF of the signal bias for an ensemble of mockobs.

        Args:
            antNames (list): Antennas   
            fnames (list): Files
            save (bool, optional): For saving the figure. Defaults to False.
        """
        plt.figure(figsize=(6, 4))
        color = get_cmap(len(fnames))
        lim = [0.68, 0.95]
        text = self.getCDF(paths=fnames, ind=3, lim=lim, labels=antNames, color=color)
        self.getTable(paths=fnames, labels=antNames, text=text, color=color)
        self.plotChiCDF()
        plt.xlabel(r"${\rm Signal\ bias\ statistic\ }(\varepsilon)$")
        plt.ylabel(r"${\rm CDF}$")
        plt.xlim(0, 4); plt.ylim(0, 1)
        plt.legend(loc='upper left', fontsize=9)
        if save: plt.savefig('FigsOutput/biasCDF_lst-%d.pdf'%self.nLST, bbox_inches='tight')
        else: plt.show()
        
    def plotRmsCDF(self, antNames, fnames, save=False):
        """Plots the CDF of the rms uncertainty for an ensemble of mockobs.

        Args:
            antNames (list): Antennas
            fnames (list): Files
            save (bool, optional): For saving the figure. Defaults to False.
        """
        plt.figure(figsize=(6, 4))
        color = get_cmap(len(fnames))
        lim = [0.68, 0.95]
        text = self.getCDF(paths=fnames, ind=5, lim=lim, labels=antNames, color=color)
        self.getTable(paths=fnames, labels=antNames, text=text, color=color)
        plt.xscale("log")
        plt.ylim(0, 1); plt.xlim(1, pow(10, 3))
        plt.legend(loc='upper left', fontsize=9)
        plt.xlabel(r"${\rm RMS\ uncertainity\ }({\rm mK})$")
        plt.ylabel(r"${\rm CDF}$")
        if save: plt.savefig('FigsOutput/rmsCDF_lst-%d.pdf'%self.nLST, bbox_inches='tight')
        else: plt.show()
    
    def plotNormD(self, antNames, fnames, save=False, bins=20):
        """Plots the normalized Deviance for an ensemble of mockobs.

        Args:
            antNames (list): Antennas
            fnames (list): Files
            save (bool, optional): For saving the figure. Defaults to False.
            bins (int, optional): Number of bins. Defaults to 20.
        """
        plt.figure(figsize=(6, 4))
        color = get_cmap(len(fnames))
        labels = antNames
        paths = fnames
        for i in range(len(paths)):
            f = np.loadtxt("%s"%paths[i])
            weights = np.ones_like(f[:,4])/float(len(f[:,4]))
            plt.hist(f[:,4], weights=weights, bins=bins, histtype='step', label=labels[i],
                     color=color(i))
        plt.xlim(0.5, 1.5)
        plt.legend(loc='best')
        plt.xlabel(r"$\chi^2$")
        plt.ylabel(r"${\rm PDF}$")
        plt.tight_layout()
        if save: plt.savefig('FigsOutput/normDevPDF_lst-%d.pdf'%self.nLST, bbox_inches='tight')
        else: plt.show()

    @staticmethod
    def getCDF(paths, ind, lim, labels, color):
        text = np.zeros(shape=(len(paths), len(lim)))
        for i in range(len(paths)):
            f = np.loadtxt("%s"%paths[i])
            values, base = np.histogram(f[:, ind], bins = 5000)
            cumulative = np.cumsum(values)/np.sum(values)
            for j in range(len(lim)):
                total_ind = list(np.where(cumulative-lim[j]>=pow(10,-31)))
                index = total_ind[0][0]
                text[i][j] = "%.2f"%base[:-1][index]
            plt.plot(base[:-1], cumulative, label = labels[i], color = color(i))     
        plt.axhline(0.68, c = "lightgray", ls = ":")
        plt.axhline(0.95, c = "lightgray", ls = "-.")   
        return text

    @staticmethod
    def getTable(paths, labels, text, color):
        cell_t = []
        for i in range(len(paths)):
            cell_t.append("%s" %labels[i])
            cell_t.append("%.2f"%text[:,0][i])
            cell_t.append("%.2f"%text[:,1][i])
        ct = np.reshape(cell_t, (len(paths), 3))
        cellC = []
        for i in range(len(paths)):
            cellC.append(lighten_color(color(i), 0.25))
        cellcol = np.empty(shape=(len(paths), 3)).tolist()
        for i in range(len(paths)):
            for j in range(3):
                cellcol[i][j] = cellC[i]
        c_lab = (r"${\rm Observation}$", r"$68\%$", r"$95\%$")
        the_table = plt.table(cellText=ct, colLabels=c_lab, cellLoc="center",
                              colWidths=[0.32, 0.1, 0.1], cellColours=cellcol, loc = 4)
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(9)

    @staticmethod
    def plotChiCDF():
        pm = []; pm2 = []
        ii = np.linspace(0, 100, 5000)
        for jj in ii:
            pm.append(np.sqrt(2/np.pi)*np.exp(-pow(jj,2.)/2))
            pm2.append(math.fsum(pm))
        pm2 = pm2/sum(pm)
        plt.plot(ii, pm2, c="k")
