import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib

font = {'weight': 'normal', 'size': 12}
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rc('font', **font)

def newlegend(col=1, fsize=12):
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels))
              if l not in labels[:i]]
    plt.gca().legend(*zip(*unique), fontsize=fsize, ncol=int(col))

def get_cmap(n, name='jet'):
    return plt.cm.get_cmap(name, n)


class Visual:
    def __init__(self, nu, nLST, ant, save=False):
        self.nu = nu
        self.nLST = nLST
        self.ant = ant
        self.save = save
        if self.save:
            if not os.path.exists('output'): os.mkdir('output')
        
    def plotModset(self, set, opt, n_curves=100):
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
        if self.save: plt.savefig('output/modeling%sSet.pdf'%opt, bbox_inches='tight')
        else: plt.show()

    def plotMockObs(self, y21, yFg, noise):
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
        if self.save: plt.savefig('output/mockObservation.pdf', bbox_inches='tight')
        else: plt.show()

    def plotBasis(self, basis, opt, n_curves=5):
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
        if self.save: plt.savefig('output/basis%s.pdf'%opt, bbox_inches='tight')
        else: plt.show()

    def plotInfoGrid(self, file, modesFg, modes21):
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
        if self.save: plt.savefig('output/DICinfoGrid.pdf', bbox_inches='tight')
        else: plt.show()

    def plotExtSignal(self, y21, recons21, sigma21):
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
        if self.save: plt.savefig('output/extracted21.pdf', bbox_inches='tight')
        else: plt.show()
