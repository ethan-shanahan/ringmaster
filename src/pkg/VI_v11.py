''' Visual Interface '''
import numpy as np
import matplotlib.pyplot as mplp
import matplotlib.figure as mplf


class VisualInterface():
    def __init__(self, draft : bool = True) -> None:
        self.figures = {}
        if draft:
            self.dpi = 100
        else:
            self.dpi = 300

    def set_fig(self, title : str) -> None:
        self.figures[title] = mplp.figure(num=title, figsize=[8,6], dpi=self.dpi, layout='constrained')
    
    def setget_ax(self, figure_title : str, rows : int = 1, cols : int = 1):
        return self.figures[figure_title].subplots(rows, cols)
    
    def plotter(self, data : np.ndarray | list[np.ndarray], title : str, rows : int = 1, cols : int = 1):
        self.set_fig(title); ax = self.setget_ax(title, rows, cols)
        if type(data) == np.ndarray:
            ax.plot(data[0], data[1], '.b', label=title)
        elif type(data) == list:
            colours = ('b', 'r', 'g')
            for i, d in enumerate(data): ax.plot(d[0], d[1], f'.{colours[i]}', label=f'{i} ~ {title}')

        ax.grid(); ax.legend()
        ax.set_xscale('log'); ax.set_xlabel('x')
        ax.set_yscale('log'); ax.set_ylabel('Pr(x)')
    
    @staticmethod
    def show():
        mplp.show()



if __name__ == '__main__':
    print(f'{mplf.dpi=}')