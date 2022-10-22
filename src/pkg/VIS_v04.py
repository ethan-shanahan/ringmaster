import numpy as np
import pandas as pd
import matplotlib.pyplot as mplp
import matplotlib.animation as mpla
from pkg import utils


class Visualiser():
    '''docstring'''

    def __init__(
            self, 
            data: pd.DataFrame | np.ndarray | list | dict, 
            output_path: str, 
            art_type: str, 
            save: bool = True
        ) -> None:
        self.data, dtype = data, type(data).__name__.lower()
        self.path = output_path
        self.artist = getattr(self, f'gen_{art_type}_from_{dtype}')
        if art_type == 'image':
            self.fig = mplp.figure(figsize=self.data.shape)
            self.ax = mplp.axes()
            self.artist = getattr(self, "gen_image")
            self.fig.set_tight_layout(True) # !!!
        elif art_type == 'movie':
            self.fig = mplp.figure(figsize=self.data[0].shape)
            self.ax = mplp.axes()
            self.artist = getattr(self, "gen_movie")
            self.nframes = len(self.data)
            self.fig.set_tight_layout(True) # !!!
        elif art_type != 'graph':
            raise TypeError(f'Visualiser does not support {art_type} outputs.')
        self.save = save

    # * Graph Generators
    def gen_graph_from_dataframe(self):
        fig, axes = mplp.subplots(1, len(self.data.columns))
        for col, axe in zip(list(self.data.columns), axes):
            axe.plot(col, data=self.data)  # TODO design graph
        if self.save: mplp.savefig(self.path, dpi=300)
        else: fig.show()
    
    def gen_graph_from_ndarray(self):
        fig, axe = mplp.subplots()
        axe.plot(self.data[0], self.data[1])
        if self.save: mplp.savefig(self.path, dpi=300)
        else: fig.show()
    
    def gen_graph_from_list(self):
        fig, axe = mplp.subplots()
        for d in self.data:
            if isinstance(d, list):
                for dd in d:
                    axe.plot(dd[0], dd[1])
            elif isinstance(d, np.ndarray):
                axe.plot(d[0], d[1])
            else:
                axe.plot(self.data)
                break
        
        if self.save: mplp.savefig(self.path, dpi=300)
        else: fig.show()

    def gen_graph_from_dict(self):
        if len(self.data) > 4:
            utils.dual_print('There are too many samples to graph a time-series currently.')
            return
        elif len(self.data) == 4:
            rows, cols = 2, 2
        else:
            rows, cols = len(self.data), 1
        fig, axes = mplp.subplots(nrows=rows, ncols=cols)
        for n, d in enumerate(self.data.values()):
            if isinstance(d, list):
                axes.flatten()[n].plot(d)
            elif isinstance(d, np.ndarray):
                axes.flatten()[n].plot(d[0], d[1])
            else:
                print('graphing not supported')
                break
        if self.save: mplp.savefig(self.path, dpi=300)
        else: fig.show()
    
    # * Image Generators
    def gen_image(self):
        mplp.figure(self.fig)
        self.ax.set_axis_off()
        self.ax.imshow(self.data, interpolation='none', cmap='gray')
        if self.save: mplp.savefig(self.path, dpi=300)

    # * Movie Generators
    def gen_movie(self):
        mplp.figure(self.fig)
        self.ax.set_axis_off()
        animator = mpla.FuncAnimation(
            self.fig, self.gen_frame, frames=self.nframes, interval=200, blit=True)
        utils.dual_print(f'\nGenerating {self.nframes} Frames: ', end='')
        if self.save: animator.save(self.path, dpi=100, writer='pillow',
                                    progress_callback=lambda i, _: utils.dual_print(i, end='-', flush=True))

    def gen_frame(self, i) -> list:
        return [self.ax.imshow(self.data[i], interpolation='none', cmap='gray')] # !pre-map to remove fluctuations


if __name__ == '__main__':
    rng = np.random.default_rng()
    out_type = input('image/movie/graphs\t\t| ')
    out_num = input('output number\t\t\t| ')
    if out_type == 'image':
        grid = rng.integers(0, 3, size=(20, 20), endpoint=True)
        out_path = f'output/visualiser_test{out_num}.png'
        outputting = Visualiser(data=grid, output_path=out_path)
    elif out_type == 'movie':
        book = [rng.integers(0, 3, size=(20, 20), endpoint=True)
                for _ in range(10)]
        out_path = f'output/visualiser_test{out_num}.gif'
        outputting = Visualiser(data=book, output_path=out_path)
    outputting.artist()
