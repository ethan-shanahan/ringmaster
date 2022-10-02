import numpy as np
import matplotlib.pyplot as mplp
import matplotlib.animation as mpla


class Visualiser():
    '''docstring'''

    def __init__(self, data: np.ndarray | list | dict, output_path: str) -> None:
        self.path = output_path
        self.data = data
        if isinstance(self.data, np.ndarray):
            self.fig = mplp.figure(figsize=self.data.shape)
            self.ax = mplp.axes()
            self.artist = getattr(self, "gen_image")
        elif isinstance(self.data, list):
            self.fig = mplp.figure(figsize=self.data[0].shape)
            self.ax = mplp.axes()
            self.artist = getattr(self, "gen_movie")
            self.nframes = len(self.data)
        elif isinstance(self.data, dict):
            self.fig, self.axes = mplp.subplots(1, len(self.data)-1)
            self.artist = getattr(self, "gen_graphs")
            self.graphs = dict(
                zip([k for n, k in enumerate(self.data.keys()) if n != 0], self.axes))
        else:
            raise TypeError(
                f'Visualiser does not support {type(self.data)} data.')
        self.fig.set_tight_layout(True)

    def gen_image(self):
        mplp.figure(self.fig)
        self.ax.set_axis_off()
        self.ax.imshow(self.data, interpolation='none', cmap='gray')
        mplp.savefig(self.path, dpi=300)

    def gen_movie(self):
        mplp.figure(self.fig)
        self.ax.set_axis_off()
        animator = mpla.FuncAnimation(
            self.fig, self.gen_frame, frames=self.nframes, interval=200, blit=True)
        print(f'Generating {self.nframes} Frames: ', end='')
        animator.save(self.path, dpi=100, writer='pillow',
                      progress_callback=lambda i, n: print(f'{i}', end='-')) # ? look into progress_callback

    def gen_graphs(self):
        mplp.figure(self.fig)
        for column, axe in self.graphs.items():
            axe.plot(next(iter(self.data)), column,
                     data=self.data)  # TODO design graphs
        mplp.savefig(self.path, dpi=300)

    def gen_frame(self, i) -> list:
        return [self.ax.imshow(self.data[i], interpolation='none', cmap='gray')]


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
