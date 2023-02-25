import os, sys, time, math, tomli

def parse_config() -> dict:
    with open(f'{get_src()}/config.toml', mode='rb') as toml:
        config = tomli.load(toml)
    options = config.pop('DEFAULT')
    print(f'\nThe following presets were found: {list(config.keys())}', sep=' - ', end='\n\n')
    if preset := input('Please enter the name of the preset you would like to use,'
                                ' or enter none to use the default settings.\t\t| '): options.update(config[preset])
    else: preset = 'DEFAULT'
    print(); uprint(indent := f'Loading {preset}:') ; indent = ' ' * len(indent)
    for k, v in options.items(): uprint(f'{indent}{k:.<25}{v}')
    uprint()
    return options

def get_src() -> str:
    '''Determines the absolute path of the project's src folder.'''
    src = os.path.realpath(os.path.dirname(__file__))
    while src[-3:] != 'src': src = os.path.dirname(src)
    return src

def mk_output_dir() -> str:
    '''Creates a consecutively numbered folder in the project's output folder. Stops after 99 folders.'''
    x = 0
    while (x := x + 1) != 100:
        try: os.mkdir(get_src().replace('src', f'output\{x:02d}')); break
        except: continue
    else: raise Exception('! The number of output folders has reached the maximum permissible number, 99.')

def open_log() -> None:
    '''Opens a TextIOWrapper for the log of the currently active output folder, and makes it global.'''
    global log; log = open(mk_output_dir()+r'log.txt', 'w', encoding='utf-8')

def close_log() -> None:
    '''Closes the global log file.'''
    log.close()

def uprint(*values: object, sep: str | None = ' ', end: str | None = '\n', file = None, flush : bool = False) -> None:
    '''Prints to both the terminal and the log file.'''
    print(*values, sep=sep, end=end, file=sys.stdout, flush=flush)
    try: print(*values, sep=sep, end=end, file=log, flush=flush)
    except NameError: pass  # print('WARNING: Trying to use uprint() without an opened log file.')

def intlen(integer : int) -> int:
    return int(math.log10(integer))+1

def dim_check(subject : tuple[int], dim : tuple[int], ndim : int) -> bool:
    checker = []
    for n in range(ndim): checker.append(-1 < subject[n] < dim[n])
    return all(checker)


class ProgressBar():
    '''A progress bar, but dirty.'''
    def __init__(self, header : str = '', footer : str = '', entity : str = '', buffer : int = 25, jobs : int = 1, steps : int = 100) -> None:
        self.master_start = time.time()
        self.job_times = []
        self.buffer = buffer
        self.footer = '*'*self.buffer + f'*{" " + footer + " ":*^100}*' + '*'*35 + '\n'
        self.jobs = jobs
        self.j = 1
        self.offset = 0
        self.steps_len = intlen(steps) if intlen(steps) > 3 else 4
        h = (
            f'{entity:^{self.buffer}}'+
            f'|{header:^100}|'+
            f'{"DONE/TODO":^{(self.steps_len*2)+3}}[   %] ~ elapsed ~  eta '
        )
        uprint(h)

    def mk_bar(self, steps : int, prefix : str | int = '') -> None:
        self.start = True
        self.start_time = time.time()
        self.steps = steps
        self.prefix = prefix
        bar = (
            f'{self.prefix:^{self.buffer}}|'+
            '·'*100+
            f'| {0:{self.steps_len}d}/{self.steps:<{self.steps_len}d} [  0%] ~  00:00  ~'
        )
        print(bar, end='            \r')

    def bar_step(self, current_step) -> None:
        if self.start: 
            self.init_time = time.time() - self.start_time
            self.start_time = time.time()
            self.start = False
        m, s = divmod(round(self.init_time+time.time()-self.start_time), 60)
        if current_step == 0: self.offset = 1
        current_step = self.offset + current_step
        percent = int((current_step / self.steps) * 100)
        done = '█' * percent
        todo = '·' * (100 - percent)
        bar = (
            f'{self.prefix:^{self.buffer}}|'+
            done + todo +
            f'| {current_step:{self.steps_len}d}/{self.steps:<{self.steps_len}d} [{percent:3}%] ~{f"{m:02d}:{s:02d}":^9}~'
        )
        if current_step != self.steps:
            # print(f'{current_step=}    ')
            if current_step == 1: print(bar, end='\r')
            else:
                if percent != 0 and percent % 10 == 0:
                    est_total = ((time.time()-self.start_time) * (100/percent) + self.init_time) * self.jobs
                    self.eta = est_total - sum(self.job_times) - (self.init_time+time.time()-self.start_time)
                    m, s = divmod(round(self.eta), 60)
                    bar += f'{f"{m:02d}:{s:02d}":^9}'
                if ((current_step / self.steps) * 100) % 1 == 0:
                    print(bar, end='\r')
        else:
            self.job_times.append(self.init_time+time.time()-self.start_time)
            uprint(bar, end='           \n')
            if self.j == self.jobs: print(self.footer)
            self.j += 1


if __name__ == '__main__':
    c = parse_config(path_to_config=r'src/')
