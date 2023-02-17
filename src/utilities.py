import os, sys, time, math

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

def uprint(*values: object, sep: str | None = ' ', end: str | None = '\n', file = None, flush : bool = False):
    '''Prints to both the terminal and the log file.'''
    print(*values, sep=sep, end=end, file=sys.stdout, flush=flush)
    try: print(*values, sep=sep, end=end, file=log, flush=flush)
    except NameError: print('WARNING: Trying to use uprint() without an opened log file.')

def intlen(integer : int):
    return int(math.log10(integer))+1


class Progress():
    '''A progress bar, but dirty.'''
    def __init__(self, buffer : int = 20, header : str = '', footer : str = '', jobs : int = 1, steps : int = 100) -> None:
        self.master_start = time.time()
        self.job_times = []
        self.buffer = buffer
        self.footer = '*'*self.buffer + f'*{" " + footer + " ":*^100}*' + '*'*35
        self.jobs = jobs
        self.j = 1
        self.offset = 0
        self.steps_len = intlen(steps) if intlen(steps) > 3 else 4
        h = (
            f'{header:^{self.buffer}}|'+
            ' '*100+
            f'|{"DONE/TODO":^{(self.steps_len*2)+3}}[   %] ~ elapsed ~  eta '
        )
        uprint(h)

    def make_bar(self, steps) -> None:
        self.start = True
        self.start_time = time.time()
        self.steps = steps
        self.prefix = f'Sample {self.j}/{self.jobs}'
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
            if percent == 0: print(bar, end='\r')
            else:
                if percent % 10 == 0:
                    est_total = ((time.time()-self.start_time) * (100/percent) + self.init_time) * self.jobs
                    self.eta = est_total - sum(self.job_times) - (self.init_time+time.time()-self.start_time)
                    m, s = divmod(round(self.eta), 60)
                    bar += f'{f"{m:02d}:{s:02d}":^9}'
                print(bar, end='\r')
        else:
            self.job_times.append(self.init_time+time.time()-self.start_time)
            uprint(bar, end='           \n')
            if self.j == self.jobs: print(self.footer)
            self.j += 1





if __name__ == '__main__':
    print()
    jobs = 4
    duration = 500
    P = Progress(buffer=20, header='head', footer='foot', jobs=jobs)
    for j in range(jobs):
        P.make_bar(duration)
        time.sleep(1)
        for i in range(duration):
            P.bar_step(i)
            time.sleep(0.01)
    time.sleep(1)
    print()