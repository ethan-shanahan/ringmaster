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
    except NameError: raise('Trying to use uprint() without an opened log file.')

def ProgressBar():
    '''Currently using old tech. Due refinement.'''
    #! IDEA: Do not print at every update, perhaps only every whole percent or less.
    def __init__(self, start: int, total: int, bar_length=100, heading='heading', prefix='', conclusion='') -> None:
        self.init = time.time()
        self.start = start
        self.total = total
        self.w = int(math.log10(total))+1 # width of number total, for formatting purposes
        self.bar_length = bar_length
        self.prefix = prefix
        prefix_w = len(self.prefix)
        self.header = f'{heading:^{prefix_w+1}}'
        self.mid_sec = f'|{"progress bar":^{bar_length}}|'
        self.footer = f' {"DONE/TODO":^{(self.w*2)+1}} [ p %] ~eta ~ elapsed'
        bar_template = '\033[90m' + self.header + self.mid_sec + self.footer + '\033[00m'
        uprint(bar_template)
        self.bar_template_w = len(bar_template)
        self.conclusion = conclusion

    def update(self, current, conclusion_adjustment='', restart=False):
        # print()
        p = (self.start+current)/self.total
        # print(f'current {current} - {(p*100) % 1 == 0}')
        if p > 1: return
        try: eta = round(((time.time()-self.init)/p)-(time.time()-self.init))
        except ZeroDivisionError: eta = ' - '
        self.blocks = '█' * int(self.bar_length * p)
        self.dots = '·' * int(self.bar_length * (1-p))
        # if not math.isclose(p*100, int(math.log10(self.total))+2): self.dots += '·'
        if (p*100) % 1 != 0: self.dots += '·'
        # if 0 < p < 1:
        #     self.dots += '·'
        # elif 0.5 <= p < 1:
        #     self.blocks += '█'
        bar = f' {self.prefix}|{self.blocks}{self.dots}| {self.start+current:{self.w}d}/{self.total} [{int(p*100):3}%] ~ {eta}s'
        if self.start+current != self.total:
            print(bar, end='    \r')
        elif restart:
            # print('restarting...')
            return
        elif not restart:
            m, s = divmod(round(time.time()-self.init), 60)
            h, m = divmod(m, 60)
            uprint(bar+f' ~ {h}:{m:02d}:{s:02d}')
            uprint('*'*len(self.header) + f'{self.conclusion+conclusion_adjustment:*^{len(self.mid_sec)}}' + '*'*len(self.footer))
            # print('this 2 is ' + conclusion_adjustment)


if __name__ == '__main__':
    print(f'{get_src()=}')