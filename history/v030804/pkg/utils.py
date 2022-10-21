import os, sys
import time, math

def get_root():
    root = os.path.dirname(os.path.realpath(os.path.dirname(__file__)))
    return root

class ProgressBar():
    '''
    duration = 786
    bar = ProgressBar(start=1, total=duration, bar_length=100, prefix='Machine A:')
    time.sleep(1)
    for i in range(duration):
        bar.update(i)
        time.sleep(0.01)
    '''
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
        dual_print(bar_template)
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
            dual_print(bar+f' ~ {h}:{m:02d}:{s:02d}')
            dual_print('*'*len(self.header) + f'{self.conclusion+conclusion_adjustment:*^{len(self.mid_sec)}}' + '*'*len(self.footer))
            # print('this 2 is ' + conclusion_adjustment)


def dual_print(*values: object, sep: str | None = ' ', end: str | None = '\n', file=None, flush=False):
    if file is None: file = opened 
    print(*values, sep=sep, end=end, file=sys.stdout, flush=flush)
    print(*values, sep=sep, end=end, file=file, flush=flush)

def mk_output_dir() -> str:
    x = 1
    while True:
        output_dir = get_root().replace('\src', '')+f'\output\{x:03d}'
        try:
            # print(x, end='-', flush=True)
            time.sleep(0.1)
            os.mkdir(output_dir)
            # print('success')
            break
        except:
            # print(output_dir)
            # print('failure')
            x += 1
        if x == 50: raise Exception('too many folders')
    print(f'\nOutputting to:\t\t{output_dir}\n')
    return output_dir


class UtilFileLike():
    def __init__(self, path) -> None:
        self.path = path
    def __enter__(self):
        self.file = open(self.path, 'w')
        return self
    def __exit__(self, type, value, traceback):
        self.file.close()

def u_open(path):
    global opened
    opened = open(path, 'w', encoding='utf-8')

def u_close():
    opened.close()

def trunc(number, decimals=0):
    """
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor
