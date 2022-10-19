import time, math

class ProgressBar():
    '''
    duration = 786
    bar = ProgressBar(duration-1, prefix='Machine A:')
    time.sleep(1)
    for i in range(duration):
        bar.update(i)
        time.sleep(0.01)
    '''
    def __init__(self, total: int, bar_length=100, prefix='') -> None:
        self.init = time.time()
        self.total = total
        self.bar_length = bar_length
        self.w = int(math.log10(total))+1 # width of number total, for formatting purposes
        self.prefix = prefix

    def update(self, current):
        # print()
        p = current/self.total
        if p > 1: return
        try:
            eta = round(((time.time()-self.init)/p)-(time.time()-self.init))
        except ZeroDivisionError:
            eta = ' - '
        self.blocks = '█' * int(self.bar_length * p)
        self.dots = '·' * int(self.bar_length * (1-p))
        if p < 0.5 and p != 0:
            self.dots += '·'
        elif p >= 0.5 and p != 1:
            self.blocks += '█'
        bar = f' {self.prefix}|{self.blocks}{self.dots}| {current:{self.w}d}/{self.total} [{p:4.0%}] ~ {eta}s'
        if current != self.total:
            print(bar, end='    \r')
        else:
            m, s = divmod(round(time.time()-self.init), 60)
            print(bar, end=f' ~ {m}:{s:02d}\n')


