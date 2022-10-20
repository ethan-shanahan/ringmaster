class test():
    def __init__(self, a) -> None:
        self.a = a
    
    def run(self, b):
        print(f'{self.a+b}')

t = test('123')
t.run('456')
t.run('321')
