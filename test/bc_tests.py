dim = (3,3); ndim = len(dim)
cell = (0,0)

index_cell = dict(zip([i for i in range(ndim)], cell))
proaction =  {tuple(index_cell[i] for i in index_cell)}
reaction =   set(tuple(index_cell[i]+1 if n/2 == i 
                  else index_cell[i]-1 if n//2 == i
                  else index_cell[i] 
                      for i in index_cell) 
                  for n in range(ndim*2))

# print(f'{index_cell=}')
# print(f'{proaction=}')
# print(f'{reaction=}')

check = [True, True, False]

print(f'{all(check)=}')