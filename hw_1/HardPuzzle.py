from dijkstra import *
init = State("8 6 7\r\n2 5 4\r\n3 0 1")
goal = State()

print("*"*100)
print("Dijkstra Solution")
print("*"*100)
hard_puzzle = Puzzle(init, goal)
dik_start_time = datetime.datetime.now()
solve(hard_puzzle)
print('time to solve {}'.format(datetime.datetime.now()-dik_start_time))

from a_star import *
print("*"*100)
print("A Star Solution")
print("*"*100)
hard_puzzle = Puzzle(init, goal)
star_start_time = datetime.datetime.now()
solve(hard_puzzle)
print('time to solve {}'.format(datetime.datetime.now()-star_start_time))
