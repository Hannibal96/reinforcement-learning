from puzzle import *
from planning_utils import *
import heapq
import datetime


def a_star(puzzle):
    '''
    apply a_star to a given puzzle
    :param puzzle: the puzzle to solve
    :return: a dictionary mapping state (as strings) to the action that should be taken (also a string)
    '''

    # general remark - to obtain hashable keys, instead of using State objects as keys, use state.as_string() since
    # these are immutable.

    initial = puzzle.start_state
    goal = puzzle.goal_state

    # this is the heuristic function for of the start state
    initial_to_goal_heuristic = initial.get_manhattan_distance(goal)

    # the fringe is the queue to pop items from
    fringe = [(initial_to_goal_heuristic, initial)]
    # concluded contains states that were already resolved
    concluded = set()
    # a mapping from state (as a string) to the currently minimal distance (int).
    distances = {initial.to_string(): 0}
    # the return value of the algorithm, a mapping from a state (as a string) to the state leading to it (NOT as string)
    # that achieves the minimal distance to the starting state of puzzle.
    prev = {initial.to_string(): None}

    ##################################################
    #fringe = []
    #heapq.heappush(fringe, (0, initial))
    inverter_dic = {'u': 'd', 'd': 'u', 'r': 'l', 'l': 'r'}
    done = False
    while len(fringe) > 0:
        curr_h, curr_state = heapq.heappop(fringe)
        curr_d = distances[curr_state.to_string()]
        valid_act = curr_state.get_actions()
        for act in valid_act:
            new_fringe_state = curr_state.apply_action(act)
            new_h = new_fringe_state.get_manhattan_distance(goal)
            #new_h = new_fringe_state.get_hamming_distance(goal)
            if new_fringe_state.to_string() in concluded:
                continue
            heapq.heappush(fringe, (new_h+curr_d+1, new_fringe_state))
            distances[new_fringe_state.to_string()] = curr_d+1
            prev[new_fringe_state.to_string()] = inverter_dic[act]
            if new_fringe_state.to_string() == goal.to_string():
                done = True
                break
        if done:
            break
        concluded.add(curr_state.to_string())

    return prev


def solve(puzzle):
    # compute mapping to previous using dijkstra
    prev_mapping = a_star(puzzle)
    print("Open states:", len(prev_mapping))
    # extract the state-action sequence
    plan = traverse(puzzle.goal_state, prev_mapping)
    print_plan(plan)
    return plan


if __name__ == '__main__':
    # we create some start and goal states. the number of actions between them is 25 although a shorter plan of
    # length 19 exists (make sure your plan is of the same length)

    initial_state = State()
    actions = [
        'r', 'r', 'd', 'l', 'u', 'l', 'd', 'd', 'r', 'r', 'u', 'l', 'd', 'r', 'u', 'u', 'l', 'd', 'l', 'd', 'r', 'r',
        'u', 'l', 'u'
    ]
    goal_state = initial_state
    for a in actions:
        goal_state = goal_state.apply_action(a)
    puzzle = Puzzle(initial_state, goal_state)
    print('original number of actions:{}'.format(len(actions)))
    solution_start_time = datetime.datetime.now()
    solve(puzzle)
    print('time to solve {}'.format(datetime.datetime.now()-solution_start_time))
