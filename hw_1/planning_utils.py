from state import *
def traverse(goal_state, prev):
    '''
    extract a plan using the result of dijkstra's algorithm
    :param goal_state: the end state
    :param prev: result of dijkstra's algorithm
    :return: a list of (state, actions) such that the first element is (start_state, a_0), and the last is
    (goal_state, None)
    '''
    result = [(goal_state, None)]
    # remove the following line and complete the algorithm

    curr_state = result[0][0]
    next_act = 'init'
    inverter_dic = {'u': 'd', 'd': 'u', 'r': 'l', 'l':'r'}

    while next_act:
        next_act = prev[curr_state.to_string()]
        if next_act:
            curr_state = curr_state.apply_action(next_act)
            result.insert(0, (curr_state, inverter_dic[next_act]))
        #else:
        #    result.insert(0, (curr_state, next_act))
    return result


def print_plan(plan):
    print('plan length {}'.format(len(plan)-1))
    for current_state, action in plan:
        print(current_state.to_string()) # , ', d=',current_state.get_manhattan_distance(State()))
        if action is not None:
            print('apply action {}'.format(action))
