import numpy as np
from itertools import combinations

class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def astar(maze, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []

        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            # Child is on the closed list
            is_closed = False
            for closed_child in closed_list:
                if child == closed_child:
                    is_closed = True
            if is_closed: continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)
def find_four_direction(maze,start_p, end_p):
    four_direction = []
    for new_position in [(0,1),(1,0),(-1,0),(0,-1)]:
        tmp_list = (end_p[0]+new_position[0],end_p[1]+new_position[1])
        four_direction.append(tmp_list)

    reachability = 0
    for end_point in four_direction:

        path = astar(maze,start_p,end_point)
        if path != None:
            reachability = 1
            break
    return reachability

def get_solvability(indarray):

    solvable = True
    player1 = [(i, j) for i in range(5) for j in range(7) if indarray[i][j] == 7][0]
    player2 = [(i, j) for i in range(5) for j in range(7) if indarray[i][j] == 8][0]


    for k in range (2,6):
        #tmp_len = math.ceil(2 - k / 5) # 2,2,2,1을 뽐기 위함
        block_position = [(i, j) for i in range(5) for j in range(7) if indarray[i][j] == k]
        for m in range(len(block_position)):
            reachablilty1 = find_four_direction(indarray, player1, block_position[m])
            reachablilty2 = find_four_direction(indarray, player2, block_position[m])
            if reachablilty1*reachablilty2 != 1:
                solvable = False
                break

    return solvable


def hamming_distance (individual1,individual2):
    distance_value=0
    for j in range(7):
        for k in range(5):
            if individual1[k][j]!=individual2[k][j]:
                distance_value+=1

    return distance_value


def build_hamminglist_3(population):
    a = list(range (len(population)))

    array_size = (len(population), len(population))
    value_array = np.zeros(array_size, dtype=int)

    for i in combinations(a, 2):

        value_array[i[0]][i[1]]=hamming_distance(population[i[0]],population[i[1]])
    row_means = np.mean(value_array, axis=1)

    # 열의 평균 계산
    col_means = np.mean(value_array, axis=0)

    # 행의 평균과 열의 평균의 합 계산
    mean_value = row_means[:, np.newaxis] + col_means

    mask = np.eye(len(population), dtype=bool)
    mean_value[~mask] = 0
    top_N = 50
    idx = np.argpartition(mean_value, mean_value.size - top_N, axis=None)[-top_N:]
    result = np.column_stack(np.unravel_index(idx, mean_value.shape))
    #test_map_idx = result[:, 1]

    return value_array

def build_hamming_list_4(population):
    a = list(range(len(population)))
    train_list = []
    test_list = []

    array_size = (len(population), len(population))
    value_array = np.zeros(array_size, dtype=int)

    for i in combinations(a, 2):
        value_array[i[0]][i[1]] = hamming_distance(population[i[0]], population[i[1]])
    transpose_array = value_array.transpose()
    result_array = value_array+transpose_array

    # 행의 평균 계산
    row_means = np.mean(value_array, axis=1)

    # 열의 평균 계산
    col_means = np.mean(value_array, axis=0)

    # 행의 평균과 열의 평균의 합 계산
    mean_value = row_means[:, np.newaxis] + col_means

    mask = np.eye(len(population), dtype=bool)
    mean_value[~mask] = 0
    top_N = 50
    idx = np.argpartition(mean_value, mean_value.size - top_N, axis=None)[-top_N:]
    result = np.column_stack(np.unravel_index(idx, mean_value.shape))
    test_map_idx = result[:,1]

    #new_arr = np.zeros(([len(population)-top_N, len(population)-top_N]), dtype=float)
    tmp_num = 0
    for i in range(len(population)):
        if i in test_map_idx:
            test_list.append(population[i])
        else:
            train_list.append(population[i])
            tmp_num_2=0
            for j in range(len(population)):
                if j in test_map_idx:
                    pass
                else:
                    #new_arr[tmp_num][tmp_num_2]= result_array[i][j]
                    tmp_num_2+=1
            tmp_num+=1
    # min_val = np.min(new_arr)
    # max_val = np.max(new_arr)
    # normalized_arr = (new_arr - min_val) / (max_val - min_val)

# 여기서 seed 획득
    #return normalized_arr, test_list, train_list
    return 0, test_list, train_list
def input_or_not(population, individual):
    satisfy = 1
    for i in range(len(population)):
        value = hamming_distance(population[i],individual)
        if value ==0:
            satisfy = 0
            break
    return satisfy

def main():
    pass





if __name__ == '__main__':
    main()