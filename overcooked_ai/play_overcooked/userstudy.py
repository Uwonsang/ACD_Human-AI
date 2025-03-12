import sys
import pygame
import numpy as np
import time

def get_keyboard_input():
    action = [4, 4]
    events = pygame.event.get()

    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:  # up
                action[0] = 0
            if event.key == pygame.K_s:  # down
                action[0] = 1
            if event.key == pygame.K_a:  # left
                action[0] = 3
            if event.key == pygame.K_d:  # right
                action[0] = 2
            if event.key == pygame.K_LSHIFT:  # pickup
                action[0] = 5

            if event.key == pygame.K_UP:  # up
                action[1] = 0
            if event.key == pygame.K_DOWN:  # down
                action[1] = 1
            if event.key == pygame.K_LEFT:  # left
                action[1] = 3
            if event.key == pygame.K_RIGHT:  # right
                action[1] = 2
            if event.key == pygame.K_RSHIFT:  # pickup
                action[1] = 5

        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # time.sleep(0.145) ##400_timestep
    return action


def from_charlist_to_int_nparray(arr, p_position):
    arr = np.array(arr, dtype=object)
    arr[(p_position[0][1], p_position[0][0])] = '1'
    arr[(p_position[1][1], p_position[1][0])] = '2'
    # arr[arr ==' '] = '0'
    # arr[arr =='X'] = '9'
    # arr[arr =='P'] = '4'
    # arr[arr =='D'] = '5'
    # arr[arr =='O'] = '3'
    # arr[arr =='S'] = '7'
    # arr = arr.astype(np.int)
    return arr
