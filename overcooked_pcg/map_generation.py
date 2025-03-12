import cv2
import random
import numpy as np
import argparse
import logging
import os
from copy import deepcopy
from utils.check_room_count import RoomFinder, visualize_room, render_to_game, convert_to_layout, player_position
from utils.postprocessing import remove_unusable_parts
import utils.reachability_modified as reachability_modified
import tqdm

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Workspace(object):
    def __init__(self, args):
        self.args = args
        logger.info(self.args)
        self.corners = [(0, 0), (0, 6), (4, 0), (4, 6)]

        self.indices = [(i, j) for i in range(0, 5) for j in range(0, 7)]
        for i in self.indices:
            if i in self.corners:
                self.indices.remove(i)

        self.map_list = []
        self.hamming_list = []
        self.result_dir = './results/room{0}_seed{1}'.format(self.args.room_count,self.args.seed)

        logger.info('Result path : {0}'.format(self.result_dir))
        random.seed(self.args.seed)


        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(os.path.join(self.result_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.result_dir, 'layouts'), exist_ok=True)
        os.makedirs(os.path.join(self.result_dir, 'test_images'), exist_ok=True)
        os.makedirs(os.path.join(self.result_dir, 'test_layouts'), exist_ok=True)


    def rand_position(self,num_items,layout):
        # random.choice 를 통하여 블록 타입 지정

        while True:
            tmp_break = True
            tmp_layout = deepcopy(layout)
            base_arr = np.array([2,3,4,5]).astype(int) #게임을 진행하기 위한 최소한의 블록 배치

            while True:
                item_arr = np.random.choice(5, size=num_items-4, replace = True)
                counts = np.unique(item_arr, return_counts=True)[1]

                if max(counts) < 2: # 특정 블록 수가 너무 많아지면 다시 블록 타입을 지정
                    break

            item_arr = item_arr + 1

            new_arr = np.concatenate((base_arr, item_arr))
            sorted_item_list = np.sort(new_arr)
            selected_indices = np.random.choice(len(self.indices), size=num_items, replace=False) #지정된 블록을 배치할 좌표 설정
            coordinates = [self.indices[i] for i in selected_indices]
            for i in range(num_items):
                x, y = coordinates[i]
                tmp_layout[x][y] = sorted_item_list[i]

            new_layout_0, removed_list_0 = remove_unusable_parts(tmp_layout) #닫지 않는 블록 제거
            new_layout, removed_list = remove_unusable_parts(new_layout_0) #한번 더 제거

            final_removed_list = np.concatenate((removed_list_0,removed_list))
            for i in final_removed_list:
                indices = np.where(sorted_item_list == i)[0]
                sorted_item_list = np.delete(sorted_item_list, indices[0])
                if i not in sorted_item_list:
                    tmp_break =False
                    break

            if tmp_break == False:
                continue

            rooms = RoomFinder(new_layout).get_rooms()

            if len(rooms)!= 1:
                continue

            if len(new_layout[new_layout == 0]) < 15: #빈 공간이 너무 적으면 다시 layout 생성성
                continue

            return new_layout

    def run(self):

        tmp_index = 0
        for k in range(100): ##map_check

            individual = np.zeros((5, 7), dtype=int)
            for i in [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (1, 0), (1, 6), (2, 0), (2, 6), (3, 0), (3, 6), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6)]:
                individual[i[0]][i[1]] = 1 # 구석자리가 모두 1인 layout 생성

            number = np.random.randint(6, 9) # interactive block 수 지정
            new_individual = self.rand_position(number, individual) #해당 수만큼 블록 생성 및 배치

            if len(self.hamming_list) == 0:
                self.hamming_list.append(new_individual)
                tmp_index += 1

            else:
                if reachability_modified.input_or_not(self.hamming_list, new_individual) == 1: #hamming distance 계산을 통한 중복 검사
                    self.hamming_list.append(new_individual) #중복이 되지 않는 layout은 hamming list에 포함함
                    tmp_index += 1

        for j in range(len(self.hamming_list) - tmp_index, len(self.hamming_list)):
            hamming_map = self.hamming_list[j]
            hamming_map = player_position(hamming_map)

            if reachability_modified.get_solvability(hamming_map) == 1: #player 배치 이후 각 플레이어가 모든 interactive block에 도달할 수 있는지 판별
                self.map_list.append(hamming_map)
                print('producing {}th map'.format(len(self.map_list)))

            if len(self.map_list) == args.num_map + 50: #train layout 6000개 test layout 50개 완성까지 반복
                break

    def modyfing(self):

        result_list =[os.path.join(self.result_dir, 'test_layouts'), os.path.join(self.result_dir, 'layouts')]
        for layout_dir in result_list:
            layout_list = os.listdir(layout_dir)

            for layout in layout_list:
                if layout == 'modify.py':
                    continue

                new_lsit = []
                layout = os.path.join(layout_dir, layout)
                ## TODO testmap
                print(layout)
                with open(layout, 'r') as f:
                    sample = f.read()
                    for i in range(len(sample)):
                        new_lsit.append(sample[i])
                    new_lsit[121] = ''
                    # 7*5 사이즈일시 121로 변경

                result = ""
                for s in new_lsit:
                    result += s

                with open(layout, 'w') as f:
                    f.write(result)
                f.close()

    def main(self):

        while len(self.map_list) < args.num_map + 50:
            self.run()

        # hamming distance metric, test layout, train layout
        train_hamming_array, test_list, train_list = reachability_modified.build_hamming_list_4(self.map_list)

        np.save(self.result_dir+'/hamming_distance array',train_hamming_array)

        index_1 = 0
        index_2 = args.num_map

        for i in range(len(train_list)):

            tmp_map = train_list[i]
            with open(os.path.join(self.result_dir, 'layouts', '{0}_processed.layout'.format(index_1)),'w') as f:
                layout = convert_to_layout(tmp_map.reshape(5, 7))
                f.write(layout)

            player1 = [(i, j) for i in range(5) for j in range(7) if tmp_map[i][j] == 7][0]
            player2 = [(i, j) for i in range(5) for j in range(7) if tmp_map[i][j] == 8][0]

            tmp_map[player1[0], player1[1]] = 0
            tmp_map[player2[0], player2[1]] = 0

            new_game_image = render_to_game(tmp_map.reshape(5, 7)) # render to game 확인
            cv2.imwrite(os.path.join(self.result_dir, 'images', '{0}_processed.jpg'.format(index_1)), new_game_image)

            index_1 += 1

        for i in range(len(test_list)):

            tmp_map = test_list[i]
            with open(os.path.join(self.result_dir, 'test_layouts', '{0}_processed.layout'.format(index_2)), 'w') as f:
                layout = convert_to_layout(tmp_map.reshape(5, 7))
                f.write(layout)
            player1 = [(i, j) for i in range(5) for j in range(7) if tmp_map[i][j] == 7][0]
            player2 = [(i, j) for i in range(5) for j in range(7) if tmp_map[i][j] == 8][0]

            tmp_map[player1[0], player1[1]] = 0
            tmp_map[player2[0], player2[1]] = 0

            new_game_image = render_to_game(tmp_map.reshape(5, 7)) # render to game 확인
            cv2.imwrite(os.path.join(self.result_dir, 'test_images', '{0}_processed.jpg'.format(index_2)), new_game_image)

            index_2 += 1

        self.modyfing()


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Parameter settings for experiment')
    parser.add_argument('--room_count', type=int, default=1, help='Number of rooms in the map')
    parser.add_argument('--seed', type=int, default=123, help='Seed of the experiment')
    parser.add_argument('--num_map', type=int, default=100, help='Number of training map')
    args = parser.parse_args()
    workspace = Workspace(args)
    workspace.main()
