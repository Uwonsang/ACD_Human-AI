import cv2

import random

import argparse
import logging
import os

from check_room_count import tmp_render_to_game

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
        self.map_list_test=[]

        self.hamming_list = []
        self.hamming_list_test = []

        self.result_dir = './results/room{0}_seed{1}'.format(self.args.room_count,self.args.seed)

        logger.info('Result path : {0}'.format(self.result_dir))
        random.seed(self.args.seed)


        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(os.path.join(self.result_dir, 'images'), exist_ok=True)

        os.makedirs(os.path.join(self.result_dir, 'layouts'), exist_ok=True)
        os.makedirs(os.path.join(self.result_dir, 'test_images'), exist_ok=True)

        os.makedirs(os.path.join(self.result_dir, 'test_layouts'), exist_ok=True)
        os.makedirs(os.path.join(self.result_dir, 'test2_images'), exist_ok=True)

        os.makedirs(os.path.join(self.result_dir, 'test2_layouts'), exist_ok=True)

    def make_dir(self, save_path):
        # This code will be moved to utils.py in future
        if not os.path.exists(save_path):
            os.makedirs(save_path)


    def main(self):
        photo_dir = self.result_dir+'/photo/'
        self.make_dir(photo_dir)
        file_list = os.listdir(self.result_dir+'/layouts')
        for index, val in enumerate(file_list):
            save_num = val.split('_')[0]
            new_game_image = tmp_render_to_game('./layout/'+val.split('.')[0])

            cv2.imwrite(os.path.join( photo_dir, '{0}_processed.jpg'.format(save_num)), new_game_image)







if __name__ == '__main__':

    parser = argparse.ArgumentParser('Parameter settings for experiment')
    parser.add_argument('--room_count', type=int, required=True, help='Number of rooms in the map')

    parser.add_argument('--seed', type=int, required=True, help='Seed of the experiment')

    args = parser.parse_args()


    workspace = Workspace(args)

    workspace.main()
