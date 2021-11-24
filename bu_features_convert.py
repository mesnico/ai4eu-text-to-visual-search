from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import base64
import numpy as np
import csv
import sys
import argparse
import tqdm
import pickle

parser = argparse.ArgumentParser()

# output_dir
parser.add_argument('--downloaded_feats', default='data/bu_data', help='downloaded feature directory')
parser.add_argument('--output_dir', default='features', help='output feature files')
parser.add_argument('--in_file', default='')

args = parser.parse_args()

csv.field_size_limit(sys.maxsize)


FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

infiles = [args.in_file]
size_file = args.output_dir + '/sizes.pkl'

if not os.path.exists(args.output_dir+'/bu_att'):
    os.makedirs(args.output_dir+'/bu_att')
    os.makedirs(args.output_dir+'/bu_fc')
    os.makedirs(args.output_dir+'/bu_box')
cocobu_size = {}

for infile in infiles:
    print('Reading ' + infile)
    with open(infile, "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for item in tqdm.tqdm(reader):
            item['image_id'] = item['image_id']
            item['num_boxes'] = int(item['num_boxes'])
            item['image_w'] = int(item['image_w'])
            item['image_h'] = int(item['image_h'])
            if os.path.isfile(os.path.join(args.output_dir+'_att', str(item['image_id'])+'.npz')):
                # print('{} already present'.format(os.path.join(args.output_dir+'_att', str(item['image_id']))))
                continue
            try:
                for field in ['boxes', 'features']:
                    item[field] = np.frombuffer(base64.decodestring(bytes(item[field], 'utf-8')),
                            dtype=np.float32).reshape((item['num_boxes'],-1))
            except:
                print('Error reading {}. Skipping'.format(str(item['image_id'])))
                continue
            np.savez_compressed(os.path.join(args.output_dir+'/bu_att', str(item['image_id'])), feat=item['features'])
            np.save(os.path.join(args.output_dir+'/bu_fc', str(item['image_id'])), item['features'].mean(0))
            np.save(os.path.join(args.output_dir+'/bu_box', str(item['image_id'])), item['boxes'])
            # cocobu_size_temp = {}
            # cocobu_size_temp['image_w'] = item['image_w']
            # cocobu_size_temp['image_h'] = item['image_h']
            cocobu_size[str(item['image_id'])] = (item['image_w'], item['image_h'])
    with open(size_file, 'wb') as f:
        pickle.dump(cocobu_size, f)