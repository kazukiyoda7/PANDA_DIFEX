'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import errno
import time
import os.path as osp
import math
from datetime import datetime
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init

# --------------------------------------------------------------------------------------------------------
# 入力した文字列とpathが一致したディレクトリがなければ新しくディレクトリを作成
def check_dir(dir):
	if not os.path.isdir(dir):
		os.makedirs(dir)
		print('New directory created : {}'.format(dir))

# 既にあるディレクトリの上書を防ぐ
def check_filename(filename):
	f_new = filename
	root, ext = os.path.splitext(filename)
	counter = 0
	while os.path.exists(f_new):
		counter += 1
		f_new = root + '_{}'.format(counter) + ext
	return f_new


# ログを外部ファイルに書き出す関数
# Reference : https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py
class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            path = osp.join(fpath, 'log.txt')
            check_dir(os.path.dirname(path))
            self.file = open(path, 'a')
        now = datetime.now()
        dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
        self.write('\n\n')
        self.write(f'[{dt_string}]\n')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    # Write on both console and txt file
    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    # Write on console but not txt file
    def write_console(self, msg):
        self.console.write(msg)

    # Write on txt file but not console
    def write_file(self, msg):
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

# Display status for each epoch
last_time = time.time()
begin_time = last_time
def progress_bar(batch, total, msg=None):
    global last_time, begin_time
    if batch == 0:
        begin_time = time.time()  # Reset for new bar.
    current_time = time.time()
    last_time = current_time
    total_time = current_time - begin_time
    L = []
    L.append('Time consumed: %s' % format_time(total_time))
    if msg:
        L.append(' | ' + msg)
    msg = ''.join(L)
    sys.stdout.write_console(msg)
    # sys.stdout.write_console('\n')
    if batch == total-1:
        sys.stdout.write_file(msg)
        sys.stdout.write_file('\n')
    sys.stdout.flush()

# path = '~~~.pth'
# def save_networks(networks, path):
#     weights = networks.state_dict()
#     torch.save(weights, path)
def save_networks(*args, **kwargs):
    torch.save(*args, **kwargs)



def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
# --------------------------------------------------------------------------------------------------------


def load_networks(networks, result_dir, name='', loss='', criterion=None):
    weights = networks.state_dict()
    filename = '{}'.format(result_dir)
    networks.load_state_dict(torch.load(filename))
    if criterion:
        weights = criterion.state_dict()
        filename = '{}.pth'.format(result_dir)
        criterion.load_state_dict(torch.load(filename))

    return networks, criterion

# --------------------------------------------------------------------------------------------------------
# seed を固定する関数
def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True