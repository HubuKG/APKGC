"""
Utilizations for common usages.
"""
import os
import random
import torch
import numpy as np
from difflib import SequenceMatcher
from unidecode import unidecode
from datetime import datetime
from torch.nn.parallel import DataParallel, DistributedDataParallel


def invert_dict(d):
    return {v: k for k, v in d.items()}

def personal_display_settings():

    from pandas import set_option
    set_option('display.max_rows', 500)
    set_option('display.max_columns', 500)
    set_option('display.width', 2000)
    set_option('display.max_colwidth', 1000)
    from numpy import set_printoptions
    set_printoptions(suppress=True)


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize(s):

    s = s.strip().lower()
    s = unidecode(s)
    return s


def snapshot(model, epoch, save_path):

    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f'{type(model).__name__}_{epoch}_epoch.pkl')
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        torch.save(model.module.state_dict(), save_path)
    else:
        torch.save(model.state_dict(), save_path)
    return save_path


def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'models': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, path)


def load_checkpoint(path, map_location):
    checkpoint = torch.load(path, map_location=map_location)
    return checkpoint


def show_params(model):

    for name, param in model.named_parameters():
        print('%-16s' % name, param.size())


def longest_substring(str1, str2):
    seqMatch = SequenceMatcher(None, str1, str2)
    match = seqMatch.find_longest_match(0, len(str1), 0, len(str2))

    return str1[match.a: match.a + match.size] if match.size != 0 else ""


def pad(sent, max_len):

    length = len(sent)
    return (sent + [0] * (max_len - length))[:max_len] if length < max_len else sent[:max_len]


def to_cuda(*args, device=None):

    assert all(torch.is_tensor(t) for t in args), \
        'Only support for tensors, please check if any nn.Module exists.'
    if device is None:
        device = torch.device('cuda:0')
    return [None if x is None else x.to(device) for x in args]


def get_code_version(short_sha=True):
    from subprocess import check_output, STDOUT, CalledProcessError
    try:
        sha = check_output('git rev-parse HEAD', stderr=STDOUT,
                           shell=True, encoding='utf-8')
        if short_sha:
            sha = sha[:7]
        return sha
    except CalledProcessError:
        pwd = check_output('pwd', stderr=STDOUT, shell=True, encoding='utf-8')
        pwd = os.path.abspath(pwd).strip()
        print(f'Working dir {pwd} is not a git repo.')


def cat_ragged_tensors(left, right):
    assert left.size(0) == right.size(0)
    batch_size = left.size(0)
    max_len = left.size(1) + right.size(1)

    len_left = (left != 0).sum(dim=1)
    len_right = (right != 0).sum(dim=1)

    left_seq = left.unbind()
    right_seq = right.unbind()
    output = torch.zeros((batch_size, max_len), dtype=torch.long, device=left.device)
    for i, row_left, row_right, l1, l2 in zip(range(batch_size),
                                              left_seq, right_seq,
                                              len_left, len_right):
        l1 = l1.item()
        l2 = l2.item()
        j = l1 + l2
        row_cat = torch.cat((row_left[:l1], row_right[:l2]))
        output[i, :j] = row_cat
    return output


def topk_accuracy(inputs, labels, k=1, largest=True):
    assert len(inputs.size()) == 2
    assert len(labels.size()) == 2
    _, indices = inputs.topk(k=k, largest=largest)
    result = indices - labels  # boardcast
    nonzero_count = (result != 0).sum(dim=1, keepdim=True)
    num_correct = (nonzero_count != result.size(1)).sum().item()
    num_example = inputs.size(0)
    return num_correct, num_example


def get_total_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print(normalize('ǖǘǚǜ'))
