"""
Load text files.
"""
import csv


def load_raw(path):
    """ Load data in the same format as bds_1.txt. """
    with open(path) as f:
        lines = f.read().split("\n")

    ids = []
    bds = []
    for i, p in enumerate(lines):
        if i % 2 == 0:
            ids.append(p)
        else:
            bds.append(p)

    # Original bds_1.txt has a newline at the end of file,
    # so remove this blank line
    if len(lines) % 2 != 0 and ids[-1] == '':
        ids.pop(-1)

    return ids, bds


def write_data(ids, bds, path):
    """ Write some ID/BD pairs to a tab-separate file. """
    # Python docs says newline="" must be specified for csv
    assert len(ids) == len(bds)

    with open(path, "w", newline="") as outfile:
        for iid, bd in zip(ids, bds):
            outfile.write(iid + "\n")
            outfile.write(" ".join(bd) + "\n")
