import uproot as ur
import awkward as ak
import numpy as np


def loadBranchFlat(branchName, tree, padLength):
    branchInfo = tree[branchName].array()

    # we flatten the 0 index, the event index, to clear a listof clusters
    # we flatten the final 2 index, which is a dummy uproot index
    branchFlat = ak.flatten(ak.flatten(branchInfo, axis = 0), axis = 2)

    # pad the cell axis to the specified length
    branchFlatPad = ak.pad_none(branchFlat, padLength, axis=1)

    # # Do a deep copy to numpy so that the data is owned by numpy
    branchFlatNumpy = np.copy(branchFlatPad.to_numpy())

    # #replace the padding 'None' with 0's
    branchFlatNumpy[-1] = 0

    return branchFlatNumpy
