# Just some simple, quality-of-life functions. Nothing very fancy.

import numpy as np

# Print iterations progress.
# Adapted from https://stackoverflow.com/a/34325723.
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
        
# Progress bar with color.
def printProgressBarColor (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    fill_prefix = '\33[31m'
    fill_suffix = '\033[0m'
    prog = iteration/total
    if(prog > 0.33 and prog <= 0.67): fill_prefix = '\33[33m'
    elif(prog > 0.67): fill_prefix = '\33[32m'
    fill = fill_prefix + fill + fill_suffix
    printProgressBar(iteration, total, prefix = prefix, suffix = suffix, decimals = decimals, length = length, fill = fill, printEnd = printEnd)
    return

# Setting a histogram's line and fill color in one go
def SetColor(hist, color, alpha = 0.5):
    hist.SetLineColor(color)
    hist.SetFillColorAlpha(color, alpha)


# Some ROOT/numpy stuff
# Converting from ROOT type names to leaflist decorators.
# Vector decorator will not work, but gives a sensible string
# telling us the depth (how many vectors).
def RTypeConversion(type_string):
    if(type_string == 'Short_t' or type_string == 'short'):    return 'S'
    elif(type_string == 'Int_t' or type_string == 'int'):    return 'I'
    elif(type_string == 'Float_t' or type_string == 'float'):  return 'F'
    elif(type_string == 'Double_t' or type_string == 'double'): return 'D'
    elif('vector' in type_string): # special case
#         type_substring = '<'.join(type_string.split('<')[1:])
#         type_substring = '>'.join(type_substring.split('>')[:-1])
#         type_substring = RTypeConversion(type_substring)
#         return 'v_' + type_substring
        return type_string
    else: return '?'

def GetShape(shape_string):
    dims = shape_string.replace('[',' ').replace(']', ' ').split()
    return tuple([int(x) for x in dims])

def RType2NType(type_string):
    if(type_string == 'S'):   return np.dtype('i2')
    elif(type_string == 'I'): return np.dtype('i4')
    elif(type_string == 'L'): return np.dtype('i8')
    elif(type_string == 'F'): return np.dtype('f4')
    elif(type_string == 'D'): return np.dtype('f8')
    else: raise ValueError('Input not understood.')
