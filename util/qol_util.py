# Just some simple, quality-of-life functions. Nothing very fancy.

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