from datetime import datetime
import string
import numpy as np

def get_logdir_name():
    """
    Returns a name for the dir
    :returns: a name in the format dd-mm-yyyy_:mm:ss_<random string>
    """
    time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S-%f")
    rand_str = "".join(np.random.choice([*string.ascii_lowercase], 8))
    return f"{time}_{rand_str}"

class colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    ORANGE = '\033[33m'
    RESULT =  '\033[94m'
    WARNING = '\033[93m'
    DEBUGGING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def now():
    return datetime.now().strftime("%H:%M:%S")

def print_error(*args, **kwargs):
    print(now()+" ["+colors.BOLD+colors.FAIL+"ERROR"+colors.ENDC+"] " +" ".join(map(str,args))+"\n", **kwargs)
    
def print_warning(*args, **kwargs):
    print(now()+" ["+colors.BOLD+colors.ORANGE+"WARNING"+colors.ENDC+"] " +" ".join(map(str,args))+ "\n", **kwargs)

def print_debugging(*args, **kwargs):
    print(now()+" ["+colors.BOLD+colors.DEBUGGING+"DEBUG"+colors.ENDC+"] " +" ".join(map(str,args))+"\n", **kwargs)
    
def print_info(*args, **kwargs):
    print(now()+" ["+colors.BOLD+colors.OKGREEN+"INFO"+colors.ENDC+"] "+" ".join(map(str,args))+"\n", **kwargs)

def print_configs(*args, **kwargs):
    print(now()+" ["+colors.BOLD+colors.OKBLUE+"CONFIGS"+colors.ENDC+"] "+" ".join(map(str,args))+"\n", **kwargs)
    
def print_results(*args, **kwargs):
    print(now()+" ["+colors.BOLD+colors.RESULT+"RESULTS"+colors.ENDC+"] "+" ".join(map(str,args))+"\n", **kwargs)