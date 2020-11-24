import time


def elapsed_time(func):
  def wrapper(*args, **kwargs):
    begin = time.time()
    func(*args, **kwargs)
    end = time.time()
    print('Elapsed time[{}]: {} sec'.format(func.__name__, (end - begin)))
    # print('Return[{}]: {}'.format(func.__name__, ret_obj))
  return wrapper