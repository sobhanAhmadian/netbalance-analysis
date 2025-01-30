import time


def get_tqdm_desc():
    current_time_tuple = time.localtime()
    asctime_string = time.strftime("%Y-%m-%d %H:%M:%S,", current_time_tuple)
    s = str(current_time_tuple.tm_sec).zfill(3)
    asctime_string_with_milliseconds = asctime_string + s
    return "{} [INFO] Progress".format(asctime_string_with_milliseconds)
