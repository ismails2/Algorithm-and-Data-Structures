"""
Name: Sara Ismail
Coding Challenge 6
CSE 331 Spring 2021
Professor Sebnem Onsay
"""
from queue import SimpleQueue

def gates_needed(departures, arrivals):
    """
    This functions finds the maximum number of gates needed in a particular day
    Param: departures, list of ints representing the times planes leave
    Param: arrivals, list of ints representing the times planes arrive
    Return: number of gates, int representing the max gates needed
    """
    max_num = 0
    temp_num = 0
    queue = SimpleQueue()

    for arr in arrivals:
        if len(departures) == 0:
            if len(arrivals) != 0:
                temp_num = len(arrivals) + queue.qsize()
            break
        if arr > departures[0]:
            queue.get()
            departures = departures[1:]
        queue.put(arr)
        arrivals = arrivals[1:]
        not_empty = queue.qsize() != 0 and len(departures) != 0
        if not_empty and arr == departures[0]:
            queue.get()
            departures = departures[1:]
        queue_len = queue.qsize()
        if max_num < queue_len:
            max_num = queue_len

    if max_num < temp_num:
        max_num = temp_num

    return max_num

