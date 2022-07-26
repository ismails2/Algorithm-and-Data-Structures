"""
Sara Ismail
Coding Challenge 3
CSE 331 Spring 2021
Professor Sebnem Onsay
"""
from typing import List


def finding_best_bot(bots_list: List[int]) -> int:
    """
    Finds the length of increasing variables in a list of ints
    param: List of Integers
    return: Int, Length of increasing values
    """

    def finding_best_bot_helper(start: int, end: int):
        """
        Finds the location of the last incrementing value in a list through
        the use of recursion and returns length of increasing values
        param: start, is the first index of the list
        param: end, last index of comparison
        return: int of the length of increasing value
        """

        last_index = list_len - 1
        if end >= list_len:
            end = last_index
        if bots_list[start] > bots_list[end]:
            if end//2 == list_len//2:
                return finding_best_bot_helper(start, end//2 + 1)
            return finding_best_bot_helper(start, end//2)
        if bots_list[end] < bots_list[end-1]:
            return finding_best_bot_helper(start, end//2)
        if end == last_index:
            return end + 1
        if bots_list[end] > bots_list[end + 1]:
            return end + 1
        return finding_best_bot_helper(start, round(end/2) + end)

    if bots_list == []:
        return 0
    list_len = len(bots_list)
    if list_len == 1:
        return 1
    return finding_best_bot_helper(0, list_len//2)

