"""
Sara Ismail
Coding Challenge 4
CSE 331 Spring 2021
Professor Sebnem Onsay
"""
from typing import List




def challenger_finder(stocks_list: List[int], k: int) -> List[int]:
    """
    finds the stocks in range k and makes a list of number of possible players
    param: list of ints
    param: int
    return: list of ints that represent the number of in range
    """
    def challenger_finder_inner(start, end, val):
        """
        modified binary search
        """
        while start <= end:
            mid_index = (start + end) // 2
            if sort_list[mid_index] > val:
                end = mid_index - 1
            else:
                if mid_index - 1 < 0:
                    return 0
                if not sort_list[mid_index - 1] <= val:
                    return mid_index
                if sort_list[mid_index] < val:
                    start = mid_index + 1
                end = mid_index - 1
        return 0

    result = []

    end_index = len(stocks_list) - 1
    sort_list = sorted(stocks_list)

    for item in stocks_list:
        low = item - k
        high = item + k
        low_index = challenger_finder_inner(0, end_index, low)
        if low_index is None:
            result.append(0)
        else:
            high_index = challenger_finder_inner(low_index, end_index, high)
            if high_index is None:
                result.append(0)
            else:
                result.append(high_index - low_index)

    return result