"""
Name: Sara Ismail
Project 3 - Hybrid Sorting
Developed by Sean Nguyen and Andrew Haas
Based on work by Zosha Korzecke and Olivia Mikola
CSE 331 Spring 2021
Professor Sebnem Onsay
"""

from typing import TypeVar, List, Callable
from copy import deepcopy

T = TypeVar("T")            # represents generic type


def merge(S1: List[T], S2: List[T], S: List[T], num: int, comparator):
    """
    Merges two lists into one and stores them into another list while sorting them
    Param: S1, List of type T, sub list of S
    Param: S2, List of type T, sub list of S
    Param: S, List of type T
    Param: Int, number of current inversions
    Return: Int, number of current inversions
    """
    i = 0
    j = 0
    while i + j < len(S):
        if j == len(S2) or (i < len(S1)) and comparator(S1[i], S2[j]):
            S[i + j] = S1[i]
            i += 1
        else:
            S[i + j] = S2[j]
            j += 1
            num += len(S1) - i
    return num

def merge_sort(data: List[T], threshold: int = 0,
               comparator: Callable[[T, T], bool] = lambda x, y: x <= y) -> int:
    """
    Sorts a list by splitting into two lists and sorting the smaller ones
    Param: data, List of type T
    Param: threshold, int, number at which insertion sort is supposed to be used
    Param: comparator, Callable type, comparison that you are supposed to sort by
    Return: type int, number of inversions used
    """

    n = len(data)
    if n < 2:
        return 0
    if n <= threshold:
        insertion_sort(data, comparator)
    else:
        mid = n // 2
        data1 = data[0:mid]
        data2 = data[mid:n]
        int1 = merge_sort(data1, threshold, comparator)
        int2 = merge_sort(data2, threshold, comparator)
        num = int1 + int2
        num = merge(data1, data2, data, num, comparator)
    if threshold > 0:
        return 0
    return num

def insertion_sort(data: List[T], comparator: Callable[[T, T], bool] = lambda x, y: x <= y) -> None:
    """
    Compares each index to the previous value and swaps location until the list is sorted
    Param: data, List of type T
    Param: threshold, int, number at which insertion sort is supposed to be used
    Param: comparator, Callable type, comparison that you are supposed to sort by
    """
    for i in range(1, len(data)):
        j = i
        while j > 0 and comparator(data[j], data[j-1]):
            temp = data[j]
            data[j] = data[j - 1]
            data[j - 1] = temp
            j -= 1

def hybrid_sort(data: List[T], threshold: int,
                comparator: Callable[[T, T], bool] = lambda x, y: x <= y) -> None:
    """
    Calls merge sort, but does not return a value
    Param: data, List of type T
    Param: threshold, int, number at which insertion sort is supposed to be used
    Param: comparator, Callable type, comparison that you are supposed to sort by
    """
    merge_sort(data, threshold, comparator)

def inversions_count(data: List[T]) -> int:
    """
    Calls merge sort to find the number of inversions without sorting the list
    Param: data, List of type T
    Return: type int, number of inversions used
    """
    data_copy = deepcopy(data)
    result = merge_sort(data_copy)
    return result

def reverse_sort(data: List[T], threshold: int) -> None:
    """
    Calls merge sort, but sorts in reverse order
    Param: data, List of type T
    Param: threshold, int, number at which insertion sort is supposed to be used
    """
    merge_sort(data, threshold, lambda x, y: x > y)

def password_rate(password: str) -> float:
    """
    Finds the rate of a password
    Param: password, string
    Return: int, value of rate
    """
    dict_char = {}
    password_len = len(password)
    part1 = password_len ** (1/2)
    list_pass = []
    for ch in password:
        list_pass.append(ch)
        if ch in dict_char:
            dict_char[ch] += 1
            password_len -= 1
        else:
            dict_char[ch] = 1
    part2 = password_len ** (1/2)
    part3 = inversions_count(list_pass)
    return (part1 * part2) + part3


def password_sort(data: List[str]) -> None:
    """
    Sorts a list of passwords passed on its rate value
    Param: data, List[str]
    """

    list_val = []
    for item in data:
        list_val.append((item, password_rate(item)))
    merge_sort(list_val, 0, lambda x, y: x[1] > y[1])
    count = 0
    for item in list_val:
        data[count] = list_val[count][0]
        count += 1
