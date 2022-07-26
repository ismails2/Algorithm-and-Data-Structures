"""
Sara Ismail
Coding Challenge 1 - Love Is In The Air - Solution
CSE 331 Spring 2021
Professor Sebnem Onsay
"""
from typing import List, Tuple
#from collections import Counter
#from itertools import accumulate

def story_progression(answers: List[int], questions: List[Tuple[int, int]]) -> List[str]:
    """
    Determine if each tuple in the question list to check that range of the answer list
    to determine if that chunk results in a win or a loss for the player.
    It will be a win condition if the chunk contains a majority
    :param answers: list of 0’s and 1’s, 1 represents a correct choice, 0 otherwise
    :param questions: list of questions, is a list of tuple of length 2, where
           Element [0] is starting index of the interested range
           Element [1] is ending index of the interested range
    :return: A Python list of the same length as list of question,
            each element is either “Win” or “Lose,”
    """
    result = []
    list_sum = []
    list_sum.append(answers[0])
    for i in range(len(answers)-1):
        list_sum.append(answers[i+1]+list_sum[i])
    for index_tuple in questions:
        start_index = index_tuple[0]
        end_index = index_tuple[1]
        ques_len = end_index - start_index + 1
        if start_index == end_index:
            if answers[start_index] == 1:
                result.append('Win')
            else:
                result.append('Lose')
        elif start_index == 0:
            count_true = list_sum[end_index]
            count_false = ques_len - count_true
            if count_false < count_true:
                result.append('Win')
            else:
                result.append('Lose')
        else:
            count_true = list_sum[end_index] - list_sum[start_index]
            count_false = ques_len - count_true
            if count_false < count_true:
                result.append('Win')
            else:
                result.append('Lose')
    return result
