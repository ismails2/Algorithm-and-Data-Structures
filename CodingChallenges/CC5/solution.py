"""
Sara Ismail
Coding Challenge 5
CSE 331 Spring 2021
Professor Sebnem Onsay
"""
from typing import List

def look_direction(walls: List[int]) -> List[int]:
    """
    Extra function that can look in one direction to see which walls are visible
    param: List of ints, the walls heights in order
    result: List of ints, the number of walls visible
    """
    look = []
    stack = []
    look.append(0)
    stack.append(walls[0])
    max_left = walls[0]
    for i in range(1, len(walls)):
        if max_left < walls[i]:
            max_left = walls[i]
            stack.clear()
        elif walls[i] >= stack[-1]:
            while walls[i] >= stack[-1]:
                stack.pop()
                if len(stack) == 0:
                    break
        look.append(len(stack))
        stack.append(walls[i])
    return look

def check_walls_cover(walls: List[int]) -> List[int]:
    """
    Finds the amount of walls visible in your location
    param: list of ints, the walls heights in order
    return: list of ints, number of walls that are visible in both directions
    """
    look_left = look_direction(walls)
    walls = walls[::-1]
    look_right = look_direction(walls)
    look_right = look_right[::-1]
    for i in range(len(look_right)):
        look_right[i] = look_right[i] + look_left[i]
    return look_right
