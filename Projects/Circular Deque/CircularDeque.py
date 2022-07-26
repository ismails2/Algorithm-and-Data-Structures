"""
Project 4
CSE 331 S21 (Onsay)
Sara Ismail
CircularDeque.py
"""

from __future__ import annotations
from typing import TypeVar, List
# from re import split as rsplit
# import re

# for more information on typehinting, check out https://docs.python.org/3/library/typing.html
T = TypeVar("T")                                # represents generic type
CircularDeque = TypeVar("CircularDeque")        # represents a CircularDeque object


class CircularDeque:
    """
    Class representation of a Circular Deque
    """

    __slots__ = ['capacity', 'size', 'queue', 'front', 'back']

    def __init__(self, data: List[T] = [], capacity: int = 4):
        """
        Initializes an instance of a CircularDeque
        :param data: starting data to add to the deque, for testing purposes
        :param capacity: amount of space in the deque
        """
        self.capacity: int = capacity
        self.size: int = len(data)

        self.queue: list[T] = [None] * capacity
        self.front: int = None
        self.back: int = None

        for index, value in enumerate(data):
            self.queue[index] = value
            self.front = 0
            self.back = index

    def __str__(self) -> str:
        """
        Provides a string represenation of a CircularDeque
        :return: the instance as a string
        """
        if self.size == 0:
            return "CircularDeque <empty>"

        string = f"CircularDeque <{self.queue[self.front]}"
        current_index = self.front + 1 % self.capacity
        while current_index <= self.back:
            string += f", {self.queue[current_index]}"
            current_index = (current_index + 1) % self.capacity
        return string + ">"

    def __repr__(self) -> str:
        """
        Provides a string represenation of a CircularDeque
        :return: the instance as a string
        """
        return str(self)

    # ============ Modify below ============ #

    def __len__(self) -> int:
        """
        docstring
        """
        return self.size

    def is_empty(self) -> bool:
        """
        docstring
        """
        if self.size == 0:
            return True
        return False

    def front_element(self) -> T:
        """
        docstring
        """
        if self.front is None:
            return None
        return self.queue[self.front]

    def back_element(self) -> T:
        """
        docstring
        """
        if self.back is None:
            return None
        return self.queue[self.back]

    def front_enqueue(self, value: T) -> None:
        """
        docstring
        """
        if self.size == 0:
            self.front = 0
            self.back = 0
        elif self.front == 0:
            self.front = self.capacity - 1
        else:
            self.front -= 1
        self.queue[self.front] = value
        self.size += 1

        self.grow()

    def back_enqueue(self, value: T) -> None:
        """
        docstring
        """
        if self.size == 0:
            self.front = 0
            self.back = 0
        elif self.back == (self.capacity - 1):
            self.back = 0
        else:
            self.back += 1
        self.queue[self.back] = value
        self.size += 1
        self.grow()

    def front_dequeue(self) -> T:
        """
        docstring
        """
        if self.size == 0:
            return None
        answer = self.queue[self.front]
        self.queue[self.front] = None
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        self.shrink()
        return answer

    def back_dequeue(self) -> T:
        """
        docstring
        """
        if self.size == 0:
            return None
        answer = self.queue[self.back]
        self.queue[self.back] = None
        self.back = (self.back - 1) % self.capacity
        self.size -= 1
        self.shrink()
        return answer

    def grow(self) -> None:
        """
        docstring
        """
        if self.size != self.capacity:
            return
        self.capacity *= 2
        old = self.queue
        self.queue = [None] * self.capacity
        index = self.front
        for i in range(self.size):
            self.queue[i] = old[index]
            index = (1 + index) % len(old)
        self.front = 0
        self.back = self.size - 1

    def shrink(self) -> None:
        """
        docstring
        """
        if self.size > (self.capacity / 4) or (self.capacity / 2) < 4:
            return
        queue = self.queue
        self.queue = []
        for i in range(self.front, self.size + self.front):
            self.queue.append(queue[i % self.capacity])
        self.capacity = self.capacity // 2
        self.front = 0
        self.back = self.size - 1
        for j in range(self.capacity - self.size):
            self.queue.append(None)

def try_diff(item: str) -> bool:
    """
    docstring
    """
    try:
        float(item)
        return True
    except:
        return False

def calculate_val(deque: CircularDeque) -> int:
    """
    docstring
    """
    operator = '*/+-^'
    new_deque = CircularDeque()
    while deque.front_element():
        val = deque.front_dequeue()
        if val in operator:
            val1 = new_deque.back_dequeue()
            val2 = new_deque.back_dequeue()
            if val2 == None:
                return 0
            if val == '*':
                new_val = val1 * val2
            elif val == '/':
                new_val = val2 / val1
            elif val == '+':
                new_val = val1 + val2
            elif val == '-':
                new_val = val2 - val1
            elif val == '^':
                new_val = val2 ** val1
            new_deque.back_enqueue(new_val)
        else:
            new_deque.back_enqueue(float(val))
    return new_deque.front_element()



def LetsPassTrains102(infix: str) -> str:
    """
    docstring
    """
    ops = {'*': 3, '/': 3,  # key: operator, value: precedence
           '+': 2, '-': 2,
           '^': 4,
           '(': 0}  # '(' is lowest bc must be closed by ')'

    if not infix:
        return (0, "")
    infix_list = infix.split(' ')
    output_deque = CircularDeque()
    operator_deque = CircularDeque()
    for item in infix_list:
        if item == '':
            continue
        if item.isdecimal() or try_diff(item):
            output_deque.back_enqueue(item)
        elif item[0] == '(':
            count = 0
            for i in item:
                if i != '(':
                    break
                operator_deque.back_enqueue(item[count])
                count += 1
            count2 = 0
            for j in range(len(item)):
                if item[int(len(item)) - j - 1] != ')':
                    break
                if operator_deque.front_element() == '(':
                    operator_deque.front_dequeue()
                else:
                    while operator_deque.back_element() != '(':
                        val = operator_deque.back_dequeue()
                        output_deque.back_enqueue(val)
                    if operator_deque.back_element() == '(':
                        operator_deque.back_dequeue()
                count2 -= 1
            if count2 == 0:
                output_deque.back_enqueue(item[count:])
            else:
                output_deque.back_enqueue(item[count:count2])
        elif item[-1] == ')':
            output_deque.back_enqueue(item[:-1])
            if operator_deque.front_element() == '(':
                operator_deque.front_dequeue()
            else:
                while operator_deque.back_element() != '(':
                    val = operator_deque.back_dequeue()
                    output_deque.back_enqueue(val)
                if operator_deque.back_element() == '(':
                    operator_deque.back_dequeue()
        else:
            back = operator_deque.back_element()
            if back:
                back_p = ops[operator_deque.back_element()]
                item_p = ops[item]

                while (back and back_p > item_p) or \
                        (back_p == item_p and item != '^' and back != '('):
                    val = operator_deque.back_dequeue()
                    output_deque.back_enqueue(val)
                    back = operator_deque.back_element()
                    if not back:
                        break
                    back_p = ops[back]
                    item_p = ops[item]
                if back == '^':
                    val = operator_deque.back_dequeue()
                    output_deque.back_enqueue(val)
            operator_deque.back_enqueue(item)

    while operator_deque.size != 0:
        val = operator_deque.back_dequeue()
        output_deque.back_enqueue(val)


    string = ''
    current_index = output_deque.front
    while current_index < output_deque.back:
        string += output_deque.queue[current_index] + ' '
        current_index = (current_index + 1) % output_deque.capacity
    string += output_deque.queue[current_index]

    result = calculate_val(output_deque)

    return (result, string)
