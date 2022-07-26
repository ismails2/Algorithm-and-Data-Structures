"""
Project 1
CSE 331 S21 (Onsay)
Sara Ismail
DLL.py
"""

from typing import TypeVar, List, Tuple
import datetime

# for more information on typehinting, check out https://docs.python.org/3/library/typing.html
T = TypeVar("T")            # represents generic type
Node = TypeVar("Node")      # represents a Node object (forward-declare to use in Node __init__)

# pro tip: PyCharm auto-renders docstrings (the multiline strings under each function definition)
# in its "Documentation" view when written in the format we use here. Open the "Documentation"
# view to quickly see what a function does by placing your cursor on it and using CTRL + Q.
# https://www.jetbrains.com/help/pycharm/documentation-tool-window.html


class Node:
    """
    Implementation of a doubly linked list node.
    Do not modify.
    """
    __slots__ = ["value", "next", "prev"]

    def __init__(self, value: T, next: Node = None, prev: Node = None) -> None:
        """
        Construct a doubly linked list node.

        :param value: value held by the Node.
        :param next: reference to the next Node in the linked list.
        :param prev: reference to the previous Node in the linked list.
        :return: None.
        """
        self.next = next
        self.prev = prev
        self.value = value

    def __repr__(self) -> str:
        """
        Represents the Node as a string.

        :return: string representation of the Node.
        """
        return str(self.value)

    def __str__(self) -> str:
        """
        Represents the Node as a string.

        :return: string representation of the Node.
        """
        return str(self.value)


class DLL:
    """
    Implementation of a doubly linked list without padding nodes.
    Modify only below indicated line.
    """
    __slots__ = ["head", "tail", "size"]

    def __init__(self) -> None:
        """
        Construct an empty doubly linked list.

        :return: None.
        """
        self.head = self.tail = None
        self.size = 0

    def __repr__(self) -> str:
        """
        Represent the DLL as a string.

        :return: string representation of the DLL.
        """
        result = ""
        node = self.head
        while node is not None:
            result += str(node)
            if node.next is not None:
                result += " <-> "
            node = node.next
        return result

    def __str__(self) -> str:
        """
        Represent the DLL as a string.

        :return: string representation of the DLL.
        """
        return repr(self)

    # MODIFY BELOW #

    def empty(self) -> bool:
        """
        Return boolean indicating whether DLL is empty.

        Suggested time & space complexity (respectively): O(1) & O(1).

        :return: True if DLL is empty, else False.
        """
        if self.head and self.size and self.tail:
            return False
        return True

    def push(self, val: T, back: bool = True) -> None:
        """
        Create Node containing `val` and add to back (or front) of DLL. Increment size by one.

        Suggested time & space complexity (respectively): O(1) & O(1).

        :param val: value to be added to the DLL.
        :param back: if True, add Node containing value to back (tail-end) of DLL;
            if False, add to front (head-end).
        :return: None.
        """
        new_node = Node(val)
        self.size += 1
        if self.head is None:
            self.head = new_node
            self.tail = new_node
            return
        if back:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node
        else:
            self.head.prev = new_node
            new_node.next = self.head
            self.head = new_node

    def pop(self, back: bool = True) -> None:
        """
        Remove Node from back (or front) of DLL. Decrement size by 1. If DLL is empty, do nothing.

        Suggested time & space complexity (respectively): O(1) & O(1).

        :param back: if True, remove Node from (tail-end) of DLL;
            if False, remove from front (head-end).
        :return: None.
        """

        if self.head is None:
            return
        if self.size == 1:
            new_tail = self.tail.prev
            self.tail = new_tail
            self.head = self.tail
            self.size -= 1
            return
        if back:
            new_tail = self.tail.prev
            self.tail = new_tail
            self.tail.next = None
        else:
            new_head = self.head.next
            self.head = new_head
            self.head.prev = None
        self.size -= 1

    def from_list(self, source: List[T]) -> None:
        """
        Construct DLL from a standard Python list.

        Suggested time & space complexity (respectively): O(n) & O(n).

        :param source: standard Python list from which to construct DLL.
        :return: None.
        """

        if source:
            for item in source:
                self.push(item)

    def to_list(self) -> List[T]:
        """
        Construct standard Python list from DLL.

        Suggested time & space complexity (respectively): O(n) & O(n).

        :return: standard Python list containing values stored in DLL.
        """

        result = []
        size_dll = self.size
        temp_node = self.head
        if self.head is None:
            return result
        for index in range(size_dll):
            result.append(temp_node.value)
            if index != (size_dll - 1):
                temp_node = temp_node.next
        return result

    def find(self, val: T) -> Node:
        """
        Find first instance of `val` in the DLL and return associated Node object.

        Suggested time & space complexity (respectively): O(n) & O(1).

        :param val: value to be found in DLL.
        :return: first Node object in DLL containing `val`.
            If `val` does not exist in DLL, return None.
        """

        temp_head = self.head
        if not temp_head:
            return None
        while temp_head:
            if temp_head.value == val:
                return temp_head
            temp_head = temp_head.next
        return None

    def find_all(self, val: T) -> List[Node]:
        """
        Find all instances of `val` in DLL and return Node objects in standard Python list.

        Suggested time & space complexity (respectively): O(n) & O(n).

        :param val: value to be searched for in DLL.
        :return: Python list of all Node objects in DLL containing `val`.
            If `val` does not exist in DLL, return empty list.
        """

        result = []
        temp_head = self.head
        if not temp_head:
            return result
        while temp_head:
            if temp_head.value == val:
                result.append(temp_head)
            temp_head = temp_head.next
        return result

    def delete(self, val: T) -> bool:
        """
        Delete first instance of `val` in the DLL.

        Suggested time & space complexity (respectively): O(n) & O(1).

        :param val: value to be deleted from DLL.
        :return: True if Node containing `val` was deleted from DLL; else, False.
        """

        if self.head is None:
            return False
        if self.size == 1 and self.head.value == val:
            self.pop()
            return True
        if self.head.value == val:
            self.pop(False)
            return True
        if self.tail.value == val:
            self.pop()
            return True
        temp_node = self.head
        while temp_node:
            if temp_node.value == val:
                next_node = temp_node.next
                prev_node = temp_node.prev
                prev_node.next = next_node
                next_node.prev = prev_node
                self.size -= 1
                return True
            temp_node = temp_node.next
        return False

    def delete_all(self, val: T) -> int:
        """
        Delete all instances of `val` in the DLL.

        Suggested time & space complexity (respectively): O(n) & O(1).

        :param val: value to be deleted from DLL.
        :return: integer indicating the number of Nodes containing `val` deleted from DLL;
                 if no Node containing `val` exists in DLL, return 0.
        """

        count = 0
        if self.head is None:
            return count
        if self.size == 1 and self.head.value == val:
            self.pop()
            count += 1
            return count
        temp_node = self.head
        while temp_node:
            if temp_node.value == val:
                if self.size == 1:
                    self.pop()
                elif self.head.value == val:
                    self.pop(False)
                elif self.tail.value == val:
                    self.pop()
                else:
                    next_node = temp_node.next
                    prev_node = temp_node.prev
                    prev_node.next = next_node
                    next_node.prev = prev_node
                    self.size -= 1
                count += 1
            temp_node = temp_node.next
        return count



    def reverse(self) -> None:
        """
        Reverse DLL in-place by modifying all `next` and `prev` references of Nodes in the
        DLL and resetting the `head` and `tail` references.
        Must be implemented in-place for full credit. May not create new Node objects.

        Suggested time & space complexity (respectively): O(n) & O(1).

        :return: None.
        """

        if self.head is None:
            return
        elif self.size == 1:
            return

        cur_node = self.head
        holder = self.head

        while cur_node:
            holder = cur_node.prev
            cur_node.prev = cur_node.next
            cur_node.next = holder
            cur_node = cur_node.prev

        head_node = self.head
        self.head = self.tail
        self.tail = head_node


class Stock:
    """
    Implementation of a stock price on a given day.
    Do not modify.
    """

    __slots__ = ["date", "price"]

    def __init__(self, date: datetime.date, price: float) -> None:
        """
        Construct a stock.

        :param date: date of stock.
        :param price: the price of the stock at the given date.
        """
        self.date = date
        self.price = price

    def __repr__(self) -> str:
        """
        Represents the Stock as a string.

        :return: string representation of the Stock.
        """
        return f"<{str(self.date)}, ${self.price}>"

    def __str__(self) -> str:
        """
        Represents the Stock as a string.

        :return: string representation of the Stock.
        """
        return repr(self)


def intellivest(stocks: DLL) -> Tuple[datetime.date, datetime.date, float]:
    """
    Given a DLL representing daily stock prices,
    find the optimal streak of days over which to invest.
    To be optimal, the streak of stock prices must:

        (1) Be strictly increasing, such that the price of the stock on day i+1
        is greater than the price of the stock on day i, and
        (2) Have the greatest total increase in stock price from
        the first day of the streak to the last.

    In other words, the optimal streak of days over which to invest is the one over which stock
    price increases by the greatest amount, without ever going down (or staying constant).

    Suggested time & space complexity (respectively): O(n) & O(1).

    :param stocks: DLL with Stock objects as node values, as defined above.
    :return: Tuple with the following elements:
        [0]: date: The date at which the optimal streak begins.
        [1]: date: The date at which the optimal streak ends.
        [2]: float: The (positive) change in stock price between the start and end
                dates of the streak.
    """

    if stocks.head is None:
        return (None, None, 0)

    temp_node = stocks.head
    temp_list = []
    temp_list.append(stocks.head.value)
    max_list = temp_list
    stock_size = stocks.size
    for i in range(stock_size-1):
        cur_price = temp_node.value.price
        next_price = temp_node.next.value.price
        if cur_price < next_price:
            temp_list.append(temp_node.next.value)
        else:
            start = max_list[0]
            end = max_list[-1]
            change_price = end.price - start.price
            temp_change_price = 0
            if temp_list:
                temp_start = temp_list[0]
                temp_end = temp_list[-1]
                temp_change_price = temp_end.price - temp_start.price
            if temp_change_price > change_price:
                max_list = temp_list
            temp_list = []
            temp_list.append(temp_node.next.value)
        if i != (stock_size - 1):
            temp_node = temp_node.next
    start = max_list[0]
    end = max_list[-1]
    change_price = end.price - start.price
    result = (start.date, end.date, change_price)
    return result
