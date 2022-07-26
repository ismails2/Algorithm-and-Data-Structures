"""
Project 1
CSE 331 S21 (Onsay)
Your Name
DLL.py
"""

from Project2.Node import Node       # Import `Node` class
from typing import TypeVar  # For use in type hinting

# Type Declarations
T = TypeVar('T')        # generic type
SLL = TypeVar('SLL')    # forward declared


class RecursiveSinglyLinkList:
    """
    Recursive implementation of an SLL
    """

    __slots__ = ['head']

    def __init__(self) -> None:
        """
        Initializes an `SLL`
        :return: None
        """
        self.head = None

    def __repr__(self) -> str:
        """
        Represents an `SLL` as a string
        """
        return self.to_string(self.head)

    def __str__(self) -> str:
        """
        Represents an `SLL` as a string
        """
        return self.to_string(self.head)

    def __eq__(self, other: SLL) -> bool:
        """
        Overloads `==` operator to compare SLLs
        :param other: right hand operand of `==`
        :return: `True` if equal, else `False`
        """
        comp = lambda n1, n2: n1 == n2 and (comp(n1.next, n2.next) if (n1 and n2) else True)
        return comp(self.head, other.head)

# ============ Modify below ============ #

    def to_string(self, curr: Node) -> str:
        """
        Converts a SLL into a string
        param: Node (head of SLL)
        return: string of the full linked list
        Time complexity: O(n^2)
        """
        if self.head is None:
            return 'None'
        if curr is self.head:
            result = str(curr.val)
            curr = curr.next
        else:
            result = ' --> ' + str(curr.val)
            curr = curr.next
        if curr is None:
            return result
        result += self.to_string(curr)
        return result

    def length(self, curr: Node) -> int:
        """
        Calculates the length of the SLL
        param: Node (head of SLL)
        return: Int equal to the length of the SLL
        Time complexity: O(n)
        """
        if self.head is None:
            return 0
        length = 1
        curr = curr.next
        if curr is None:
            return length
        length += self.length(curr)
        return length

    def sum_list(self, curr: Node) -> T:
        """
        Calculates the sum of all of the items in the SLL
        param: Node (head of SLL)
        return: Type T (All values added together)
        Time complexity: O(n)
        """
        if self.head is None:
            return 0
        sum = curr.val
        curr = curr.next
        if curr is None:
            return sum
        sum += self.sum_list(curr)
        return sum

    def push(self, value: T) -> None:
        """
        Adds a node to the end of the SLL with value T
        Calls push_inner
        param: value of type T
        Time complexity: O(n)
        """

        def push_inner(curr: Node) -> None:
            """
            Adds the node to end of SLL
            param: value type T
            Time complexity: O(n)
            """
            if curr.next is None:
                curr.next = new_tail
                return
            else:
                curr = curr.next
                push_inner(curr)

        new_tail = Node(value)
        if self.head is None:
            self.head = new_tail
            return
        curr = self.head
        push_inner(curr)

    def remove(self, value: T) -> None:
        """
        Removes first node with a specific value
        param: value type T
        Calls remove inner
        Time complexity: O(n)
        """

        def remove_inner(curr: Node) -> Node:
            """
            Finds value and removes that node
            param: Node
            Time complexity: O(n)
            """
            if curr.next is None:
                return None
            elif curr.next.val == value:
                curr.next = curr.next.next
                return None
            else:
                remove_inner(curr.next)

        if self.head is None:
            return self.head
        if self.head.val == value:
            self.head = self.head.next
            return None
        remove_inner(self.head)

    def remove_all(self, value: T) -> None:
        """
        Removes all nodes with value T
        param: value type T
        Time complexity: O(n)
        """

        def remove_all_inner(curr):
            """
            finds all values in SLL and removes them
            param: current Node in SLL
            Time complexity: O(n)
            """
            if curr.next is None:
                return None
            if curr.next is self.head:
                if curr.next.val == value:
                    self.head = curr.next.next
                    curr.next = curr.next.next
            if curr.next.val == value:
                curr.next = curr.next.next
            remove_all_inner(curr.next)

        curr = self.head
        if self.head is None:
            return None
        if self.head.val == value:
            self.head = self.head.next
        remove_all_inner(curr)

    def search(self, value: T) -> bool:
        """
        Searches for a specific value
        param: value type T
        return: Bool is value is in SLL will return true
        Time complexity: O(n)
        """

        def search_inner(curr):
            """
            Finds first instance of value
            param: Node
            Time complexity: O(n)
            """
            if curr is None:
                return False
            if curr.val == value:
                return True
            else:
                curr = curr.next
                result = search_inner(curr)
                return result

        if self.head is None:
            return False
        if self.head.val == value:
            return True

        curr = self.head
        result = search_inner(curr)
        return result

    def count(self, value: T) -> int:
        """
        Counts the number of times a value is in the SLL
        param: value type T
        return: int of number of times the value is found
        calls count_inner
        Time complexity: O(n)
        """

        def count_inner(curr):
            """
            finds each time a value is in SLL
            param: current node
            Time complexity: O(n)
            """
            if curr is None:
                return 0
            if curr.val == value:
                return 1 + count_inner(curr.next)
            else:
                curr = curr.next
                result = count_inner(curr)
                return result

        if self.head is None:
            return 0

        curr = self.head
        result = count_inner(curr)
        return result

    def reverse(self, curr):
        """
        Reverses the order of the SLL
        param: Node pointing to the beginning of the SLL
        Time complexity: O(n)
        """
        if self.head is None:
            return
        if self.head.next is None:
            return
        if curr.next is None:
            self.head = curr
            return

        self.reverse(curr.next)
        ref_node = curr.next
        ref_node.next = curr
        curr.next = None

def crafting(recipe, pockets):
    """
    Checks if all items in recipe are in pockets
    param: recipe
    param: pockets
    return: True if values are in pockets and false otherwise
    Time complexity: O(rp)
    """

    curr_recipe = recipe.head
    curr_pockets = pockets.head

    def crafting_inner(curr_recipe: Node):
        """
        If all values in pockets return true
        """
        count_recipe = recipe.count(curr_recipe.val)
        count_pockets = pockets.count(curr_recipe.val)
        if count_pockets >= count_recipe:
            if curr_recipe.next is None:
                return True
            curr_recipe = curr_recipe.next
            result = crafting_inner(curr_recipe)
            return result
        else:
            return False

    def crafting_inner_2(curr_recipe: Node):
        """
        removes all values that are in recipe from pockets
        """
        pockets.remove(curr_recipe.val)
        if curr_recipe.next is None:
            return
        curr_recipe = curr_recipe.next
        crafting_inner_2(curr_recipe)

    if curr_recipe is None:
        return False
    if curr_pockets is None:
        return False

    result = crafting_inner(curr_recipe)

    if result:
        crafting_inner_2(curr_recipe)

    return result


