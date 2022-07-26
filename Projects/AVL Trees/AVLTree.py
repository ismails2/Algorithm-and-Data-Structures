"""
Project 5
CSE 331 S21 (Onsay)
Sara Ismail
AVLTree.py
"""

import queue
from typing import TypeVar, Generator, List, Tuple

# for more information on typehinting, check out https://docs.python.org/3/library/typing.html
T = TypeVar("T")  # represents generic type
Node = TypeVar("Node")  # represents a Node object (forward-declare to use in Node __init__)
AVLWrappedDictionary = TypeVar("AVLWrappedDictionary")  # represents a custom type used in application


####################################################################################################


class Node:
    """
    Implementation of an AVL tree node.
    Do not modify.
    """
    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["value", "parent", "left", "right", "height"]

    def __init__(self, value: T, parent: Node = None,
                 left: Node = None, right: Node = None) -> None:
        """
        Construct an AVL tree node.

        :param value: value held by the node object
        :param parent: ref to parent node of which this node is a child
        :param left: ref to left child node of this node
        :param right: ref to right child node of this node
        """
        self.value = value
        self.parent, self.left, self.right = parent, left, right
        self.height = 0

    def __repr__(self) -> str:
        """
        Represent the AVL tree node as a string.

        :return: string representation of the node.
        """
        return f"<{str(self.value)}>"

    def __str__(self) -> str:
        """
        Represent the AVL tree node as a string.

        :return: string representation of the node.
        """
        return f"<{str(self.value)}>"


####################################################################################################


class AVLTree:
    """
    Implementation of an AVL tree.
    Modify only below indicated line.
    """
    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["origin", "size"]

    def __init__(self) -> None:
        """
        Construct an empty AVL tree.
        """
        self.origin = None
        self.size = 0

    def __repr__(self) -> str:
        """
        Represent the AVL tree as a string. Inspired by Anna De Biasi (Fall'20 Lead TA).

        :return: string representation of the AVL tree
        """
        if self.origin is None:
            return "Empty AVL Tree"

        # initialize helpers for tree traversal
        root = self.origin
        result = ""
        q = queue.SimpleQueue()
        levels = {}
        q.put((root, 0, root.parent))
        for i in range(self.origin.height + 1):
            levels[i] = []

        # traverse tree to get node representations
        while not q.empty():
            node, level, parent = q.get()
            if level > self.origin.height:
                break
            levels[level].append((node, level, parent))

            if node is None:
                q.put((None, level + 1, None))
                q.put((None, level + 1, None))
                continue

            if node.left:
                q.put((node.left, level + 1, node))
            else:
                q.put((None, level + 1, None))

            if node.right:
                q.put((node.right, level + 1, node))
            else:
                q.put((None, level + 1, None))

        # construct tree using traversal
        spaces = pow(2, self.origin.height) * 12
        result += "\n"
        result += f"AVL Tree: size = {self.size}, height = {self.origin.height}".center(spaces)
        result += "\n\n"
        for i in range(self.origin.height + 1):
            result += f"Level {i}: "
            for node, level, parent in levels[i]:
                level = pow(2, i)
                space = int(round(spaces / level))
                if node is None:
                    result += " " * space
                    continue
                if not isinstance(self.origin.value, AVLWrappedDictionary):
                    result += f"{node} ({parent} {node.height})".center(space, " ")
                else:
                    result += f"{node}".center(space, " ")
            result += "\n"
        return result

    def __str__(self) -> str:
        """
        Represent the AVL tree as a string. Inspired by Anna De Biasi (Fall'20 Lead TA).

        :return: string representation of the AVL tree
        """
        return repr(self)

    def height(self, root: Node) -> int:
        """
        Return height of a subtree in the AVL tree, properly handling the case of root = None.
        Recall that the height of an empty subtree is -1.

        :param root: root node of subtree to be measured
        :return: height of subtree rooted at `root` parameter
        """
        return root.height if root is not None else -1

    def left_rotate(self, root: Node) -> Node:
        """
        Perform a left rotation on the subtree rooted at `root`. Return new subtree root.

        :param root: root node of unbalanced subtree to be rotated.
        :return: new root node of subtree following rotation.
        """
        if root is None:
            return None

        # pull right child up and shift right-left child across tree, update parent
        new_root, rl_child = root.right, root.right.left
        root.right = rl_child
        if rl_child is not None:
            rl_child.parent = root

        # right child has been pulled up to new root -> push old root down left, update parent
        new_root.left = root
        new_root.parent = root.parent
        if root.parent is not None:
            if root is root.parent.left:
                root.parent.left = new_root
            else:
                root.parent.right = new_root
        root.parent = new_root

        # handle tree origin case
        if root is self.origin:
            self.origin = new_root

        # update heights and return new root of subtree
        root.height = 1 + max(self.height(root.left), self.height(root.right))
        new_root.height = 1 + max(self.height(new_root.left), self.height(new_root.right))
        return new_root

    ########################################
    # Implement functions below this line. #
    ########################################

    def right_rotate(self, root: Node) -> Node:
        """
        Perform a right rotation on the subtree rooted at `root`. Return new subtree root.

        :param root: root node of unbalanced subtree to be rotated.
        :return: new root node of subtree following rotation.
        """
        if root is None:
            return None

        # pull left child up and shift left-right child across tree, update parent
        new_root, lr_child = root.left, root.left.right
        root.left = lr_child
        if lr_child is not None:
            lr_child.parent = root

        # left child has been pulled up to new root -> push old root down left, update parent
        new_root.right = root
        new_root.parent = root.parent
        if root.parent is not None:
            if root is root.parent.right:
                root.parent.right = new_root
            else:
                root.parent.left = new_root
        root.parent = new_root

        # handle tree origin case
        if root is self.origin:
            self.origin = new_root

        # update heights and return new root of subtree
        root.height = 1 + max(self.height(root.left), self.height(root.right))
        new_root.height = 1 + max(self.height(new_root.left), self.height(new_root.right))
        return new_root

    def balance_factor(self, root: Node) -> int:
        """
        Calculate the balance factor of a specific node
        Param: root, Node, node that the calculation is being performed on
        Return: int, the node's balance factor
        """
        if root is None:
            return 0
        height_l = self.height(root.left)
        height_r = self.height(root.right)
        return height_l - height_r

    def rebalance(self, root: Node) -> Node:
        """
        Rebalance the node that is being past into the function
        Param: root, Node, the node that the rebalance is being performed on
        Return: Node, the rebalanced node
        """
        balance = self.balance_factor(root)
        if balance == -2:
            if self.balance_factor(root.right) == 1:
                self.right_rotate(root.right)
            return self.left_rotate(root)
        if balance == 2:
            if self.balance_factor(root.left) == -1:
                self.left_rotate(root.left)
            return self.right_rotate(root)
        return root

    def insert(self, root: Node, val: T) -> Node:
        """
        Inserts a value in the correct location
        Param: root, Node, the starting node
        Param: val, type T, value that will be inserted into the tree
        Return: Node, inserted node
        """
        def insert_inner(self, root: Node, val: T) -> Node:
            """
            Where the recursion occurs to search for the location of the new node
            """
            if val < root.value and root.left is not None:
                # root.height += 1
                insert_inner(self, root.left, val)
            elif val > root.value and root.right is not None:
                # root.height += 1
                insert_inner(self, root.right, val)
            elif val < root.value:
                new_node = Node(val, root)
                root.left = new_node
            elif val > root.value:
                new_node = Node(val, root)
                root.right = new_node
            root.height = 1 + max(self.height(root.left), self.height(root.right))
            new_root = self.rebalance(root)
            return new_root
        if self.origin is None:
            root = Node(val)
            self.origin = root
            self.size += 1
            return root
        new_root = insert_inner(self, root, val)
        self.size += 1
        return new_root

    def min(self, root: Node) -> Node:
        """
        Finds the min value of the tree
        Param: root, Node, starting node
        Return: Node, the min value
        """
        if root is None:
            return root
        temp_node = root
        while temp_node.left:
            temp_node = temp_node.left
        return temp_node

    def max(self, root: Node) -> Node:
        """
        Finds the max value of the tree
        Param: root, Node, starting node
        Return: Node, the max value
        """
        if root is None:
            return root
        temp_node = root
        while temp_node.right:
            temp_node = temp_node.right
        return temp_node

    def search(self, root: Node, val: T) -> Node:
        """
        Searches for a value in a tree
        Param: root, Node, starting node
        Param: val, type T, val that is being searched for
        Return: Node, the node of the wanted value
        """
        if root is None:
            return root
        while root:
            if root.value == val:
                return root
            if val < root.value:
                if root.left is None:
                    return root
                root = root.left
            else:
                if root.right is None:
                    return root
                root = root.right

    def inorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Performs an traverses through the tree in order
        Param: root, Node, starting node
        Return: Generator[Node, None, None]
        """
        if root is None:
            return Generator[root, None, None]
        yield from self.inorder(root.left)
        yield root
        yield from self.inorder(root.right)

    def preorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Performs an preorder traversal through the tree
        Param: root, Node, starting node
        Return: Generator[Node, None, None]
        """
        if root is None:
            return Generator[root, None, None]
        yield root
        yield from self.preorder(root.left)
        yield from self.preorder(root.right)

    def postorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Performs an postorder traversal through the tree
        Param: root, Node, starting node
        Return: Generator[Node, None, None]
        """
        if root is None:
            return Generator[root, None, None]
        yield from self.postorder(root.left)
        yield from self.postorder(root.right)
        yield root

    def levelorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Performs an level traversal through the tree
        Param: root, Node, starting node
        Return: Generator[Node, None, None]
        """
        fringe = queue.SimpleQueue()
        fringe.put(root)
        if root is None:
            return Generator[root, None, None]
        while not fringe.empty():
            pos = fringe.get()
            yield pos
            if pos.left:
                fringe.put(pos.left)
            if pos.right:
                fringe.put(pos.right)

    def removal(self, root: Node, node: Node, new, val):
        """
        Helper function for removal
        Param: root, Node
        Param: node, Node
        Param: new, val node will be replaced by
        Param: val, the val that is being removed from the tree
        """
        if root is self.origin:
            self.origin = new
        elif node.right and node.right.value == val:
            node.right = new
        elif node.left and node.left.value == val:
            node.left = new

    def remove(self, root: Node, val: T) -> Node:
        """
        Removes a value from a tree if present
        Param: root, Node, starting node
        Param: val, type T, value being removed
        Return: the removed node
        """
        if root is None:
            return root
        elif val < root.value:
            self.remove(root.left, val)
        elif val > root.value:
            self.remove(root.right, val)
        else:
            if root.left and root.right:
                max_left = self.max(root.left)
                root.value = max_left.value
                self.remove(root.left, root.value)
            elif root.left:
                successor = root.parent
                next_node = root.left
                self.removal(root, successor, next_node, val)
                next_node.parent = successor
                self.size -= 1
            elif root.right:
                successor = root.parent
                next_node = root.right
                self.removal(root, successor, next_node, val)
                next_node.parent = successor
                self.size -= 1
            else:
                successor = root.parent
                self.removal(root, successor, None, val)
                self.size -= 1
        root.height = 1 + max(self.height(root.left), self.height(root.right))
        result = self.rebalance(root)
        return result

####################################################################################################


class AVLWrappedDictionary:
    """
    Implementation of a helper class which will be used as tree node values in the
    NearestNeighborClassifier implementation. Compares objects with keys less than
    1e-6 apart as equal.
    """
    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["key", "dictionary"]

    def __init__(self, key: float) -> None:
        """
        Construct a AVLWrappedDictionary with a key to search/sort on and a dictionary to hold data.

        :param key: floating point key to be looked up by.
        """
        self.key = key
        self.dictionary = {}

    def __repr__(self) -> str:
        """
        Represent the AVLWrappedDictionary as a string.

        :return: string representation of the AVLWrappedDictionary.
        """
        return f"key: {self.key}, dict: {self.dictionary}"

    def __str__(self) -> str:
        """
        Represent the AVLWrappedDictionary as a string.

        :return: string representation of the AVLWrappedDictionary.
        """
        return f"key: {self.key}, dict: {self.dictionary}"

    def __eq__(self, other: AVLWrappedDictionary) -> bool:
        """
        Implement == operator to compare 2 AVLWrappedDictionaries by key only.
        Compares objects with keys less than 1e-6 apart as equal.

        :param other: other AVLWrappedDictionary to compare with
        :return: boolean indicating whether keys of AVLWrappedDictionaries are equal
        """
        return abs(self.key - other.key) < 1e-6

    def __lt__(self, other: AVLWrappedDictionary) -> bool:
        """
        Implement < operator to compare 2 AVLWrappedDictionarys by key only.
        Compares objects with keys less than 1e-6 apart as equal.

        :param other: other AVLWrappedDictionary to compare with
        :return: boolean indicating ordering of AVLWrappedDictionaries
        """
        return self.key < other.key and not abs(self.key - other.key) < 1e-6

    def __gt__(self, other: AVLWrappedDictionary) -> bool:
        """
        Implement > operator to compare 2 AVLWrappedDictionaries by key only.
        Compares objects with keys less than 1e-6 apart as equal.

        :param other: other AVLWrappedDictionary to compare with
        :return: boolean indicating ordering of AVLWrappedDictionaries
        """
        return self.key > other.key and not abs(self.key - other.key) < 1e-6


class NearestNeighborClassifier:
    """
    Implementation of a one-dimensional nearest-neighbor classifier with AVL tree lookups.
    Modify only below indicated line.
    """
    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["resolution", "tree"]

    def __init__(self, resolution: int) -> None:
        """
        Construct a one-dimensional nearest neighbor classifier with AVL tree lookups.
        Data are assumed to be floating point values in the closed interval [0, 1].

        :param resolution: number of decimal places the data will be rounded to, effectively
                           governing the capacity of the model - for example, with a resolution of
                           1, the classifier could maintain up to 11 nodes, spaced 0.1 apart - with
                           a resolution of 2, the classifier could maintain 101 nodes, spaced 0.01
                           apart, and so on - the maximum number of nodes is bounded by
                           10^(resolution) + 1.
        """
        self.tree = AVLTree()
        self.resolution = resolution

        # pre-construct lookup tree with AVLWrappedDictionary objects storing (key, dictionary)
        # pairs, but which compare with <, >, == on key only
        for i in range(10 ** resolution + 1):
            w_dict = AVLWrappedDictionary(key=(i / 10 ** resolution))
            self.tree.insert(self.tree.origin, w_dict)

    def __repr__(self) -> str:
        """
        Represent the NearestNeighborClassifier as a string.

        :return: string representation of the NearestNeighborClassifier.
        """
        return f"NNC(resolution={self.resolution}):\n{self.tree}"

    def __str__(self) -> str:
        """
        Represent the NearestNeighborClassifier as a string.

        :return: string representation of the NearestNeighborClassifier.
        """
        return f"NNC(resolution={self.resolution}):\n{self.tree}"

    def fit(self, data: List[Tuple[float, str]]) -> None:
        """
        Creates a tree of type AVLWrappedDictionary
        Param: data, list[tuple(float, str)], values that are inserted into
        the tree
        """
        for i in data:
            rounded = round(i[0], self.resolution)
            typ = AVLWrappedDictionary(rounded)
            val = self.tree.search(self.tree.origin, typ)
            if val is None:
                typ.dictionary = {i[1]: 1}
                self.tree.insert(typ)
            else:
                if i[1] in val.value.dictionary:
                    val.value.dictionary[i[1]] += 1
                else:
                    val.value.dictionary[i[1]] = 1

    def predict(self, x: float, delta: float) -> str:
        """
        Predicts what a value might be based on surrounding data
        Param: x, float, the value that is being predicted
        Param: delta, float, the range of values to use when making a prediction
        Return: str, prediction
        """
        def predict_inner(val: int):
            """
            Helper function that updates the dictionary
            """
            node = self.tree.search(self.tree.origin, AVLWrappedDictionary(val))
            for key, item in node.value.dictionary.items():
                if key in overall_dict:
                    overall_dict[key] += item
                else:
                    overall_dict[key] = item

        rounded = round(x, self.resolution)
        max_val = round(rounded + delta, self.resolution)
        min_val = round(rounded - delta, self.resolution)
        overall_dict = {}
        val = round(min_val, self.resolution)
        lst = []
        num = 1/(10**self.resolution)
        while val <= max_val:
            lst.append(val)
            val = round(num + val, self.resolution)
        for i in lst:
            predict_inner(i)
        max_num = -1
        max_val = None
        for keys, items in overall_dict.items():
            if items > max_num:
                max_num = items
                max_val = keys
        return max_val
