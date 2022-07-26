"""
Project 6
CSE 331 S21 (Onsay)
Sara Ismail
hashtable.py
"""

from typing import TypeVar, List, Tuple

T = TypeVar("T")
HashNode = TypeVar("HashNode")
HashTable = TypeVar("HashTable")


class HashNode:
    """
    DO NOT EDIT
    """
    __slots__ = ["key", "value", "deleted"]

    def __init__(self, key: str, value: T, deleted: bool = False) -> None:
        self.key = key
        self.value = value
        self.deleted = deleted

    def __str__(self) -> str:
        return f"HashNode({self.key}, {self.value})"

    __repr__ = __str__

    def __eq__(self, other: HashNode) -> bool:
        return self.key == other.key and self.value == other.value

    def __iadd__(self, other: T) -> None:
        self.value += other


class HashTable:
    """
    Hash Table Class
    """
    __slots__ = ['capacity', 'size', 'table', 'prime_index']

    primes = (
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
        89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179,
        181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277,
        281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389,
        397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499,
        503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617,
        619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739,
        743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859,
        863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991,
        997)

    def __init__(self, capacity: int = 8) -> None:
        """
        DO NOT EDIT
        Initializes hash table
        :param capacity: capacity of the hash table
        """
        self.capacity = capacity
        self.size = 0
        self.table = [None] * capacity

        i = 0
        while HashTable.primes[i] <= self.capacity:
            i += 1
        self.prime_index = i - 1

    def __eq__(self, other: HashTable) -> bool:
        """
        DO NOT EDIT
        Equality operator
        :param other: other hash table we are comparing with this one
        :return: bool if equal or not
        """
        if self.capacity != other.capacity or self.size != other.size:
            return False
        for i in range(self.capacity):
            if self.table[i] != other.table[i]:
                return False
        return True

    def __str__(self) -> str:
        """
        DO NOT EDIT
        Represents the table as a string
        :return: string representation of the hash table
        """
        represent = ""
        bin_no = 0
        for item in self.table:
            represent += "[" + str(bin_no) + "]: " + str(item) + '\n'
            bin_no += 1
        return represent

    __repr__ = __str__

    def _hash_1(self, key: str) -> int:
        """
        ---DO NOT EDIT---
        Converts a string x into a bin number for our hash table
        :param key: key to be hashed
        :return: bin number to insert hash item at in our table, None if key is an empty string
        """
        if not key:
            return None
        hashed_value = 0

        for char in key:
            hashed_value = 181 * hashed_value + ord(char)
        return hashed_value % self.capacity

    def _hash_2(self, key: str) -> int:
        """
        ---DO NOT EDIT---
        Converts a string x into a hash
        :param key: key to be hashed
        :return: a hashed value
        """
        if not key:
            return None
        hashed_value = 0

        for char in key:
            hashed_value = 181 * hashed_value + ord(char)

        prime = HashTable.primes[self.prime_index]

        hashed_value = prime - (hashed_value % prime)
        if hashed_value % 2 == 0:
            hashed_value += 1
        return hashed_value

    def __len__(self) -> int:
        """
        returns the number of items in the hash table
        """
        return self.size

    def __setitem__(self, key: str, value: T) -> None:
        """
        calls _insert to set a key with a specific value
        Param: key, str
        Param: value, int
        """
        self._insert(key, value)

    def __getitem__(self, key: str) -> T:
        """
        calls _get to jey the specific value at a key and raises a key error
        if the key is not there
        Param: key, str
        Return: type T
        """
        result = self._get(key)
        if result is None:
            raise KeyError
        return result.value

    def __delitem__(self, key: str) -> None:
        """
        calls _delete to remove an item and raises a key error if not present
        Param: key, str
        """
        x = self[key]
        self._delete(key)

    def __contains__(self, key: str) -> bool:
        """
        Checks if a value is present and returns true if there and false if not
        Param: Key, str
        Return: bool
        """
        node = self._get(key)
        if node and node.key == key:
            return True
        return False

    def hash(self, key: str, inserting: bool = False) -> int:
        """
        Finds the correct index for a certain key
        Param: key, str
        Param: inserting, bool, checks to see if item is deleted
        return: int, index of the hashnode
        """
        hash_1 = self._hash_1(key)
        hash_2 = self._hash_2(key)
        index = hash_1
        j = 1
        while True:
            if self.table[index] is None:
                break
            if inserting and self.table[index].deleted:
                return index
            if self.table[index].deleted is False and self.table[index].key == key:
                return index
            index = (hash_1 + j * hash_2) % self.capacity
            j += 1
        return index

    def _insert(self, key: str, value: T) -> None:
        """
        Inserts a key into a specific index in the hashtable
        Param: key, str
        Param: value, type T
        """
        index = self.hash(key, True)
        node = HashNode(key, value)
        if self.table[index] is None or self.table[index].deleted:
            self.table[index] = node
            self.size += 1
        else:
            self.table[index] = node

        if self.size >= self.capacity//2:
            self._grow()

    def _get(self, key: str) -> HashNode:
        """
        Finds a key and returns its value
        Param: key, str
        Return: hashnode
        """
        j = self.hash(key)
        return self.table[j]

    def _delete(self, key: str) -> None:
        """
        Deletes the value of a specific key
        Param: key, str
        """
        j = self.hash(key)
        if self.table[j] is None:
            return
        if self.table[j].key == key:
            self.table[j].deleted = True
            self.table[j].key = None
            self.table[j].value = None
            self.size -= 1

    def _grow(self) -> None:
        """
        Doubles the capacity of the hashtable and rehashes the indexs
        """

        old = self.table
        self.capacity *= 2
        self.size = 0
        self.table = [None] * self.capacity

        i = 0
        while HashTable.primes[i] <= self.capacity:
            i += 1
        self.prime_index = i - 1

        for i in range(len(old)):
            if old[i] is None:
                continue
            if old[i].deleted:
                continue
            self._insert(old[i].key, old[i].value)

    def update(self, pairs: List[Tuple[str, T]] = []) -> None:
        """
        Takes a list of items and adds them to the hashtable
        Param: pairs, List[Tuple[str, T]]
        """
        for item in pairs:
            self._insert(item[0], item[1])

    def keys(self) -> List[str]:
        """
        Creates a list of keys in hashtable
        Return: List[str]
        """
        result = []
        for item in self.table:
            if item is None or item.deleted:
                continue
            result.append(item.key)
        return result

    def values(self) -> List[T]:
        """
        Creates a list of values in hashtable
        Return: List[type T]
        """
        result = []
        for item in self.table:
            if item is None or item.deleted:
                continue
            result.append(item.value)
        return result

    def items(self) -> List[Tuple[str, T]]:
        """
        Creates a list of keys and values in hashtable
        Return: List[(str, type T)]
        """
        result = []
        for item in self.table:
            if item is None or item.deleted:
                continue
            result.append((item.key, item.value))
        return result

    def clear(self) -> None:
        """
        Clears the hashtable
        """
        self.table = [None] * self.capacity
        self.size = 0

class CataData:
    def __init__(self) -> None:
        """
        Initializes CataData
        """
        self.enter_cata = HashTable()
        self.system = HashTable()

    def enter(self, idx: str, origin: str, time: int) -> None:
        """
        Inserts information into the enter cata hash table
        Param: idx, str
        Param: origin, str
        Param: time, int
        """
        self.enter_cata[idx] = (origin, time)

    def exit(self, idx: str, dest: str, time: int) -> None:
        """
        Removes information out of the enter cata hash table and enters information
        on total time into the system hashtable
        Param: idx, str
        Param: dest, str
        Param: time, int
        """
        hash_node = self.enter_cata[idx]
        del self.enter_cata[idx]
        origin = hash_node[0]
        start_time = hash_node[1]
        key = origin + ',' + dest
        try:
            start_stop = self.system[key]
            overall_time = start_stop[0] + (time - start_time)
            num_people = start_stop[1] + 1
            self.system[key] = (overall_time, num_people)
        except KeyError:
            self.system[key] = (time - start_time, 1)
        pass

    def get_average(self, origin: str, dest: str) -> float:
        """
        Pulls information from system and finds the average runtime
        Param: origin, str
        Param: dest, str
        Result: float (average time)
        """
        key = origin + ',' + dest
        try:
            val = self.system[key]
        except KeyError:
            return 0.0
        top = val[0]
        bottom = val[1]
        return top/bottom
