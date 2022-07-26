"""
Sara Ismail
Coding Challenge 8
CSE 331 Spring 2021
Professor Sebnem Onsay
"""

from typing import Set, Tuple, Dict
from CC8.InventoryItems import ItemInfo, ALL_ITEMS


class Bundle:
    """ Bundle Class """

    def __init__(self) -> None:
        """
        Initializes the Bundle class with an empty dictionary and the fullness of the stack
        """
        self.items_in_bundle = {}
        self.fullness = 0
        pass

    def to_set(self) -> Set[Tuple[str, int]]:
        """
        Converts the bundle of items to a set of tuples
        Return: Set[Tuple[str, int]], with each item in the set being the item and
        the number of each item
        """
        result = set()
        for key, item in self.items_in_bundle.items():
            if item > 0:
                result.add((key, item))
        return result

    def add_to_bundle(self, item_name: str, amount: int) -> bool:
        """
        Checks if an item can be added to a bundle and returns if the items have been
        added to the stack
        Param: item_name, str, the item that is trying to be added to the stack
        Param: amount, int, number of a specific item trying to be added
        Return: bool, returns true if an item is added and false if not
        """
        if ALL_ITEMS[item_name] is None:
            return False
        new_fullness = self.fullness + amount/ALL_ITEMS[item_name].amount_in_stack
        if new_fullness > 1:
            return False
        if self.items_in_bundle.get(item_name):
            self.items_in_bundle[item_name] += amount
        else:
            self.items_in_bundle[item_name] = amount
        self.fullness = new_fullness
        return True

    def remove_from_bundle(self, item_name: str, amount: int) -> bool:
        """
        Checks if an item can be removed from a bundle and returns if the items have been
        removed from the stack
        Param: item_name, str, the item that is trying to be removed from the stack
        Param: amount, int, number of a specific item trying to be removed
        Return: bool, returns true if an item is removed and false if not
        """
        try:
            self.items_in_bundle[item_name]
        except KeyError:
            return False
        number_in_bundle = self.items_in_bundle[item_name]
        if number_in_bundle < amount:
            return False
        self.items_in_bundle[item_name] = number_in_bundle - amount
        self.fullness -= amount/ALL_ITEMS[item_name].amount_in_stack
        return True
