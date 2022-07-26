"""
Sara Ismail
Coding Challenge 2
CSE 331 Spring 2021
Professor Sebnem Onsay
"""
from typing import List, Tuple
from linked_list import DLLNode, LinkedList


def pokemon_machine(pokemon: LinkedList, orders: List[Tuple]) -> LinkedList:
    """
    """'''
    #__slots__ = [pokemon, orders]
    index_list = []
    order_list = []
    string_list = []
    for order in orders:
        order_list.append(order[0])
        if order[0] == 'swap':
            index_list.append(order[1])
            index_list.append(order[2])
        elif order[0] == 'add':
            index_list.append(order[1])
            string_list
    
    for order in orders:
        #cur_node = pokemon.head
        type = order[0]
        if type == 'add':
            #add_pokemon(cur_node, )
        elif type == 'remove':
            #remove_pokemon(cur_node)
        elif type == 'swap':
            #swap_pokemon(cur_node)'''


    def add_pokemon(cur_node: DLLNode, added_pokemon: str) -> None:
        """
        """
        new_node = DLLNode(added_pokemon)
        #if cur_node.prev is None:
        #    next_node = cur_node.next
        #    next_node.prev = cur_node
        #    return
        #if cur_node.next is None:
        cur_next = cur_node.nxt
        cur_node.nxt = new_node
        new_node.prev = cur_node
        new_node.nxt = cur_next

    def remove_pokemon(cur_node: DLLNode) -> None:
        """
        """

        cur_node = cur_node.nxt
        if cur_node.nxt is None:
            prev_node = cur_node.prev
            cur_node.prev = None
            prev_node.nxt = None
            return

        next_node = cur_node.nxt
        prev_node = cur_node.prev
        prev_node.nxt = next_node
        next_node.prev = prev_node

    def swap_pokemon(first_node: DLLNode, second_node: DLLNode) -> None:
        """
        """
        first_node = first_node.nxt
        second_node = second_node.nxt

        first_next = first_node.nxt
        first_node.nxt = second_node.nxt
        second_node.nxt = first_next
        if first_node.nxt:
            first_node.nxt.prev = first_node
        if second_node.nxt:
            second_node.nxt.prev = second_node
        first_prev = first_node.prev
        first_node.prev = second_node.prev
        second_node.prev = first_prev
        if first_node.prev:
            first_node.prev.nxt = first_node
        if second_node.prev is None:
            return
        second_node.prev.nxt = second_node


        '''
        first_node = first_node.nxt
        second_node = second_node.nxt

        if first_node.nxt is None:
            first_prev = first_node.prev
            first_next = first_node.nxt
            second_prev = second_node.prev
            second_next = second_node.nxt

            first_node.prev = second_prev
            second_prev.nxt = first_prev
            first_node.nxt = second_next
            second_next.prev = first_next

            second_node.prev = first_prev
            first_prev.nxt = second_prev
            second_node.nxt = first_next
            #first_next.prev = second_next
            return
        if second_node.nxt is None:
            first_prev = first_node.prev
            first_next = first_node.nxt
            second_prev = second_node.prev
            second_next = second_node.nxt

            first_node.prev = second_prev
            second_prev.nxt = first_prev
            first_node.nxt = second_next
            #second_next.prev = first_next

            second_node.prev = first_prev
            first_prev.nxt = second_prev
            second_node.nxt = first_next
            first_node.prev = second_next
            return
        #if first_node.p
        if second_node.nxt and first_node.nxt:
            first_prev = first_node.prev
            first_next = first_node.nxt
            second_prev = second_node.prev
            second_next = second_node.nxt

            first_node.prev = second_prev
            #second_prev.nxt = first_prev
            first_node.nxt = second_next
            second_next.prev = first_next

            second_node.prev = first_prev
            first_prev.nxt = second_prev
            second_node.nxt = first_next
            first_node.prev = second_next
        '''
        '''
        first_node = first_node.nxt
        second_node = second_node.nxt
        f_prev = first_node.prev
        f_next = first_node.nxt
        s_prev = second_node.prev
        s_next = second_node.nxt
        first_node.prev = s_prev
        first_node.nxt = s_next
        second_node.prev = f_prev
        second_node.nxt = f_next
        return
        '''
    for order in orders:
        cur_node = pokemon.head
        if len(order) == 2:
            index = order[1]
            for i in range(index):
                cur_node = cur_node.nxt
            remove_pokemon(cur_node)
        elif len(order) == 3:
            type = order[0]
            index1 = order[1]
            index2 = order[2]
            if type == 'add':
                for i in range(index1):
                    cur_node = cur_node.nxt
                add_pokemon(cur_node, index2)
            elif type == 'swap':
                cur_node1 = cur_node
                cur_node2 = cur_node
                for i in range(index1):
                    cur_node1 = cur_node1.nxt
                for i in range(index2):
                    cur_node2 = cur_node2.nxt
                swap_pokemon(cur_node1, cur_node2)
    return pokemon
