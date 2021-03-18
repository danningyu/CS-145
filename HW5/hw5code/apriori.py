from itertools import chain, combinations,islice
from collections import defaultdict
from time import time
import pandas as pd
import operator


def run_apriori(infile, min_support, min_conf):
    """
    Run the Apriori algorithm. infile is a record iterator.
    Return:
        rtn_items: list of (set, support)
        rtn_rules: list of ((preset, postset), confidence)
    """
    one_cand_set, all_transactions = gen_one_item_cand_set(infile)

    set_count_map = defaultdict(int)  # maintains the count for each set

    one_freq_set, set_count_map = get_items_with_min_support(
        one_cand_set, all_transactions, min_support, set_count_map)

    freq_map, set_count_map = run_apriori_loops(
        one_freq_set, set_count_map, all_transactions, min_support)

    rtn_items = get_frequent_items(set_count_map, freq_map)
    rtn_rules = get_frequent_rules(set_count_map, freq_map, min_conf)

    return rtn_items, rtn_rules


def gen_one_item_cand_set(input_fileator):
    """
    Generate the 1-item candidate sets and a list of all the transactions.
    """
    all_transactions = list()
    one_cand_set = set()
    for record in input_fileator:
        transaction = frozenset(record)
        all_transactions.append(transaction)
        #========================#
        # STRART YOUR CODE HERE  #
        #========================#
        for item in transaction:
            temp_set = frozenset([item])
            if temp_set not in one_cand_set:
                one_cand_set.add(temp_set)
        #========================#
        #   END YOUR CODE HERE   #
        #========================# 
    return one_cand_set, all_transactions


def get_items_with_min_support(item_set, all_transactions, min_support,
                               set_count_map):
    """
    item_set is a set of candidate sets.
    Return a subset of the item_set
    whose elements satisfy the minimum support.
    Update set_count_map.
    """
    rtn = set()
    local_set = defaultdict(int)

    for item in item_set:
        for transaction in all_transactions:
            if item.issubset(transaction):
                set_count_map[item] += 1
                local_set[item] += 1

    #========================#
    # STRART YOUR CODE HERE  #
    #========================#
    for item, count in local_set.items():
        if count >= min_support:
            rtn.add(item)
    #========================#
    #   END YOUR CODE HERE   #
    #========================# 

    return rtn, set_count_map


def run_apriori_loops(one_cand_set, set_count_map, all_transactions,
                      min_support):
    """
    Return:
        freq_map: a dict
            {<length_of_set_l>: <set_of_frequent_itemsets_of_length_l>}
        set_count_map: updated set_count_map
    """
    freq_map = dict()
    current_l_set = one_cand_set
    i = 1
    #========================#
    # STRART YOUR CODE HERE  #
    #========================#
    while (current_l_set != set([])):
        freq_map[i] = current_l_set
        current_l_set = join_set(current_l_set, i)
        current_c_set, set_count_map = get_items_with_min_support(current_l_set, all_transactions, min_support, set_count_map)
        current_l_set = current_c_set
        i += 1
    #========================#
    #   END YOUR CODE HERE   #
    #========================# 

    return freq_map, set_count_map


def get_frequent_items(set_count_map, freq_map):
    """ Return frequent items as a list. """
    rtn_items = []
    for key, value in freq_map.items():
        rtn_items.extend(
            [(tuple(item), get_support(set_count_map, item))
             for item in value])
    return rtn_items


def get_frequent_rules(set_count_map, freq_map, min_conf):
    """ Return frequent rules as a list. """
    rtn_rules = []
    for key, value in islice(freq_map.items(),1,None):
        for item in value:
            _subsets = map(frozenset, [x for x in subsets(item)])
            for element in _subsets:
                remain = item.difference(element)
                if len(remain) > 0:
               #========================#
               # STRART YOUR CODE HERE  #
               #========================#
                    both = element.union(remain)
                    confidence = set_count_map[both] / set_count_map[element]
               #========================#
               #   END YOUR CODE HERE   #
               #========================# 
                    if confidence >= min_conf:
                        rtn_rules.append(
                            ((tuple(element), tuple(remain)), confidence))
    return rtn_rules


def get_support(set_count_map, item):
    """ Return the support of an item. """
    #========================#
    # STRART YOUR CODE HERE  #
    #========================#
    sup_item = 0         
    sup_item = set_count_map[item]
    #========================#
    #   END YOUR CODE HERE   #
    #========================# 
    return sup_item


def join_set(s, l):
    """
    Join a set with itself .
    Return a set whose elements are unions of sets in s with length==l.
    """
    #========================#
    # STRART YOUR CODE HERE  #
    #========================#
    join_set = set()
    for item1 in s:
        for item2 in s:
            if item1 is not item2:
                # union must be of length l+1 for this to work
                # otherwise, we're generating a set that's too long
                combined = item1.union(item2)
                if len(combined) == l+1:
                    join_set.add(frozenset(combined))
        
    #========================#
    #   END YOUR CODE HERE   #
    #========================# 
    return join_set


def subsets(x):
    """ Return non =-empty subsets of x. """
    return chain(*[combinations(x, i + 1) for i, a in enumerate(x)])


def print_items_rules(items, rules, ignore_one_item_set=False, name_map=None):
    for item, support in sorted(items, key=operator.itemgetter(1)):
        if len(item) == 1 and ignore_one_item_set:
            continue
        print ('item: ')
        print (convert_item_to_name(item, name_map), support)
    print ('\n------------------------ RULES:')
    for rule, confidence in sorted(
            rules, key=operator.itemgetter(1)):
        pre, post = rule
        print ('Rule: ')
        print( convert_item_to_name(pre, name_map), convert_item_to_name(post, name_map),confidence)


def convert_item_to_name(item, name_map):
    """ Return the string representation of the item. """
    if name_map:
        return ','.join([name_map[x] for x in item])
    else:
        return str(item)


def read_data(fname):
    """ Read from the file and yield a generator. """
    file_iter = open(fname, 'rU')
    for line in file_iter:
        line = line.strip().rstrip(',')
        record = frozenset(line.split(','))
        yield record


def read_name_map(fname):
    """ Read from the file and return a dict mapping ids to names. """
    df = pd.read_csv(fname, sep=',\t ', header=None, names=['id', 'name'],
                     engine='python')
    return df.set_index('id')['name'].to_dict()