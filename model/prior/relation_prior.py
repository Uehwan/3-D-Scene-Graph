import math
import pickle


def load_obj(filename):
    with open(filename + '.pkl', 'rb') as f:
        return pickle.load(f)


def cal_normal_prob(mean, var, val):
    prob = math.exp(-0.5*((val - mean)/math.sqrt(var+0.0000000001))**2)
    return prob


def most_probable_relation_for_paired(pair_of_objs, relation_stat, dist):
    """
        returns
            1. most probable relation
            2. corresponding probability
    """
    if not pair_of_objs in relation_stat:
        return None, 0.0

    one_case = relation_stat[pair_of_objs]
    items = list(one_case)
    max_prob = -1
    max_pred = ""
    sum_count = 0
   
    for item in items:
        curr_count = one_case[item]['count']
        curr_mean = one_case[item]['mean']
        curr_var = one_case[item]['var']
        curr_dist_prob = cal_normal_prob(curr_mean, curr_var, dist)
        curr_total_prob = curr_count * curr_dist_prob
        sum_count += curr_count
        if curr_total_prob > max_prob:
            max_prob = curr_total_prob
            max_pred = item
    return max_pred, max_prob / float(sum_count)


def most_probable_relation_for_unpaired(list_of_two_objs, relation_stat, dist):
    """
        returns
            1. most probable relation
            2. corresponding probability
            3. direction (subject, object)
    """
    candidate1, prob1 = most_probable_relation_for_paired((list_of_two_objs[1], list_of_two_objs[0]), relation_stat, dist)
    candidate2, prob2 = most_probable_relation_for_paired((list_of_two_objs[0], list_of_two_objs[1]), relation_stat, dist)

    if prob1 > prob2:
        return candidate1, prob1, (list_of_two_objs[1], list_of_two_objs[0])
    else:
        return candidate2, prob2, (list_of_two_objs[0], list_of_two_objs[1])

def most_probable_relation_for_unpaired2(list_of_two_objs, relation_stat, dist):
    """
        returns
            1. most probable relation
            2. corresponding probability
            3. direction (subject, object)
    """
    candidate1, prob1 = most_probable_relation_for_paired((list_of_two_objs[1], list_of_two_objs[0]), relation_stat, dist)
    candidate2, prob2 = most_probable_relation_for_paired((list_of_two_objs[0], list_of_two_objs[1]), relation_stat, dist)

    if prob1 > prob2:
        return candidate1, prob1, prob1 > prob2
    else:
        return candidate2, prob2, prob1 > prob2

def triplet_prob_from_statistics(triplet, relation_stat, dist):
    """
        args
            1. triplet: (sbj,pred,obj).  ex ('desk','on','floor')
            2. relation_stat: statistics dataset
            3. dist: pixel_wise distance (scalar). ex 50.
        returns
            2. corresponding probability
    """
    pair_of_objs, predicate = (triplet[0], triplet[2]), triplet[1]
    max_prob = -1
    max_pred = ""
    sum_count = 0
    if pair_of_objs in relation_stat and predicate in relation_stat[pair_of_objs]:
        keys =relation_stat[pair_of_objs].keys()
        dist_probs = []
        total_probs = {}
        for key in keys:
            triplet_stat = relation_stat[pair_of_objs][key]
            curr_count = triplet_stat['count']
            curr_mean = triplet_stat['mean']
            curr_var = triplet_stat['var']
            curr_dist_prob = cal_normal_prob(curr_mean, curr_var, dist)
            dist_probs.append(curr_dist_prob)
            total_probs[key]=curr_count * curr_dist_prob
            sum_count += curr_count
        return total_probs[predicate]/float(sum_count)
    else:
        return 0.0


if __name__=="__main__":
    relation_statistics = load_obj("relation_prior_prob")
    print(most_probable_relation_for_unpaired(['floor', 'rock'], relation_statistics, 50))

