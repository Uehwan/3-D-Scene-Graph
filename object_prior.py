import numpy as np
import pickle


def load_obj(filename):
    with open(filename + '.pkl', 'rb') as f:
        return pickle.load(f)


# p(x, y)
def cal_p_xy_joint(x_class, y_class, key2ind_pair, joint_prob):
    p_xy = joint_probability[key2ind_pair[x_class], key2ind_pair[y_class]] / np.sum(joint_probability)
    return p_xy


# p(x|y)
def cal_p_x_given_y(x_class, y_class, key2ind_pair, joint_prob):
    single_prob = np.sum(joint_probability, axis=1)
    p_y = single_prob[key2ind_pair[y_class]]
    p_xy = joint_probability[key2ind_pair[x_class], key2ind_pair[y_class]]
    return p_xy / p_y


# p(x|y,z) approximated
def cal_p_x_given_xy(x_class, y_class, z_class, key2ind_pair, joint_prob):
    p_x_given_y = cal_p_x_given_y(x_class, y_class, key2ind_pair, joint_prob)
    p_x_given_z = cal_p_x_given_y(x_class, z_class, key2ind_pair, joint_prob)
    return min(p_x_given_y, p_x_given_z)

if __name__=="__main__":

    key2ind = load_obj("object_prior_key2ind")
    joint_probability = load_obj("object_prior_prob")
    print(cal_p_x_given_y('floor', 'rock', key2ind, joint_probability))

