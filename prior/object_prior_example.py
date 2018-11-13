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
def cal_p_x_given_yz(x_class, y_class, z_class, key2ind_pair, joint_prob):
    p_x_given_y = cal_p_x_given_y(x_class, y_class, key2ind_pair, joint_prob)
    p_x_given_z = cal_p_x_given_y(x_class, z_class, key2ind_pair, joint_prob)
    return min(p_x_given_y, p_x_given_z)


key2ind = load_obj("object_prior_key2ind")
joint_probability = load_obj("object_prior_prob")

candidates = ['bed','apple','ball','keyboard','table','desk','rug','pillow']
filtered_candidates = []
for i,candidate in enumerate(candidates):
    ''' condition check'''
    pillow_and_floor = cal_p_xy_joint('pillow','floor',key2ind,joint_probability)
    print('p(x=pillow,z=floor) = ',pillow_and_floor)
    pillow_and_bed = cal_p_xy_joint('pillow',candidate,key2ind,joint_probability)
    print('p(x=pillow,y={c}) = {p}'.format(c=candidate,p=pillow_and_bed))
    bed_and_floor = cal_p_xy_joint(candidate,'floor',key2ind,joint_probability)
    print('p(y={c},z=floor) = {p}'.format(c=candidate,p=bed_and_floor))
    print('p(x=pillow,z=floor)^2 = {p}'.format(p=pillow_and_floor**2))
    print('p(x=pillow,y={c})*p(y={c},z=floor) = {p}'.format(c=candidate,p=pillow_and_bed*pillow_and_floor))
    condition_check = pillow_and_floor**2<pillow_and_bed*pillow_and_floor
    print('p(x,z)^2 < p(x,y)*p(y,z): '+str(condition_check)+'\n')
    if condition_check: filtered_candidates.append(candidate)

p_y_given_xz =[cal_p_x_given_yz(candidate,'pillow','floor',key2ind,joint_probability) for candidate in filtered_candidates]

print('finally chosen object!!!: '+filtered_candidates[ np.array(p_y_given_xz).argmax()])

