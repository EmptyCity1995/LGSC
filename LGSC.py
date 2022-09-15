import cv2
import math
import numpy as np

# calculate the distance between any two nodes in the key point set
def distance_cal(cord_list):
    num = len(cord_list)  # the number of node
    distance_matrix = np.zeros([num, num])  # the distance matrix
    for i, cord_i in enumerate(cord_list):
        for j, cord_j in enumerate(cord_list):
            # calculate the Euclidean distance
            distance = math.sqrt((cord_i[0][0] - cord_j[0][0]) ** 2 + (cord_i[0][1] - cord_j[0][1]) ** 2)
            if i != j:
                # avoid zero distance except for diagonal elements
                distance_matrix[i][j] = distance + 0.0000001
        # print(distance_matrix[i])
    return distance_matrix

# search K nearest neighbors of the i-th node
def find_KNN(K, distance_matrix, index):
    distance_vector = np.array(distance_matrix[index])  # extract the distance vector of the i-th node
    # get an enumeration index vector ranging from 0 to the number of nodes
    index_vector = np.arange(0, len(distance_vector))
    sorted_dis_vec = np.lexsort((index_vector.T, distance_vector.T))  # sort by distance in ascending order
    # print(sorted_dis_vec)
    # get the indexes of K nearest neighbors of the i-th node and the first one is itself
    K_neighbor_index = sorted_dis_vec[:K+1]
    # get the distances of K nearest neighbors of the i-th node and the first one is 0
    K_neighbor_distance = distance_vector[K_neighbor_index]
    return K_neighbor_index, K_neighbor_distance, sorted_dis_vec

# calculate the ranking shift of the i-th node
def cal_ranking_shift(K_idx_list, sorted_idx_list):
    phi_idx = []  # a binary sequence phi_vi of the i-th node vi
    # Iterate over the K nearest neighbors of the i-th node
    for k, n in enumerate(K_idx_list):
        # n is the k-th nearest neighbor vj of the node vi
        # r is the ranking of n's corresponding node v'j centered on v'i
        r = list(sorted_idx_list).index(n)
        phi = r > k  # if i > k, phi = 1 else phi = 0
        phi_idx.append(phi)
    return phi_idx

# calculate the node affinity score s(vi, v'i)
def cal_node_affinity_score(ori_phi_idx, dst_phi_idx, K):
    # convert the binary sequences phi_vi and phi_v'i to numpy arrays
    ori_phi_idx = np.array(ori_phi_idx)
    dst_phi_idx = np.array(dst_phi_idx)
    # s(vi, v'i) = 1 - 1/2K * |phi_vi + phi_v'i|
    node_s_idx = 1 - np.sum(abs(np.add(ori_phi_idx, dst_phi_idx))) / (2 * K)
    return node_s_idx

# calculate the edge affinity score s(e_iij, e'_iij)
def cal_edge_affinity_score(idx, x_idx, ori_K_idx, dst_K_idx, ori_dis_mat, dst_dis_mat, K):
    edge_s_idx_list = []
    # Iterate over the K nearest neighbors of the i-th node expect for itself
    for j in range(len(ori_K_idx[1:])):
        # if x_idx[n] == 1:
        # i_j is the j-th neighbor of node vi
        i_j = ori_K_idx[j+1]
        # get the edge distance d(e_iij) between node vi and its j-th neighbor node vij
        ori_edge_dis = ori_dis_mat[idx][i_j]
        # i_j_ is the j-th neighbor of corresponding node v'i
        i_j_ = dst_K_idx[j+1]
        # get the edge distance d(e'_iij) between corresponding node v'i and its j-th neighbor node v'ij
        dst_edge_dis = dst_dis_mat[idx][i_j_]
        # print(idx, i_j, ori_edge_dis, i_j_,des_edge_dis)
        # s(e_iij, e'_iij) = 1/k * exp(-|d(e_iij) - d(e'_iij)| / max(d(e_iij), d(e'_iij)))
        edge_s_idx = np.exp(- abs(ori_edge_dis - dst_edge_dis) / max(ori_edge_dis, dst_edge_dis)) / K
        # print(edge_s_idx)
        edge_s_idx_list.append(edge_s_idx)

    return edge_s_idx_list


ori_image_name = 'frame0.jpg'  # the original image name
ori_img = cv2.imread(ori_image_name)  # read the original image
# ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
dst_image_name = 'frame1.jpg'  # the destination image name
dst_img = cv2.imread(dst_image_name)  # read the destination image
# dst_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2GRAY)  # convert to grayscale



sift = cv2.SIFT_create()  # create a SIFT feature extractor
ori_kp, ori_des = sift.detectAndCompute(ori_img, None)  # get key points and feature descriptions of the original image
dst_kp, dst_des = sift.detectAndCompute(dst_img, None)  # get key points and feature descriptions of the destination image

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)  # create a Brute-Force matcher
try:
    bf_matches = bf.match(ori_des, dst_des)  # get the initial matches
except:
    print(ori_des, dst_des)

# extract the coordinates of original match points
ori_cord_list = np.float32([ori_kp[m.queryIdx].pt for m in bf_matches]).reshape(-1, 1, 2)
# extract the coordinates of destination match points
dst_cord_list = np.float32([dst_kp[m.trainIdx].pt for m in bf_matches]).reshape(-1, 1, 2)

# calculate the distance matrix of original match points
ori_dis_mat = distance_cal(ori_cord_list)
# calculate the distance matrix of destination match points
dst_dis_mat = distance_cal(dst_cord_list)

iter_num = 2  # the iterate number
for iter in range(iter_num):
    ini_match_number = len(ori_cord_list)  # the number of initial match points
    print(ini_match_number)
    K_list = [7, 10, 13]  # the set of K
    lambda_ = 0.3  # the parameter lambda
    flitered_match = []  # the list of filtered match points
    # iterate over the initial match points
    for idx in range(0, ini_match_number):
        s_idx = 0  # the match score of the i-th node
        # iterate over the set of K
        for K in K_list:
            # search the K nearest neighbors of the i-th node vi
            ori_K_idx, ori_K_dis, sorted_ori_idx = find_KNN(K, ori_dis_mat, idx)
            # search the K nearest neighbors of the corresponding node v'i
            dst_K_idx, dst_K_dis, sorted_dst_idx = find_KNN(K, dst_dis_mat, idx)
            # print(ori_K_idx, des_K_idx)
            # construct the local graph
            # if the neighbor of node vi is also in the corresponding node v'i neighborhoods, x_i = 1, else x_i = 0
            x_idx = [1 if i in dst_K_idx else 0 for i, j in zip(ori_K_idx, dst_K_idx)]
            # calculate the ranking shift of the i-th node vi
            ori_phi_idx = cal_ranking_shift(ori_K_idx, sorted_dst_idx)
            # calculate the ranking shift of the corresponding node v'i
            dst_phi_idx = cal_ranking_shift(dst_K_idx, sorted_ori_idx)
            # calculate the node affinity score of nodes vi and v'i
            node_s_idx = cal_node_affinity_score(ori_phi_idx, dst_phi_idx, K)
            # calculate the edge affinity score of nodes vi and v'i
            edge_s_idx_list = cal_edge_affinity_score(idx, x_idx, ori_K_idx, dst_K_idx, ori_dis_mat, dst_dis_mat, K)
            edge_s_idx_list.insert(0, node_s_idx)
            # construct the local affinity vector w_i
            w_idx = edge_s_idx_list
            # calculate the match score s_i = w_i * x_i
            s_idx_K = np.matmul(np.array(w_idx), np.array(x_idx).T)
            # accumulate the match score
            s_idx += s_idx_K
        # calculate the mean match score among different K
        s_idx = s_idx / len(K_list)
        # if the match score is greater than the preset threshold lambda
        if s_idx > lambda_:
            # the match node vi will be remained
            flitered_match.append(idx)
    # update the original match points
    ori_cord_list = ori_cord_list[flitered_match]
    # update the destination match points
    dst_cord_list = dst_cord_list[flitered_match]
# get the final matches
final_match = [bf_matches[i] for i in flitered_match]
# calculate the Homography matrix
M, mask = cv2.findHomography(ori_cord_list, dst_cord_list, cv2.RANSAC, 5.0)

# flat the mask to 1-D vector and convert to a list
matchesMask = mask.ravel().tolist()

# get the size of original image
h, w = ori_img.shape[0], ori_img.shape[1]
# get the coordinates of four vertexes of the image
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
# calculate the transformed coordinates
dst = cv2.perspectiveTransform(pts, M)

# draw the transformed bounding box in the destination image
dst_img = cv2.polylines(dst_img, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)

# get the transformed original image
ori_img_wraped = cv2.warpPerspective(ori_img, M, ori_img.shape[1::-1], flags=cv2.INTER_LINEAR)

# the visualization of match result
draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask,  # draw only inliers
                   flags=2)

final_result = cv2.drawMatches(ori_img, ori_kp, dst_img, dst_kp, final_match , None, **draw_params)

# final = cv2.addWeighted(dst_img, 0.6, ori_img_wraped, 1, 0)
cv2.imwrite('./final_result.jpg', final_result)
cv2.imshow('final', final_result)
cv2.waitKey()








