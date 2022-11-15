import cv2
import math
import numpy as np

# calculate the Atan_{-PI,PI}, equation (8)
def Atan(vec_x):
    if vec_x[0] >= 0 :
        if vec_x[0] == 0 and vec_x[1] > 0:
            return np.pi/2
        elif vec_x[0] == 0 and vec_x[1] < 0:
            return -np.pi / 2
        elif vec_x[0] == 0 and vec_x[1] == 0:
            return 0
        else:
            return np.arctan(vec_x[1]/vec_x[0])
    elif vec_x[0] < 0 and vec_x[1] >= 0:
        return np.arctan(vec_x[1]/vec_x[0]) + np.pi
    else:
        return np.arctan(vec_x[1]/vec_x[0]) - np.pi


# calculate the theta(k,i), equation (7)
def theta_function(vec_k, vec_i):
    return Atan(vec_k - vec_i) - Atan(vec_k - vec_i)


# calculate the angle distance, equation (4)
def calculate_angle_distance(ori_vec_i, ori_vec_j, dst_vec_i, dst_vec_j, ori_vec_k):
    # calculate the Rot(), equation (5)
    Rot = np.array([[np.cos(theta_function(ori_vec_k, ori_vec_i)), np.sin(theta_function(ori_vec_k, ori_vec_i))],
                   [-np.sin(theta_function(ori_vec_k, ori_vec_i)), np.cos(theta_function(ori_vec_k, ori_vec_i))]])
    ori_vec_j_i = ori_vec_j - ori_vec_i  # the vector pj - pi
    dst_vec_j_i = dst_vec_j - dst_vec_i  # the vector pj' - pi'
    ori_dis_j_i = np.linalg.norm(ori_vec_j_i, axis=0)  # the L2-norm of vector pj - pi
    dst_dis_j_i = np.linalg.norm(dst_vec_j_i, axis=0)  # the L2-norm of vector pj' - pi'
    dst_vec_j_i = np.array([dst_vec_j_i])
    # calculate the cosine similarity of vector pj - pi and vector pj' - pi'
    cos_similarity = np.dot(ori_vec_j_i, np.dot(dst_vec_j_i, Rot)[0])/(ori_dis_j_i * dst_dis_j_i)
    if cos_similarity < -1:
        cos_similarity = -1
    elif cos_similarity > 1:
        cos_similarity = 1
    return np.abs(np.arccos(cos_similarity))

# calculate the optimal rotation angle k_min
def calculate_k_min(ori_cord_list, dst_cord_list, ori_adj_matrix):
    sum_min = 10000000  # initialize the minimize sum of angle distances
    # iterate all matching points pk
    for k in range(len(ori_cord_list)):
        # initialize the sum of angle distances
        sum_ang_dis = 0
        # iterate the k nearest neighbors of point pk
        for i in np.where(ori_adj_matrix[k] == 1)[0]:
            # iterate the k nearest neighbors of point pi
            for j in np.where(ori_adj_matrix[i] == 1)[0]:
                # calculate the angle distance of pj and pi with k
                ang_dis = calculate_angle_distance(ori_cord_list[i][0], ori_cord_list[j][0],
                                                   dst_cord_list[i][0], dst_cord_list[j][0], ori_cord_list[k][0])
                sum_ang_dis += ang_dis  # sum the angle distance
        if sum_ang_dis < sum_min:
            # if the current sum of angle distances is smaller than the minimize sum
            k_min = k  # get the minimize k, k_min
            sum_min = sum_ang_dis  # update the minimize sum
    return k_min

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

# generate a median KNN graph for the given point set
def generate_median_KNN_graph(K, distance_matrix, match_number):
    # initialize the adjacency matrix Ap or Ap'
    adj_matrix = np.zeros((match_number, match_number))
    # calculate the median distance \eta
    dis_median = np.median(distance_matrix)
    for i in range(0, match_number):
        # search the K nearest neighbors of the i-th node vi
        K_idx, K_dis, sorted_idx = find_KNN(K, ori_dis_mat, i)
        for j, dis in zip(K_idx, K_dis):
            # if j in the K nearest neighbors list and the distance between vi and vj is less than \eta
            if dis <= dis_median:
                # connect vi and vj with non-direction edge
                if i != j:
                    adj_matrix[i][j] = 1
                    # adj_matrix[j][i] = 1
    # return the adjacency matrix Ap or Ap'
    return adj_matrix

# find the outlier
def find_outlier(ori_adj_matrix, dst_adj_matrix):
    # calculate the residual adjacency matrix R
    residual_adj_mat = abs(ori_adj_matrix - dst_adj_matrix)
    # find the outlier column that yields the maximal number of different edges in both graphs
    outlier_j = np.argmax(np.sum(residual_adj_mat, axis=1))
    # return the residual adjacency matrix R and the index of outlier column
    return residual_adj_mat, outlier_j

# remove the outlier from the distance matrix and point set
def remove_outlier(distance_matrix, cord_list, adjacency_matrix, outlier_j):
    update_dis_matrix = np.delete(distance_matrix, outlier_j, axis=0)
    update_dis_matrix = np.delete(update_dis_matrix, outlier_j, axis=1)
    update_cord_list = np.delete(cord_list, outlier_j, axis=0)
    update_adj_matrix = np.delete(adjacency_matrix, outlier_j, axis=0)
    update_adj_matrix = np.delete(update_adj_matrix, outlier_j, axis=1)
    return update_dis_matrix, update_cord_list, update_adj_matrix

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
ori_pt_list = []  # create a list of original matching points
dst_pt_list = []  # create a list of destination matching points
filter_matches = [] # create a list of matches
for m in bf_matches:
    ori_pt_x = ori_kp[m.queryIdx].pt[0]  # get abscissas of original matching points
    ori_pt_y = ori_kp[m.queryIdx].pt[1]  # get ordinates of original matching points

    dst_pt_x = dst_kp[m.trainIdx].pt[0]  # get abscissas of destination matching points
    dst_pt_y = dst_kp[m.trainIdx].pt[1]  # get ordinates of destination matching points

    # filter out duplicate matching points
    if ([ori_pt_x, ori_pt_y] not in ori_pt_list) and ([dst_pt_x, dst_pt_y] not in dst_pt_list):
        ori_pt_list.append([ori_pt_x, ori_pt_y])
        dst_pt_list.append([dst_pt_x, dst_pt_y])  
        filter_matches.append(m)

# extract the coordinates of original match points
# ori_cord_list = np.float32([ori_kp[m.queryIdx].pt for m in bf_matches]).reshape(-1, 1, 2)
ori_cord_list = np.float32(ori_pt_list).reshape(-1, 1, 2)
# extract the coordinates of destination match points
# dst_cord_list = np.float32([dst_kp[m.trainIdx].pt for m in bf_matches]).reshape(-1, 1, 2)
dst_cord_list = np.float32(dst_pt_list).reshape(-1, 1, 2)


# calculate the distance matrix of original match points
ori_dis_mat = distance_cal(ori_cord_list)
# calculate the distance matrix of destination match points
dst_dis_mat = distance_cal(dst_cord_list)

match_number = len(ori_cord_list)  # the number of initial match points, N

K = 5
mu_old = 2 * np.pi
eps = 0.005
while True:
    match_number = len(ori_cord_list)
    ori_adj_matrix = generate_median_KNN_graph(K, ori_dis_mat, match_number)  # the adjacency matrix of ori_graph, Ap
    dst_adj_matrix = generate_median_KNN_graph(K, dst_dis_mat, match_number)  # the adjacency matrix of dst_graph, Ap'
    while True:
        # find all points of Gp with at most one edge with other points
        one_edge_list = [i for i, e in enumerate(np.sum(ori_adj_matrix, 1)) if e <= 1]
        if len(one_edge_list) != 0:
            # remove these points with at most one edge with other points from Gp and Gp'
            for i in one_edge_list:
                # update the ori_dis_mat, ori_cord_list, and ori_adj_matrix
                ori_dis_mat, ori_cord_list, ori_adj_matrix = remove_outlier(ori_dis_mat, ori_cord_list, ori_adj_matrix, i)
                # update the dst_dis_mat, dst_cord_list, and dst_adj_matrix
                dst_dis_mat, dst_cord_list, dst_adj_matrix = remove_outlier(dst_dis_mat, dst_cord_list, dst_adj_matrix, i)
                filter_matches.pop(i)
        else:
            break
    # calculate the optimal rotation angle k_min 
    k_min = calculate_k_min(ori_cord_list, dst_cord_list, ori_adj_matrix)
    # update the number of matching points 
    match_number = len(ori_cord_list)
    # create a weight matrix 
    w = np.zeros([match_number, match_number])
    # create a mean weight list
    w_mean_list = []
    # iterate all matching points pi
    for i in range(match_number):
        w_list = []
        # iterate the k nearest neighbors of point pi
        for m in np.where(ori_adj_matrix[i] == 1)[0]:
            # calculate the edge's weight W(i,m)
            w[i][m] = calculate_angle_distance(ori_cord_list[i][0], ori_cord_list[m][0],
                                               dst_cord_list[i][0], dst_cord_list[m][0], ori_cord_list[k_min][0])
            # find the percentage of edges connected to vi with their correspondences connected to vi'
            if np.sum(dst_adj_matrix[i]) / np.sum(ori_adj_matrix[i]) < 0.5:
                # if the percentage is smaller than 50%, the weight value is replaced by PI
                w[i][m] = np.pi
            w_list.append(w[i][m])
        # calculate the mean weight w(i) of point pi
        w_mean = np.mean(np.array(w_list))
        # print(w_list)
        w_mean_list.append(w_mean)
    # find the point corresponding to the maximum value of w
    outlier_j = np.argmax(w_mean_list)
    print(outlier_j)
    # remove the point and its corresponding point from Gp and Gp'
    ori_dis_mat, ori_cord_list, ori_adj_matrix = remove_outlier(ori_dis_mat, ori_cord_list, ori_adj_matrix, outlier_j)
    dst_dis_mat, dst_cord_list, dst_adj_matrix = remove_outlier(dst_dis_mat, dst_cord_list, dst_adj_matrix, outlier_j)
    filter_matches.pop(outlier_j)  # update the matches list
    w_max = np.max(w)  # find the maximum element of W, w_max
    mu_new = np.mean(np.array(w_mean_list))  # calculate the mean value of w(i), mu_new
    print(w_max, np.abs(mu_new - mu_old), match_number)

    if w_max < np.pi and np.abs(mu_new - mu_old) < eps:
        # if w_max < PI and abs(mu_new - mu_old ) < epsilon, the iteration stop
        break
    else:
        # update the mu_old
        mu_old = mu_new
# get the final matches
final_match = filter_matches

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
draw_params = dict(
                   matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask,  # draw only inliers
                   flags=2)

final_result = cv2.drawMatches(ori_img, ori_kp, dst_img, dst_kp, final_match , None, **draw_params)

# final = cv2.addWeighted(dst_img, 0.6, ori_img_wraped, 1, 0)
cv2.imwrite('./final_result_WGTM.jpg', final_result)
cv2.imshow('final', final_result)
cv2.waitKey()



