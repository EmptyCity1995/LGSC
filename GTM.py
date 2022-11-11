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

# generate a median KNN graph for the given point set
def generate_median_KNN_graph(K, distance_matrix, match_number):
    # initialize the adjacency matrix Ap or Ap'
    adj_martix = np.zeros((match_number, match_number))
    # calculate the median distance \eta
    dis_median = np.median(distance_matrix)
    for i in range(0, match_number):
        # search the K nearest neighbors of the i-th node vi
        K_idx, K_dis, sorted_idx = find_KNN(K, ori_dis_mat, i)
        for j, dis in zip(K_idx, K_dis):
            # if j in the K nearest neighbors list and the distance between vi and vj is less than \eta
            if dis <= dis_median:
                # connect vi and vj with non-direction edge
                adj_martix[i][j] = 1
                adj_martix[j][i] = 1
    # return the adjacency matrix Ap or Ap'
    return adj_martix

# find the outlier
def find_outlier(ori_adj_martix, dst_adj_martix):
    # calculate the residual adjacency matrix R
    residual_adj_mat = abs(ori_adj_martix - dst_adj_martix)
    # find the outlier column that yields the maximal number of different edges in both graphs
    outlier_j = np.argmax(np.sum(residual_adj_mat, axis=1))
    # return the residual adjacency matrix R and the index of outlier column
    return residual_adj_mat, outlier_j

# remove the outlier from the distance matrix and point set
def remove_outlier(distance_matrix, cord_list, outlier_j):
    update_dis_martix = np.delete(distance_matrix, outlier_j, axis=0)
    update_dis_martix = np.delete(update_dis_martix, outlier_j, axis=1)
    update_cord_list = np.delete(cord_list, outlier_j, axis=0)
    return update_dis_martix, update_cord_list

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


match_number = len(ori_cord_list)  # the number of initial match points, N
flitered_match = [i for i in range(match_number)]  # the index list of match points
while True:
    K = 5  # the number of nearest neighbors
    ori_adj_martix = generate_median_KNN_graph(K, ori_dis_mat, match_number)  # the adjacency matrix of ori_graph, Ap
    dst_adj_martix = generate_median_KNN_graph(K, dst_dis_mat, match_number)  # the adjacency matrix of dst_graph, Ap'
    # the residual adjacency matrix and the outlier index, R and j^out
    residual_adj_mat, outlier_j = find_outlier(ori_adj_martix, dst_adj_martix)
    # if the residual adjacency matrix is a zero matrix
    if np.all(residual_adj_mat == 0):
        # stop iteration
        break
    # update the ori_dis_mat and ori_cord_list
    ori_dis_mat, ori_cord_list = remove_outlier(ori_dis_mat, ori_cord_list, outlier_j)
    # update the dst_dis_mat and dst_cord_list
    dst_dis_mat, dst_cord_list = remove_outlier(dst_dis_mat, dst_cord_list, outlier_j)
    match_number -= 1  # the number of match points is decreased by one
    flitered_match.pop(outlier_j) # remove the outlier index from the index list

# get the final matches
final_match = [bf_matches[i] for i in flitered_match]
print(len(final_match), final_match)
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
cv2.imwrite('./final_result_GTM.jpg', final_result)
cv2.imshow('final', final_result)
cv2.waitKey()






