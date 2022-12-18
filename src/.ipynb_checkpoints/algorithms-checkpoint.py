import numpy as np


def max_width(a):
    v = [0 for i in range(len(a))]
    stack = []
    for i in range(len(a)):
        if len(stack) == 0:
            stack.append((a[i], i))
        else:
            j = len(stack) - 1
            while (j >= 0 and stack[j][0] > a[i]):
                v[stack[j][1]] += i - stack[j][1]
                stack.pop()
                j -= 1
            stack.append((a[i], i))
    j = len(stack) - 1
    while (len(stack) != 0):
        v[stack[j][1]] += i - stack[j][1] + 1
        stack.pop()
        j -= 1
    return v


def total_max_width(a):
    """
    :param a: array of integers
    :return: max width of array of all value
    """
    first = max_width(a)
    second = max_width(a[::-1])[::-1]
    for i in range(len(second)):
        first[i] += second[i] - 1
    return np.array(first)


def add_block(wid, hi, polygon):
    """
    :param wid: width of rectangle
    :param hi:  height of rectangle
    :param polygon: the polygon where we want to put
    :return: polygon with rect and true/false (can be placed or not)
    """

    # MAIN ALG FOR ADD BLOCK
    polygon_width = polygon.copy()
    for j in range(polygon.shape[1] - 1, -1, -1):
        for i in range(0, polygon.shape[0], 1):
            if polygon_width[i][j] != 0:
                polygon_width[i][j] = polygon_width[i - 1][j] + polygon_width[i][j]

    for i in range(polygon_width.shape[0]):

        polygon_one = polygon_width[i]
        inds_wid = polygon_one >= wid
        inds_hi = polygon_one >= hi
        polygon_two = total_max_width(polygon_one)

        vls_first = polygon_two[inds_wid]
        vls_first = vls_first[vls_first >= hi]

        vls_second = polygon_two[inds_hi]
        vls_second = vls_second[vls_second >= wid]
        
        
        if len(vls_first) > 0:
            ind = np.where(polygon_two == vls_first[0])[0][0]
            ind_first = ind
            while True:
                ind_first -= 1
                if polygon_one[ind_first] < wid:
                    ind_first += 1
                    break  
                if ind_first == 0:
                    break
            
            for i_1 in range(i, i - wid, -1):
                for j_1 in range(ind_first, ind_first + hi - 1):
                    polygon[i_1][j_1] = 0
        elif len(vls_second) > 0:
            ind = np.where(polygon_two == vls_second[0])[0][0]
            ind_first = ind
            while True:
                ind_first -= 1
                if polygon_one[ind_first] < hi:
                    ind_first += 1
                    break     
                if ind_first == 0:
                    break
            for i_1 in range(i, i - hi, -1):
                for j_1 in range(ind_first, ind_first + wid - 1):
                    polygon[i_1][j_1] = 0
        if len(vls_first) + len(vls_second) > 0:
            return True
    return False
