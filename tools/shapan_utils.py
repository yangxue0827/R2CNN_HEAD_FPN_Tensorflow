# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import cv2
import numpy as np
from xml.dom.minidom import Document, parse
import xml.dom.minidom


def read_vif(vif_path=None):
    dom = parse(vif_path)
    root = dom.documentElement
    children = root.getElementsByTagName('Child')
    geo_label, geo_points = [], []
    for child in children:

        points = child.getElementsByTagName('GeoShapePoint')
        text_label = child.getAttribute('name')

        geo_point = []
        for point in points:
            text_x = point.getAttribute("x")
            text_y = point.getAttribute("y")
            geo_point.append(abs(float(eval(text_x))))
            geo_point.append(abs(float(eval(text_y))))

        geo_points.append(geo_point)
        geo_label.append(text_label)
    return geo_points, geo_label


def get_param(param_path):
    f = open(param_path, 'r')
    line = f.readline()
    if len(line) < 5:
        print('Invalid transfroming parameters, ', param_path)
        return False
    line = line.replace('\n', '')
    x_tr = line.split('\t')
    line = f.readline()
    line = line.replace('\n', '')
    y_tr = line.split('\t')

    return x_tr, y_tr


def convert_coordinate(coordinate, img_h, img_w, x_tr, y_tr):
    """
    :param coordinate: pix coordinate [x, y]
    :param img_h: image height
    :param img_w: image width
    :param x_tr: x transformation matrix
    :param y_tr: y transformation matrix
    :return: Latitude and longitude [xx, yy]
    """

    x = float(coordinate[0]) - img_w * 0.5
    y = float(coordinate[1]) - img_h * 0.5

    xx = float(x_tr[0]) + float(x_tr[1]) * x + float(x_tr[2]) * y + float(x_tr[3]) * x * x + \
        float(x_tr[4]) * x * y + float(x_tr[5]) * y * y
    yy = float(y_tr[0]) + float(y_tr[1]) * x + float(y_tr[2]) * y + float(y_tr[3]) * x * x + \
        float(y_tr[4]) * x * y + float(y_tr[5]) * y * y
    return [xx, yy]


def filter_obstacle(obstacles, img_h, img_w, x_tr, y_tr):
    """
    :param obstacles Latitude and longitude [[x1, y1, x2, y2, x3, y3, x4, y4] ...]
    :param img_h: image height
    :param img_w: image width
    :param x_tr: x transformation matrix
    :param y_tr: y transformation matrix
    :return: Latitude and longitude [[x1, y1, x2, y2, x3, y3, x4, y4] ...]
    """

    region = []
    windows = [[0, 0], [0, img_w], [img_h, img_w], [img_h, 0]]
    for coord in windows:
        coord_convet = convert_coordinate(coord, img_h, img_w, x_tr, y_tr)
        region.append(coord_convet)
    region = np.array(region) * 1e6
    region = np.array(region, np.int32)
    obstacle_left, labels = [], []
    for obstacle in obstacles:
        region = np.reshape(region, [-1, 2])
        if cv2.pointPolygonTest(region, (int(obstacle[0] * 1e6), int(obstacle[1] * 1e6)), False) == 1 and \
                        cv2.pointPolygonTest(region, (int(obstacle[2] * 1e6), int(obstacle[3] * 1e6)), False) == 1 and \
                        cv2.pointPolygonTest(region, (int(obstacle[4] * 1e6), int(obstacle[5] * 1e6)), False) == 1 and \
                        cv2.pointPolygonTest(region, (int(obstacle[6] * 1e6), int(obstacle[7] * 1e6)), False) == 1:
            obstacle_left.append(obstacle)
            labels.append(4)
    return np.array(obstacle_left), np.array(labels)


def get_detect_res(obstacle, obstacle_label, tank, label, img_h, img_w, x_tr, y_tr):
    """
    :param obstacle: Latitude and longitude [[x1, y1, x2, y2, x3, y3, x4, y4] ...]
    :param tank: pix coordinate [[x1, y1, x2, y2, x3, y3, x4, y4, x_c, y_c, x_head, y_head] ...]
    :return: Latitude and longitude [[x1, y1, x2, y2, x3, y3, x4, y4] ...]
    """

    tank_coordinate = np.array(tank)[:, :8]
    tank_coordinate_convert = []
    for coord in tank_coordinate:
        coord_convert = []
        coord = np.reshape(coord, [-1, 2])
        for point in coord:
            coord_convert.append(convert_coordinate(point, img_h, img_w, x_tr, y_tr))
        tank_coordinate_convert.append(np.reshape(coord_convert, [-1, ]))
    if len(obstacle) == 0:
        return np.array(tank_coordinate_convert), np.array(label)
    else:
        return np.concatenate([np.array(tank_coordinate_convert), np.array(obstacle)], axis=0), \
               np.concatenate([np.array(label), np.array(obstacle_label)], axis=0),


def get_angles(thetas, heads):
    angles = []
    for i, th in enumerate(thetas):
        if heads[i] == 1:
            angles.append(th)
        elif heads[i] == 2:
            angles.append(th - 90)
        elif heads[i] == 3:
            angles.append(th + 180)
        elif heads[i] == 0:
            angles.append(th + 90)
    return np.array(angles)


def get_head(center_point, angle):
    """
    :param center_point: [x, y]
    :param angle: angle [-180, 180]
    :return: head point [x, y]
    """
    temp_point = [20, 0]
    angle = angle / 180. * 3.1415926
    head_x = np.cos(angle) * temp_point[0] - np.sin(angle) * temp_point[1] + center_point[0]
    head_y = np.sin(angle) * temp_point[0] + np.cos(angle) * temp_point[1] + center_point[1]
    return [head_x, head_y]


def get_convex_points(points, angles):
    dis = np.square(points[:, 0]) + np.square(points[:, 1])
    indx = np.argsort(dis)
    if 2 < len(angles) <= 4:
        indx = indx[1:-1]
    if len(angles) > 4:
        indx = indx[1:-2]
    # -1:顺时针, 0:逆时针
    convex_points = cv2.convexHull(points[indx], clockwise=0)
    convex_points = np.reshape(convex_points, [-1, 2])
    center_point = np.mean(convex_points, axis=0)
    angle = np.mean(angles[indx])
    return convex_points, center_point, angle


def filter_box(points, angles, threshold):
    while True:
        mean_point = np.mean(points, axis=0)
        dis = np.sqrt(np.square(points[:, 0] - mean_point[0]) + np.square(points[:, 1] - mean_point[1]))
        flag = np.less_equal(dis, threshold)
        if not (False in flag):
            break
        inx = np.argsort(-dis)[1:]
        points, angles = points[inx], angles[inx]
    return points, angles

if __name__ == '__main__':
    # points = np.array([[2224, 986], [2402, 1173], [2497, 1028], [3157, 1312], [2761, 1008],
    #                    [2496, 1028], [1911, 1272], [2728, 1272], [2065, 1641], [2194, 1491],
    #                    [2367, 1375], [3156, 1310], [2911, 1270], [2728, 1270], [2823, 1732],
    #                    [2068, 1640]])
    # thetas = np.array([-47, -31, -73, -56, -77, -69, -53, -60, -79, -61, -60, -58, -56, -65, -50, -79])
    # heads = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    # angles = get_angles(thetas, heads)
    # convex_points, center_point, angle = task3(points, angles)
    # print(convex_points)
    # print(center_point)
    # print(angle)

    # region = np.array([0, 0, 0, 2, 2, 3, 3, 2, 2, 0])
    # boxes = np.array([[2, 2, 2, 4, 4, 4, 4, 2]])
    # print(task2(boxes, region))

    # vif_path = r'C:\Users\yangxue\PycharmProjects\shapan_demo\sp_whole_reg.vif'
    # geo_points, geo_label = read_vif(vif_path)
    # print('debug')

    # print(get_head([0, 0], 45))

    points = np.array([[40, 40], [40, 35], [35, 40], [35, 35], [70, 70], [60, 60]])
    angles = np.array([1, 2, 3, 4, 5, 6])
    points, angles = filter_box(points, angles, 20)
    print(points, angles)
