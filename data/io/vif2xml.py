# -*-encoding:utf-8-*-
import os
from xml.dom.minidom import Document
from xml.dom.minidom import parse
import xml.dom.minidom
import scipy.misc as misc
import math
import cv2
import numpy as np

# the data root folder
VIF_ROOT_FOLDER = 'E:/demo/feiji_vif/'

# the source images root folder
SRC_IMAGE_ROOT_FOLDER = 'E:/demo/feiji_image/'

DES_IMAGE_ROOT_FOLDER = "E:/demo/feiji_head/VOCdevkit/JPEGImages/"

DES_XML_ROOT_FOLDER = "E:/demo/feiji_head/VOCdevkit/Annotations/"

DELTA = 0.00001

# get file names in given working folder
import shutil
def GetVIFFiles():
    global VIF_ROOT_FOLDER
    file_list = []
    files = os.listdir(VIF_ROOT_FOLDER)
    for file in files:
        if file.endswith(".vif"):
            #shutil.move(os.path.join(DATA_ROOT_FOLDER, file), os.path.join(DATA_ROOT_FOLDER, file.replace('.tif', '.tiff')))
            file_list.append(file)
    return file_list


def CalculateCenterPoint(geo_points):
    centerPoint = []
    centerPoint_x = (float(geo_points[0][0]) + float(geo_points[1][0]) + float(geo_points[2][0]) + float(geo_points[3][0])) / 4
    centerPoint_y = (float(geo_points[0][1]) + float(geo_points[1][1]) + float(geo_points[2][1]) + float(geo_points[3][1])) / 4
    centerPoint.append(centerPoint_x)
    centerPoint.append(centerPoint_y)
    return centerPoint


def CalculateTopCenterPoint(pointLU, pointRU):
    topCenterPoint = []
    topCenterPoint_x = (float(pointLU[0]) + float(pointRU[0])) / 2
    topCenterPoint_y = (float(pointLU[1]) + float(pointRU[1])) / 2
    topCenterPoint.append(topCenterPoint_x)
    topCenterPoint.append(topCenterPoint_y)
    return topCenterPoint


def GetQuadrant(topCenterPoint, centerPoint):
    difX = topCenterPoint[0] - centerPoint[0]
    difY = topCenterPoint[1] - centerPoint[1]
    if difX >= 0 and difY > 0:
        return 1
    elif difX > 0 and difY <= 0:
        return 4
    elif difX <= 0 and difY < 0:
        return 3
    else:
        return 2


def GetrotateAngle(topCenterPoint, centerPoint):
    RecQuadrant = GetQuadrant(topCenterPoint, centerPoint)
    width = abs(topCenterPoint[0] - centerPoint[0])
    height = abs(topCenterPoint[1] - centerPoint[1])
    if width <= DELTA:
        if RecQuadrant == 1 or 2:
            return 0
        else:
            return math.pi
    if height <= DELTA:
        if RecQuadrant == 1 or 4:
            return math.pi / 2
        else:
            return math.pi * 3 / 2

    angle = math.atan(width / height)
    if RecQuadrant == 1:
        return math.pi + angle
    elif RecQuadrant == 2:
        return math.pi - angle
    elif RecQuadrant == 3:
        return angle
    else:
        return math.pi * 2 - angle


def GetAngle(geo_points):
    centerPoint = CalculateCenterPoint(geo_points)
    topCenterPoint = CalculateTopCenterPoint(geo_points[0], geo_points[1])
    rotateAngle = int(GetrotateAngle(topCenterPoint, centerPoint) * 180 / math.pi)
    # print(rotateAngle)
    return rotateAngle


def ReadVIFFiles(vif_path=None, name=None):
    if name is None:
        print("Invalid input parameters, please check!")

    doc = Document()
    root_node = doc.createElement("Objects")
    doc.appendChild(root_node)

    dom = parse(vif_path)
    root = dom.documentElement
    children = root.getElementsByTagName('Child')
    geo_label, geo_points, geo_angle = [], [], []
    for child in children:

        points = child.getElementsByTagName('GeoShapePoint')
        text_label = child.getAttribute('name')

        geo_point = []
        for point in points:
            text_x = point.getAttribute("x")
            text_y = point.getAttribute("y")
            geo_point.append(abs(int(eval(text_x))))
            geo_point.append(abs(int(eval(text_y))))

        angle = GetAngle(np.reshape(np.array(geo_point), [-1, 2]))
        geo_points.append(geo_point)
        geo_label.append(text_label)
        geo_angle.append(angle)
    return geo_points, geo_label, geo_angle


def WriterXMLFiles(filename, box_list, angle_list, label_list, width, height):

    # dict_box[filename]=json_dict[filename]
    doc = xml.dom.minidom.Document()
    root = doc.createElement('annotation')
    doc.appendChild(root)

    foldername = doc.createElement("folder")
    foldername.appendChild(doc.createTextNode("JPEGImages"))
    root.appendChild(foldername)

    nodeFilename = doc.createElement('filename')
    nodeFilename.appendChild(doc.createTextNode(filename))
    root.appendChild(nodeFilename)

    pathname = doc.createElement("path")
    pathname.appendChild(doc.createTextNode("xxxx"))
    root.appendChild(pathname)

    sourcename=doc.createElement("source")

    databasename = doc.createElement("database")
    databasename.appendChild(doc.createTextNode("Unknown"))
    sourcename.appendChild(databasename)

    annotationname = doc.createElement("annotation")
    annotationname.appendChild(doc.createTextNode("xxx"))
    sourcename.appendChild(annotationname)

    imagename = doc.createElement("image")
    imagename.appendChild(doc.createTextNode("xxx"))
    sourcename.appendChild(imagename)

    flickridname = doc.createElement("flickrid")
    flickridname.appendChild(doc.createTextNode("0"))
    sourcename.appendChild(flickridname)

    root.appendChild(sourcename)

    nodesize = doc.createElement('size')
    nodewidth = doc.createElement('width')
    nodewidth.appendChild(doc.createTextNode(str(width)))
    nodesize.appendChild(nodewidth)
    nodeheight = doc.createElement('height')
    nodeheight.appendChild(doc.createTextNode(str(height)))
    nodesize.appendChild(nodeheight)
    nodedepth = doc.createElement('depth')
    nodedepth.appendChild(doc.createTextNode(str(3)))
    nodesize.appendChild(nodedepth)
    root.appendChild(nodesize)

    segname = doc.createElement("segmented")
    segname.appendChild(doc.createTextNode("0"))
    root.appendChild(segname)

    for (box, angle, label) in zip(box_list, angle_list, label_list):

        nodeobject = doc.createElement('object')
        nodename = doc.createElement('name')
        nodename.appendChild(doc.createTextNode(str(label)))
        nodeobject.appendChild(nodename)
        nodebndbox = doc.createElement('bndbox')
        nodex1 = doc.createElement('x1')
        nodex1.appendChild(doc.createTextNode(str(box[0])))
        nodebndbox.appendChild(nodex1)
        nodey1 = doc.createElement('y1')
        nodey1.appendChild(doc.createTextNode(str(box[1])))
        nodebndbox.appendChild(nodey1)
        nodex2 = doc.createElement('x2')
        nodex2.appendChild(doc.createTextNode(str(box[2])))
        nodebndbox.appendChild(nodex2)
        nodey2 = doc.createElement('y2')
        nodey2.appendChild(doc.createTextNode(str(box[3])))
        nodebndbox.appendChild(nodey2)
        nodex3 = doc.createElement('x3')
        nodex3.appendChild(doc.createTextNode(str(box[4])))
        nodebndbox.appendChild(nodex3)
        nodey3 = doc.createElement('y3')
        nodey3.appendChild(doc.createTextNode(str(box[5])))
        nodebndbox.appendChild(nodey3)
        nodex4 = doc.createElement('x4')
        nodex4.appendChild(doc.createTextNode(str(box[6])))
        nodebndbox.appendChild(nodex4)
        nodey4 = doc.createElement('y4')
        nodey4.appendChild(doc.createTextNode(str(box[7])))
        nodebndbox.appendChild(nodey4)

        nodeheadx = doc.createElement('head_x')
        nodeheadx.appendChild(doc.createTextNode(str(box[8])))
        nodebndbox.appendChild(nodeheadx)
        nodeheady = doc.createElement('head_y')
        nodeheady.appendChild(doc.createTextNode(str(box[9])))
        nodebndbox.appendChild(nodeheady)

        # ang = doc.createElement('angle')
        # ang.appendChild(doc.createTextNode(str(angle)))
        # nodebndbox.appendChild(ang)
        nodeobject.appendChild(nodebndbox)
        root.appendChild(nodeobject)
    fp = open(DES_XML_ROOT_FOLDER + filename, 'w')
    doc.writexml(fp, indent='\n')
    fp.close()


def IsPointInRect(point, box):
    return (point[0] > box[0]) and (point[0] < box[2]) and (point[1] > box[1]) and (point[1] < box[3])


def CropImage(image_path, des_path, geo_points, geo_angle, geo_label, sub_image_h, sub_image_w, overlap_h, overlap_w):
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        image_h, image_w, _ = image.shape
        for hh in range(0, image_h, sub_image_h - overlap_h):
            if (hh + sub_image_h) > image_h:
                break
            for ww in range(0, image_w, sub_image_w - overlap_w):
                sub_image = image[hh:(hh+sub_image_h), ww:(ww+sub_image_w), :]
                if (ww+sub_image_w) > image_w:
                    break
                sub_boxes = []
                sub_labels = []
                sub_angles = []
                for inx, box in enumerate(geo_points):
                    box = np.array(box)
                    box = np.reshape(box, [-1, 2])
                    xmin = min(box[:-1, 0])
                    xmax = max(box[:-1, 0])
                    ymin = min(box[:-1, 1])
                    ymax = max(box[:-1, 1])
                    # if (int(IsPointInRect(box[0], [ww, hh, ww+sub_image_w, hh+sub_image_h])) +
                    #         int(IsPointInRect(box[1], [ww, hh, ww+sub_image_w, hh+sub_image_h])) +
                    #         int(IsPointInRect(box[2], [ww, hh, ww+sub_image_w, hh+sub_image_h])) +
                    #         int(IsPointInRect(box[3], [ww, hh, ww+sub_image_w, hh+sub_image_h]))) >= 2:
                    if (ww < xmin) and (hh < ymin) and (ww + sub_image_w > xmax) and (hh + sub_image_h > ymax):
                        box = np.array(box)
                        box = np.reshape(box, [-1, 2])
                        box[:, 0] -= ww
                        box[:, 1] -= hh
                        box = np.reshape(box, [-1, ])
                        sub_boxes.append(list(box))
                        sub_labels.append(geo_label[inx])
                        sub_angles.append(geo_angle[inx])
                if len(sub_labels) != 0:
                    sub_image_name = image_path.split('/')[-1].split('.')[0] + '%{}%{}'.format(ww, hh) + '.jpg'
                    sub_xml_name = image_path.split('/')[-1].split('.')[0] + '%{}%{}'.format(ww, hh) + '.xml'
                    cv2.imwrite(des_path + sub_image_name, sub_image)
                    WriterXMLFiles(sub_xml_name, sub_boxes, sub_angles, sub_labels, sub_image_w, sub_image_h)


def FilterObject(geo_points, geo_label, geo_angle):
    p, l, a = [], [], []
    for inx, label in enumerate(geo_label):
        if label in ['M41', 'M603A', 'M48H']:
            p.append(geo_points[inx])
            l.append(geo_label[inx])
            a.append(geo_angle[inx])
    return p, l, a


def GetMinRect(geo_points):
    boxes = []
    for rect in geo_points:
        point = np.int0(rect[:])
        box = point.reshape([-1, 2])
        rect1 = cv2.minAreaRect(box)
        x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
        box = cv2.boxPoints(((x, y), (w, h), theta))
        box = np.reshape(box, [-1, ]).astype(np.int32)
        boxes.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7], point[0], point[1]])
    return boxes


if __name__ == "__main__":

    file_names = GetVIFFiles()
    for count, file_name in enumerate(file_names):
        vif_path = VIF_ROOT_FOLDER + file_name
        name = file_name.split('.')[0]
        geo_points, geo_label, geo_angle = ReadVIFFiles(vif_path, name)

        image_path = SRC_IMAGE_ROOT_FOLDER + name + '.jpg'
        sub_image_h = 600
        sub_image_w = 1000
        overlap_h = 200
        overlap_w = 200

        geo_points, geo_label, geo_angle = FilterObject(geo_points, geo_label, geo_angle)

        geo_points = GetMinRect(geo_points)

        if len(geo_label) != 0:
            CropImage(image_path, DES_IMAGE_ROOT_FOLDER, geo_points, geo_angle, geo_label,
                      sub_image_h, sub_image_w, overlap_h, overlap_w)

        if count % 200 == 0:
            print(count)


