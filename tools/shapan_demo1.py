# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import sys
sys.path.append('../')
import random
import matplotlib.pyplot as plt
# from osgeo import gdal, gdalconst
import xml.dom.minidom
import time
from timeit import default_timer as timer
import cv2
from data.io import image_preprocess
from libs.networks.network_factory import get_network_byname
from libs.rpn import build_rpn
from libs.fast_rcnn import build_fast_rcnn
from tools import restore_model
from libs.configs import cfgs
from help_utils.tools import *
from help_utils.help_utils import *
from libs.label_name_dict.label_dict import *
import shutil
from tools.shapan_utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def writer_XML(filename, box_list, label_list, width, height):

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

    for (box, label) in zip(box_list, label_list):

        nodeobject = doc.createElement('object')
        nodename = doc.createElement('name')
        nodename.appendChild(doc.createTextNode(LABEl_NAME_MAP[label]))
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

        node_x_c = doc.createElement('x_c')
        node_x_c.appendChild(doc.createTextNode(str(box[8])))
        nodebndbox.appendChild(node_x_c)
        node_y_c = doc.createElement('y_c')
        node_y_c.appendChild(doc.createTextNode(str(box[9])))
        nodebndbox.appendChild(node_y_c)

        node_head_x = doc.createElement('head_c')
        node_head_x.appendChild(doc.createTextNode(str(box[10])))
        nodebndbox.appendChild(node_head_x)
        node_head_y = doc.createElement('head_c')
        node_head_y.appendChild(doc.createTextNode(str(box[11])))
        nodebndbox.appendChild(node_head_y)

        nodeobject.appendChild(nodebndbox)
        root.appendChild(nodeobject)
    fp = open(filename, 'w')
    doc.writexml(fp, indent='\n')
    fp.close()


def writer_XML2(filename, box_list, label_list):

    doc = xml.dom.minidom.Document()
    root_node = doc.createElement('ImageInfo')

    root_node.setAttribute('resolution', str(0.6000))
    root_node.setAttribute('imagingtime', time.strftime('%Y-%m-%dT%H:%M:%S'))

    doc.appendChild(root_node)

    BaseInfo_node = doc.createElement('BaseInfo')

    BaseInfo_node.setAttribute('description', '')
    BaseInfo_node.setAttribute('ID', 'JP14000006')
    BaseInfo_node.setAttribute('name', 'Taiwan')

    root_node.appendChild(BaseInfo_node)

    result_node = doc.createElement('result')

    DetectNumber_node = doc.createElement('DetectNumber')
    DetectNumber_value = doc.createTextNode(str(len(label_list)))
    DetectNumber_node.appendChild(DetectNumber_value)

    result_node.appendChild(DetectNumber_node)
    airplane = ["F-16", "E-2T", "mirage2000"]
    ship = ["Tuojiang", "bison hovercraft", "072A"]
    tank = ["M41", "M48H", "M603A"]
    count = 0
    for (box, label) in zip(box_list, label_list):
        count += 1
        DetectResult_node = doc.createElement('DetectResult')

        ResultID_node = doc.createElement('ResultID')
        ResultID_value = doc.createTextNode(str(count))
        ResultID_node.appendChild(ResultID_value)

        box = np.reshape(box, [-1, 2])
        Shape_node = doc.createElement('Shape')
        for b in box:
            Point_node = doc.createElement('Point')
            Point_value = doc.createTextNode(str(b[0])+','+str(b[1]))
            Point_node.appendChild(Point_value)
            Shape_node.appendChild(Point_node)

        Location_node = doc.createElement('Location')
        Location_value = doc.createTextNode('unknown')
        Location_node.appendChild(Location_value)

        CenterLonLat_node = doc.createElement('CenterLonLat')
        CenterLonLat_value = doc.createTextNode('0.000000, 0.000000')
        CenterLonLat_node.appendChild(CenterLonLat_value)

        Length_node = doc.createElement('Length')
        Length_value = doc.createTextNode('0')
        Length_node.appendChild(Length_value)

        Width_node = doc.createElement('Width')
        Width_value = doc.createTextNode('0')
        Width_node.appendChild(Width_value)

        Area_node = doc.createElement('Area')
        Area_value = doc.createTextNode('0')
        Area_node.appendChild(Area_value)

        Angle_node = doc.createElement('Angle')
        Angle_value = doc.createTextNode(str(1.0))
        Angle_node.appendChild(Angle_value)

        Recorded_node = doc.createElement('Recorded')
        Recorded_value = doc.createTextNode('0')
        Recorded_node.appendChild(Recorded_value)

        ValidationType_node = doc.createElement('ValidationType')
        ValidationType_value = doc.createTextNode('0')
        ValidationType_node.appendChild(ValidationType_value)

        Weapontype_node = doc.createElement('Weapontype')
        Weapontype_value = doc.createTextNode('0')
        Weapontype_node.appendChild(Weapontype_value)

        PossibleResults_node = doc.createElement('PossibleResults')

        Type_node = doc.createElement('Type')

        if LABEl_NAME_MAP[label] in airplane:
            Type_value = doc.createTextNode('airplane')
            Type_node.appendChild(Type_value)
        elif LABEl_NAME_MAP[label] in ship:
            Type_value = doc.createTextNode('ship')
            Type_node.appendChild(Type_value)
        elif LABEl_NAME_MAP[label] in tank:
            Type_value = doc.createTextNode('tank')
            Type_node.appendChild(Type_value)
        else:
            Type_value = doc.createTextNode('obstacle')
            Type_node.appendChild(Type_value)

        Class_node = doc.createElement('Class')
        Class_value = doc.createTextNode(LABEl_NAME_MAP[label])
        Class_node.appendChild(Class_value)

        Reliability_node = doc.createElement('Reliability')
        Reliability_value = doc.createTextNode(str(1.0))
        Reliability_node.appendChild(Reliability_value)

        Attribution_node = doc.createElement('Attribution')
        ShipLength_node = doc.createElement('ShipLength')
        ShipLength_value = doc.createTextNode('0')
        ShipLength_node.appendChild(ShipLength_value)

        ShipWidth_node = doc.createElement('ShipWidth')
        ShipWidth_value = doc.createTextNode('0')
        ShipWidth_node.appendChild(ShipWidth_value)
        Attribution_node.appendChild(ShipLength_node)
        Attribution_node.appendChild(ShipWidth_node)

        PossibleResults_node.appendChild(Type_node)
        PossibleResults_node.appendChild(Class_node)
        PossibleResults_node.appendChild(Reliability_node)
        PossibleResults_node.appendChild(Attribution_node)

        DetectResult_node.appendChild(ResultID_node)
        DetectResult_node.appendChild(Shape_node)
        DetectResult_node.appendChild(Location_node)
        DetectResult_node.appendChild(CenterLonLat_node)
        DetectResult_node.appendChild(Length_node)
        DetectResult_node.appendChild(Width_node)
        DetectResult_node.appendChild(Area_node)
        DetectResult_node.appendChild(Angle_node)
        DetectResult_node.appendChild(Recorded_node)
        DetectResult_node.appendChild(ValidationType_node)
        DetectResult_node.appendChild(Weapontype_node)
        DetectResult_node.appendChild(PossibleResults_node)

        result_node.appendChild(DetectResult_node)

    root_node.appendChild(result_node)
    fp = open(filename, 'w')
    doc.writexml(fp, indent='\n')
    fp.close()


def get_points(boxes, head_quadrant):
    geo_boxes = []
    for i, rect in enumerate(boxes):
        y_c, x_c, h, w, theta = rect[0], rect[1], rect[2], rect[3], rect[4]
        box = cv2.boxPoints(((x_c, y_c), (w, h), theta))
        # [x1, y1, x2, y2, x3, y3, x4, y4]
        box = list(np.reshape(box, [-1, ]))
        # [x1, y1, x2, y2, x3, y3, x4, y4, x_c, y_c]
        box.extend([x_c, y_c])

        if w > h:
            head_x, head_y = w / 2. + np.sqrt(8.) * h / 2., 0.
            if head_quadrant[i] == 1:
                angle = theta
            else:
                angle = theta - 180
        else:
            head_x, head_y = 0, h / 2. + np.sqrt(8.) * w / 2.
            if head_quadrant[i] == 0:
                angle = theta
            else:
                angle = theta - 180
        angle = angle / 180. * math.pi

        head_x_ = np.cos(angle) * head_x - np.sin(angle) * head_y + x_c
        head_y_ = np.sin(angle) * head_x + np.cos(angle) * head_y + y_c

        box.extend([head_x_, head_y_])
        geo_boxes.append(np.array(box, np.int32))

    return geo_boxes


def detect_img(file_paths, des_folder, paramPath, bakpath, det_th, h_len, w_len, h_overlap, w_overlap, file_ext, show_res=False):
    with tf.Graph().as_default():

        img_plac = tf.placeholder(shape=[None, None, 3], dtype=tf.uint8)

        img_tensor = tf.cast(img_plac, tf.float32) - tf.constant([103.939, 116.779, 123.68])
        img_batch = image_preprocess.short_side_resize_for_inference_data(img_tensor,
                                                                          target_shortside_len=cfgs.SHORT_SIDE_LEN,
                                                                          is_resize=False)

        # ***********************************************************************************************
        # *                                         share net                                           *
        # ***********************************************************************************************
        _, share_net = get_network_byname(net_name=cfgs.NET_NAME,
                                          inputs=img_batch,
                                          num_classes=None,
                                          is_training=True,
                                          output_stride=None,
                                          global_pool=False,
                                          spatial_squeeze=False)
        # ***********************************************************************************************
        # *                                            RPN                                              *
        # ***********************************************************************************************
        rpn = build_rpn.RPN(net_name=cfgs.NET_NAME,
                            inputs=img_batch,
                            gtboxes_and_label=None,
                            is_training=False,
                            share_head=cfgs.SHARE_HEAD,
                            share_net=share_net,
                            stride=cfgs.STRIDE,
                            anchor_ratios=cfgs.ANCHOR_RATIOS,
                            anchor_scales=cfgs.ANCHOR_SCALES,
                            scale_factors=cfgs.SCALE_FACTORS,
                            base_anchor_size_list=cfgs.BASE_ANCHOR_SIZE_LIST,  # P2, P3, P4, P5, P6
                            level=cfgs.LEVEL,
                            top_k_nms=cfgs.RPN_TOP_K_NMS,
                            rpn_nms_iou_threshold=cfgs.RPN_NMS_IOU_THRESHOLD,
                            max_proposals_num=cfgs.MAX_PROPOSAL_NUM,
                            rpn_iou_positive_threshold=cfgs.RPN_IOU_POSITIVE_THRESHOLD,
                            rpn_iou_negative_threshold=cfgs.RPN_IOU_NEGATIVE_THRESHOLD,
                            rpn_mini_batch_size=cfgs.RPN_MINIBATCH_SIZE,
                            rpn_positives_ratio=cfgs.RPN_POSITIVE_RATE,
                            remove_outside_anchors=False,  # whether remove anchors outside
                            rpn_weight_decay=cfgs.WEIGHT_DECAY[cfgs.NET_NAME])

        # rpn predict proposals
        rpn_proposals_boxes, rpn_proposals_scores = rpn.rpn_proposals()  # rpn_score shape: [300, ]

        # ***********************************************************************************************
        # *                                         Fast RCNN                                           *
        # ***********************************************************************************************
        fast_rcnn = build_fast_rcnn.FastRCNN(feature_pyramid=rpn.feature_pyramid,
                                             rpn_proposals_boxes=rpn_proposals_boxes,
                                             rpn_proposals_scores=rpn_proposals_scores,
                                             img_shape=tf.shape(img_batch),
                                             roi_size=cfgs.ROI_SIZE,
                                             roi_pool_kernel_size=cfgs.ROI_POOL_KERNEL_SIZE,
                                             scale_factors=cfgs.SCALE_FACTORS,
                                             gtboxes_and_label=None,
                                             gtboxes_and_label_minAreaRectangle=None,
                                             fast_rcnn_nms_iou_threshold=cfgs.FAST_RCNN_NMS_IOU_THRESHOLD,
                                             fast_rcnn_maximum_boxes_per_img=100,
                                             fast_rcnn_nms_max_boxes_per_class=cfgs.FAST_RCNN_NMS_MAX_BOXES_PER_CLASS,
                                             show_detections_score_threshold=det_th,
                                             # show detections which score >= 0.6
                                              num_classes=cfgs.CLASS_NUM,
                                             fast_rcnn_minibatch_size=cfgs.FAST_RCNN_MINIBATCH_SIZE,
                                             fast_rcnn_positives_ratio=cfgs.FAST_RCNN_POSITIVE_RATE,
                                             fast_rcnn_positives_iou_threshold=cfgs.FAST_RCNN_IOU_POSITIVE_THRESHOLD,
                                             # iou>0.5 is positive, iou<0.5 is negative
                                              use_dropout=cfgs.USE_DROPOUT,
                                             weight_decay=cfgs.WEIGHT_DECAY[cfgs.NET_NAME],
                                             is_training=False,
                                             level=cfgs.LEVEL,
                                             head_quadrant=None)

        fast_rcnn_decode_boxes, fast_rcnn_score, num_of_objects, detection_category, \
        fast_rcnn_decode_boxes_rotate, fast_rcnn_score_rotate, fast_rcnn_head_quadrant, \
        num_of_objects_rotate, detection_category_rotate = fast_rcnn.fast_rcnn_predict()

        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )

        restorer, restore_ckpt = restore_model.get_restorer()

        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init_op)
            if not restorer is None:
                restorer.restore(sess, restore_ckpt)
                print('restore model')

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)

            obstacle_points, obstacle_labels = read_vif('./sp_whole_reg.vif')
            while True:

                alldemo = os.listdir(file_paths)
                file_names = []
                for singledemo in alldemo:
                    singlepath = os.path.join(file_paths, singledemo)
                    file_names.append(singlepath)

                for img_path in file_names:
                    if img_path.endswith(('.jpg')):  # and f not in fs_found:
                        l_f = img_path + '.lock'
                        if os.path.exists(l_f):
                            time.sleep(0.01)
                            continue
                    # try:
                    start = timer()
                    img = cv2.imread(img_path)

                    box_res = []
                    label_res = []
                    score_res = []
                    box_res_rotate = []
                    label_res_rotate = []
                    score_res_rotate = []
                    head_rotate = []
                    imgH = img.shape[0]
                    imgW = img.shape[1]
                    for hh in range(0, imgH, h_len - h_overlap):
                        h_size = min(h_len, imgH - hh)
                        if h_size < 10:
                            break
                        for ww in range(0, imgW, w_len - w_overlap):
                            w_size = min(w_len, imgW - ww)
                            if w_size < 10:
                                break

                            src_img = img[hh:(hh + h_size), ww:(ww + w_size), :]

                            # boxes, labels, scores = sess.run([fast_rcnn_decode_boxes, detection_category, fast_rcnn_score],
                            #                                  feed_dict={img_plac: src_img})

                            boxes_rotate, labels_rotate, scores_rotate, _fast_rcnn_head_quadrant = \
                                sess.run([fast_rcnn_decode_boxes_rotate, detection_category_rotate,
                                          fast_rcnn_score_rotate,
                                          fast_rcnn_head_quadrant],
                                         feed_dict={img_plac: src_img})

                            # if len(boxes) > 0:
                            #     for ii in range(len(boxes)):
                            #         box = boxes[ii]
                            #         box[0] = box[0] + hh
                            #         box[1] = box[1] + ww
                            #         box[2] = box[2] + hh
                            #         box[3] = box[3] + ww
                            #         box_res.append(box)
                            #         label_res.append(labels[ii])
                            #         score_res.append(scores[ii])
                            if len(boxes_rotate) > 0:
                                for ii in range(len(boxes_rotate)):
                                    box_rotate = boxes_rotate[ii]
                                    box_rotate[0] = box_rotate[0] + hh
                                    box_rotate[1] = box_rotate[1] + ww
                                    box_res_rotate.append(box_rotate)
                                    label_res_rotate.append(labels_rotate[ii])
                                    score_res_rotate.append(scores_rotate[ii])
                                    head_rotate.append(_fast_rcnn_head_quadrant[ii])

                    time_elapsed = timer() - start
                    print("{} detection time : {:.4f} sec".format(img_path.split('/')[-1].split('.')[0], time_elapsed))

                    mkdir(des_folder)

                    if len(head_rotate) != 0:
                        # img_np = draw_box_cv(np.array(img, np.float32) - np.array([103.939, 116.779, 123.68]),
                        #                      boxes=np.array(box_res),
                        #                      labels=np.array(label_res),
                        #                      scores=np.array(score_res))
                        img_np_rotate = draw_rotate_box_cv(np.array(img, np.float32) - np.array([103.939, 116.779, 123.68]),
                                                           boxes=np.array(box_res_rotate),
                                                           labels=np.array(label_res_rotate),
                                                           scores=np.array(score_res_rotate),
                                                           head=np.argmax(head_rotate, axis=1))

                        geo_points = get_points(box_res_rotate, np.argmax(head_rotate, axis=1))
                        image_name = img_path.split('/')[-1]
                        xml_path_1 = os.path.join(des_folder, '1_' + image_name).replace(file_ext, ".xml")
                        param_path = os.path.join(paramPath, 'UAV_' + image_name).replace(file_ext, ".param")
                        x_tr, y_tr = get_param(param_path)
                        obstacle_left, obstacle_labels = filter_obstacle(obstacle_points, imgH, imgW, x_tr, y_tr)

                        ######################################################
                        # obstacle_left = []
                        # temp = np.array([[2233, 1013], [2196, 980], [2215, 959], [2252, 993]])
                        # for coord in temp:
                        #     coord_convet = convert_coordinate(coord, imgH, imgW, x_tr, y_tr)
                        #     obstacle_left.extend(coord_convet)
                        # geo_points, obstacle_labels = filter_obstacle(np.array(geo_points)[:, :8], imgH, imgW, x_tr, y_tr)
                        ######################################################

                        detect_res, label_res = get_detect_res(obstacle_left, obstacle_labels, geo_points, label_res_rotate, imgH, imgW, x_tr, y_tr)

                        # writer_XML(xml_name, geo_points, label_res, imgW, imgH)
                        writer_XML2(xml_path_1, detect_res, label_res)
                        shutil.move(img_path, os.path.join(bakpath, image_name))
                        # cv2.imwrite(des_folder + '/{}_horizontal_fpn.jpg'.format(img_path.split('/')[-1].split('.')[0]), img_np)
                        cv2.imwrite(des_folder + '/{}_rotate_fpn.jpg'.format(img_path.split('/')[-1].split('.')[0]), img_np_rotate)

                        final_points = []
                        final_labels = []
                        for type in range(3):
                            indx = np.where(np.equal(label_res_rotate, type))[0]
                            if len(indx) != 0:
                                box_res_rotate_ = np.array(box_res_rotate)[indx]
                                label_res_rotate_ = np.array(label_res_rotate)[indx]
                                head_rotate_ = np.array(np.argmax(head_rotate, axis=1))[indx]
                                angles_ = get_angles(box_res_rotate_[:, 4], head_rotate_)
                                convex_points_, center_point_, angle_ = get_convex_points(box_res_rotate_[:, :2], angles_)
                                head_ = get_head(center_point_, angle_)
                                all_points = []
                                for ii in box_res_rotate_:
                                    all_points.extend(convert_coordinate(ii, imgH, imgW, x_tr, y_tr))
                                all_points.extend(convert_coordinate(center_point_, imgH, imgW, x_tr, y_tr))
                                all_points.extend(convert_coordinate(head_, imgH, imgW, x_tr, y_tr))

                                final_points.append(all_points)
                                final_labels.append(type)
                        xml_path_2 = os.path.join(des_folder, '2_' + image_name).replace(file_ext, ".xml")
                        writer_XML2(xml_path_2, final_points, final_labels)

                    # except:
                    #     print("Get an error, filename: {}".format(img_path))
            coord.request_stop()
            coord.join(threads)

if __name__ == "__main__":
    demoPath = "/yangxue/FPN_v21/tools/shapan_src/"
    demoOutPath = "/yangxue/FPN_v21/tools/shapan_des/"
    paramPath = "/yangxue/FPN_v21/tools/shapan_param/"
    bakpath = "/yangxue/FPN_v21/tools/shapan_bak/"
    # args = parse_args()
    # print('Called with args:')
    # print(args)

    # detect_img(demoPath, demoOutPath, bakpath, args.det_th, args.h_len, args.w_len,
    #            args.h_overlap, args.w_overlap, args.image_ext, False)

    detect_img(demoPath, demoOutPath, paramPath, bakpath, 0.9, 600, 1000,
               200, 200, '.jpg', False)



