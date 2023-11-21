import re
from flask import Flask, jsonify,request
import cv2
import layoutparser as lp
from paddleocr import PaddleOCR, draw_ocr
import tensorflow as tf
import numpy as np
import pandas as pd 
from pathlib import Path
import os
from pdf2image import convert_from_path
from functions import callAPI,deleteFile,find_indices,getIndexofItems,auto_detect_type,serialNumberLogic,blankSpaceLogic,getIndexofItems_2,blankSpaceLogic_2
import json
from flask_cors import CORS
from PIL import Image


app = Flask(__name__)
# app.debug = True
# Configure CORS
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/paddler", methods=['POST'])
def paddler():
    json_data = request.form.get('json_data')

    print(json_data)

    # trained_template = json.loads(callAPI(vendorcode))
    trained_template = json.loads(json_data)
    cropped_image_path = ""
    pdf_path = ""

    s_no = ""
    item_code = ""
    hsn_code = ""
    quantity = ""
    uom = ""
    unit_rate = ""
    amount = ""

    for ocr in trained_template:
        if 'label' in ocr:
            label_name = ocr['label']
            if label_name == 'detail_area':
                print("Label:", ocr['boundingPoly']['vertices'])

                # Define the coordinates
                start_x = ocr['boundingPoly']['vertices'][0]['x']  # Starting X-coordinate
                start_y = ocr['boundingPoly']['vertices'][0]['y']  # Starting Y-coordinate
                end_x = ocr['boundingPoly']['vertices'][1]['x']   # Ending X-coordinate
                end_y = ocr['boundingPoly']['vertices'][1]['y']   # Ending Y-coordinate

                print(start_x,start_y,end_x,end_y)
                # Get the PDF file from the request
                pdf = request.files['file']
                print(os.getcwd()+'\\'+pdf.filename )
                # Save the PDF file to a directory
                save_path = os.getcwd()+"\\output\\"+pdf.filename
                pdf_path = save_path
                pdf.save(save_path)

                # Replace these values with the desired dimensions
                new_width = 1722
                new_height = 2435

                # Load the PDF pages as images
                pdf_images = convert_from_path(save_path, 300)

                for idx, image in enumerate(pdf_images):
                    # Resize the image to the desired dimensions
                    resized_image = image.resize((new_width, new_height))
                    
                    # Save the resized image
                    output_path = os.path.join(os.getcwd(), "output", f"{idx + 1}.jpg")
                    crop_image = resized_image.crop((start_x, start_y, end_x, end_y))
                    crop_image.save(output_path, dpi=(300, 300))
                    cropped_image_path = output_path
            if 'value' in ocr:
                if label_name == "s_no_1":
                    s_no = ocr['value']
                elif label_name == "PRODUCT_1":
                    item_code = ocr['value']
                elif label_name == "HSN_CODE_1":
                    hsn_code = ocr['value']
                elif label_name == "QUANTITY_1":
                    quantity = ocr['value']
                elif label_name == "UOM_1":
                    uom = ocr['value']
                elif label_name == "UNIT_RATE_1":
                    unit_rate = ocr['value']
                elif label_name == "AMOUNT_1":
                    amount = ocr['value']
            
    print(s_no,item_code,hsn_code,quantity,uom,unit_rate,amount)
    print(cropped_image_path)
    # return trained_template

    
    image = cv2.imread(cropped_image_path)
    image = image[..., ::-1]
    
    # load model
    model = lp.PaddleDetectionLayoutModel(
        config_path="lp://TableBank/ppyolov2_r50vd_dcn_365e_tableBank_word/config",
        threshold=0.5,
        label_map={0: "Table"},
        enforce_cpu=True,
        enable_mkldnn=True
    )

    # detect
    layout = model.detect(image)

    x_1=0
    y_1=0
    x_2=0
    y_2=0

    for l in layout:
        #print(l)
        if l.type == 'Table':
            x_1 = int(l.block.x_1)
            print(l.block.x_1)
            y_1 = int(l.block.y_1)
            x_2 = int(l.block.x_2)
            y_2 = int(l.block.y_2)
            
            break
    
    # Printing the values
    print(f'x_1: {x_1}')
    print(f'y_1: {y_1}')
    print(f'x_2: {x_2}')
    print(f'y_2: {y_2}')

    # Corrected slicing and saving the image
    # cv2.imwrite('output.jpg', image[y_1:y_2, x_1:x_2])

    ocr = PaddleOCR(lang='en')
    image_path = cropped_image_path
    image_cv = cv2.imread(image_path)
    image_height = image_cv.shape[0]
    image_width = image_cv.shape[1]
    output = ocr.ocr(image_path)[0]

    # print(output)
    boxes = [line[0] for line in output]
    texts = [line[1][0] for line in output]
    probabilities = [line[1][1] for line in output]

    image_boxes = image_cv.copy()

    for box,text in zip(boxes,texts):
        cv2.rectangle(image_boxes, (int(box[0][0]),int(box[0][1])), (int(box[2][0]),int(box[2][1])),(0,0,255),1)
        cv2.putText(image_boxes, text,(int(box[0][0]),int(box[0][1])),cv2.FONT_HERSHEY_SIMPLEX,1,(222,0,0),1)

    # cv2.imwrite('detections.jpg', image_boxes)

    im = image_cv.copy()
    horiz_boxes = []
    vert_boxes = []

    for box in boxes:
        x_h, x_v = 0,int(box[0][0])
        y_h, y_v = int(box[0][1]),0
        width_h,width_v = image_width, int(box[2][0]-box[0][0])
        height_h,height_v = int(box[2][1]-box[0][1]),image_height

        horiz_boxes.append([x_h,y_h,x_h+width_h,y_h+height_h])
        vert_boxes.append([x_v,y_v,x_v+width_v,y_v+height_v])

        cv2.rectangle(im,(x_h,y_h), (x_h+width_h,y_h+height_h),(0,0,255),1)
        cv2.rectangle(im,(x_v,y_v), (x_v+width_v,y_v+height_v),(0,255,0),1)

    # cv2.imwrite('horiz_vert.jpg',im)

    
    horiz_out = tf.image.non_max_suppression(
        horiz_boxes,
        probabilities,
        max_output_size = 1000,
        iou_threshold=0.1,
        score_threshold=float('-inf'),
        name=None
    )

    horiz_lines = np.sort(np.array(horiz_out))
    print(horiz_lines)

    im_nms = image_cv.copy()

    for val in horiz_lines:
        cv2.rectangle(im_nms, (int(horiz_boxes[val][0]),int(horiz_boxes[val][1])), (int(horiz_boxes[val][2]),int(horiz_boxes[val][3])),(0,0,255),1)
    
    # cv2.imwrite('im_nms.jpg',im_nms)

    vert_out = tf.image.non_max_suppression(
        vert_boxes,
        probabilities,
        max_output_size = 1000,
        iou_threshold=0.1,
        score_threshold=float('-inf'),
        name=None
    )

    print(vert_out)

    vert_lines = np.sort(np.array(vert_out))
    print(vert_lines)

    for val in vert_lines:
        cv2.rectangle(im_nms, (int(vert_boxes[val][0]),int(vert_boxes[val][1])), (int(vert_boxes[val][2]),int(vert_boxes[val][3])),(255,0,0),1)

    # cv2.imwrite('im_nms.jpg',im_nms)

    out_array = [["" for i in range(len(vert_lines))] for j in range(len(horiz_lines))]
    print(np.array(out_array).shape)
    print(out_array)

    unordered_boxes = []

    for i in vert_lines:
        print(vert_boxes[i])
        unordered_boxes.append(vert_boxes[i][0])

    ordered_boxes = np.argsort(unordered_boxes)
    print(ordered_boxes)

    def intersection(box_1, box_2):
        return [box_2[0], box_1[1],box_2[2], box_1[3]]
    
    def iou(box_1, box_2):
        x_1 = max(box_1[0], box_2[0])
        y_1 = max(box_1[1], box_2[1])
        x_2 = min(box_1[2], box_2[2])
        y_2 = min(box_1[3], box_2[3])

        inter = abs(max((x_2 - x_1, 0)) * max((y_2 - y_1), 0))
        if inter == 0:
            return 0
        
        box_1_area = abs((box_1[2] - box_1[0]) * (box_1[3] - box_1[1]))
        box_2_area = abs((box_2[2] - box_2[0]) * (box_2[3] - box_2[1]))
        
        return inter / float(box_1_area + box_2_area - inter)
    
    for i in range(len(horiz_lines)):
        for j in range(len(vert_lines)):
            resultant = intersection(horiz_boxes[horiz_lines[i]], vert_boxes[vert_lines[ordered_boxes[j]]])

            for b in range(len(boxes)):
                the_box = [boxes[b][0][0],boxes[b][0][1],boxes[b][2][0],boxes[b][2][1]]
                if(iou(resultant,the_box)>0.1):
                    out_array[i][j] = texts[b]

    paddler_res=np.array(out_array)
    print(paddler_res)
    # pd.DataFrame(paddler_res).to_csv('output/sample.csv')

    response_lineItem = np.array2string(paddler_res)

    # ========================> External Code  <=====================
    # 1-
    saved = {
        "s_no" : re.sub(r'[^\w\s]|\n', '', s_no.replace(" ", "").lower()),
        "item_code" :  re.sub(r'[^\w\s]|\n', '',  item_code.replace(" ", "").lower()) ,
        "hsn_code" : re.sub(r'[^\w\s]|\n', '', hsn_code.replace(" ", "").lower()) ,
        "quantity" : re.sub(r'[^\w\s]|\n', '', quantity.replace(" ", "").lower()) ,
        "uom" : re.sub(r'[^\w\s]|\n', '', uom.replace(" ", "").lower()) ,
        "unit_rate" : re.sub(r'[^\w\s]|\n', '', unit_rate.replace(" ", "").lower()), 
        "amount" : re.sub(r'[^\w\s]|\n', '', amount.replace(" ", "").lower() )
    }

    print(saved)


    paddler_res_index = 0
    total_line_item = 0
    index_dict = {}
    serial_number_logic = []
    detail_payload = []



    # Iterate through the array and create JSON objects
    for index, row in enumerate(paddler_res):
        res = getIndexofItems(saved,row)
        print(res)
        if res is True:
            paddler_res_index = index
            print(row)

            find_indices(saved, np.array(row).tolist(),index_dict)
            break

    print(index_dict)
    print(len(index_dict))

    if len(index_dict) != 0 :

        for i in paddler_res:
            value, value_type = auto_detect_type(re.sub(r'[^\w\s]|\n', '', i[index_dict['s_no']]) )
            print(value_type == "int")
            print(f"Value: {value}, Type: {value_type}")
            if value_type == "int" or value_type == "float" and value > total_line_item:
                total_line_item = value
                serial_number_logic.append(value)

        # Data parsing from paddler response based on list index '
        def dataParser(count):
            count = count
            for i in range(paddler_res_index+1,len(paddler_res)):
                payload = {}
                print(type(paddler_res[i][index_dict['s_no']]))
                print(paddler_res[i][index_dict['s_no']])
                value, value_type = auto_detect_type(paddler_res[i][index_dict['s_no']])
                print(value,value_type)
                if value_type == 'float':
                    value = int(value)

                print(count)
                if value_type == 'int' or value_type == 'float':
                    if count == value:
                        payload["s_no"] = paddler_res[i][index_dict["s_no"]]
                        payload["item_code"] = paddler_res[i][index_dict["item_code"]]
                        payload["hsn_code"] = paddler_res[i][index_dict["hsn_code"]]
                        payload["quantity"] = re.sub("[^0-9.]", "", paddler_res[i][index_dict["quantity"]]).replace(",","")
                        payload["uom"] =  re.sub("[0-9]", "", paddler_res[i][index_dict["uom"]]) 
                        payload["unit_amount"]= re.sub("[^0-9.]", "",paddler_res[i][index_dict["unit_rate"]]).replace(",","")
                        payload["amount"] = paddler_res[i][index_dict["amount"]].replace(",","")
                        detail_payload.append(payload)
                        count = count + 1


       
        print("total line item in invoice : ", total_line_item)
        serialLogic,count = serialNumberLogic(serial_number_logic,0)


       

        if serialLogic:
            dataParser(count)
        else: 
            for i in range(paddler_res_index+1,len(paddler_res)):
                payload = {}
                blankSpace = blankSpaceLogic(paddler_res[i],index_dict)
                if blankSpace:
                    payload["s_no"] = paddler_res[i][index_dict["s_no"]]
                    payload["item_code"] = paddler_res[i][index_dict["item_code"]]
                    payload["hsn_code"] = paddler_res[i][index_dict["hsn_code"]]
                    payload["quantity"] = re.sub("[^0-9.]", "", paddler_res[i][index_dict["quantity"]]).replace(",","")
                    payload["uom"] =  re.sub("[0-9]", "", paddler_res[i][index_dict["uom"]]) 
                    payload["unit_amount"]= re.sub("[^0-9.]", "",paddler_res[i][index_dict["unit_rate"]]).replace(",","")
                    payload["amount"] = re.sub("[^0-9.]", "", paddler_res[i][index_dict["amount"]]).replace(",","")
                    detail_payload.append(payload)


        print(detail_payload)
            
        # if paddler_res[i][index_dict['s_no']] > total_line_item:
        #     total_line_item  =  paddler_res[i][index_dict['s_no']]
        # print(i[index_dict['s_no']])
        deleteFile(pdf_path)
        return detail_payload
    
    else : 
       
         # Iterate through the array and create JSON objects
        for index, row in enumerate(paddler_res):
            res = getIndexofItems_2(saved,row)
            print(res)
            if res is True:
                paddler_res_index = index
                print(row)

            # else:
            #     return {"error" : "Please train template"}
                find_indices(saved, np.array(row).tolist(),index_dict)
                break
        
        print(index_dict)

        if len(index_dict) != 0:

            if index_dict['s_no'] != None:
                for i in paddler_res:
                    value, value_type = auto_detect_type(re.sub(r'[^\w\s]|\n', '', i[index_dict['s_no']]) )
                    print(value_type == "int")
                    print(f"Value: {value}, Type: {value_type}")
                    if value_type == "int" or value_type == "float" and value > total_line_item:
                        total_line_item = value
                        serial_number_logic.append(value)
            else:
                for i in range(paddler_res_index+1,len(paddler_res)):
                    payload = {}
                    blankSpace = blankSpaceLogic_2(paddler_res[i],index_dict)
                    if blankSpace:
                        payload["s_no"] = "" if index_dict["s_no"] is None else paddler_res[i][index_dict["s_no"]]
                        payload["item_code"] = "" if index_dict["item_code"] is None else paddler_res[i][index_dict["item_code"]]
                        payload["hsn_code"] = paddler_res[i][index_dict["hsn_code"]]
                        payload["quantity"] = re.sub("[^0-9.]", "", paddler_res[i][index_dict["quantity"]]).replace(",","")
                        payload["uom"] =  re.sub("[0-9]", "", paddler_res[i][index_dict["uom"]]) 
                        payload["unit_amount"]= re.sub("[^0-9.]", "",paddler_res[i][index_dict["unit_rate"]]).replace(",","")
                        payload["amount"] = re.sub("[^0-9.]", "", paddler_res[i][index_dict["amount"]]).replace(",","")
                        detail_payload.append(payload)
            print(detail_payload)
           
            deleteFile(pdf_path)
            return detail_payload
        else:
            print({"error" : "Please Train data for details in template"})
            deleteFile(pdf_path)
            return {"error" : "Please Train data for details in template"}


if __name__ == "__main__":
    app.run(debug=False)