import requests
import os
import re

def callAPI(vendorcode):
    response = requests.get(f'http://192.168.50.81:8080/ap_automation_backend/ocrtraining/get?supplier={vendorcode}&template=Template_1')
    if response.status_code == 200:
        return response.text
    else:
        print(response.status_code)


# --->delete file<---
def deleteFile(pdf_path):
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    else:
        print("The file does not exist")


# --->Getting index of line items in paddler array<---
def find_indices(saved, key_list,index_dict):
    new_key_list = [  re.sub(r'[^\w\s]|\n', '', item.replace(" ", "").lower())  for item in key_list]
    print(new_key_list)
    for key, value in saved.items():
        print(f"key : {key}",f"value : {value}")
        if re.sub(r'[^\w\s]|\n', '', value) in new_key_list:
            index = new_key_list.index(re.sub(r'[^\w\s]|\n', '', value))
            index_dict[key] = index
        else:
            index_dict[key] = None
    return index_dict

 # --->Index of item start from<---
def getIndexofItems(saved, key_list):
    # Iterate through the keys in the saved dictionary
    print(key_list)
    for key in saved.values():
        filtered_list = [re.sub(r'[^\w\s]|\n', '',item.replace(" ", "").lower())for item in key_list]
        if re.sub(r'[^\w\s]|\n', '', key) not in filtered_list :
            return False
    return True

 # --->Data type detection<---
def auto_detect_type(value):
    try:
        # Attempt to convert the value to an integer
        int_value = int(value)
        return int_value, "int"
    except ValueError:
        try:
            # Attempt to convert the value to a floating-point number
            float_value = float(value)
            return float_value, "float"
        except ValueError:
            # If neither conversion works, consider it a string
            return value, "string"


 # Checking for serial number logic work or not
def serialNumberLogic(serial_number_logic,start_index):
    if len(serial_number_logic) > 0 : 
        for index,serial_number in enumerate(serial_number_logic):
            index = index + 1 + start_index
            print(f"index : {index} serial_number : {serial_number}")
            if index != serial_number:
                return False,index
        return True,1
    else : 
        return False,1


 # Function to check the indices for a given item
def blankSpaceLogic(item, indices):
        print(item)
        if item[indices["item_code"]] == "" or item[indices["quantity"]] == "" or item[indices["unit_rate"]] == "" or item[indices["amount"]] == "":
            return False
        else:
            return True


 # --->Index of item start from with 2nd logic<---
def getIndexofItems_2(saved, key_list):
    # Iterate through the keys in the saved dictionary
    print(key_list)
    print(len(saved))
    temp_len = 0
    for key in saved.values():
        print(key)
        filtered_list = [re.sub(r'[^\w\s]|\n', '',item.replace(" ", "").lower())for item in key_list]
        print(filtered_list)
        for data in filtered_list:
            if data == key:
                temp_len = temp_len+1
                break
        # if re.sub(r'[^\w\s]|\n', '', key) not in filtered_list :
        #     return False

    if  temp_len >= 5:
        return True
    else: 
        return False
    

  # Function to check the indices for a given item
def blankSpaceLogic_2(item, indices):
    for key, value in indices.items():
        print(f"key : {key}",f"value : {value}")

    if indices["item_code"] != None and indices["quantity"] != None and indices["unit_rate"] != None and indices["amount"] != None:
        if item[indices["item_code"]] == "" or item[indices["quantity"]] == "" or item[indices["unit_rate"]] == "" or item[indices["amount"]] == "":
            return False
        else:
            return True
    elif indices["quantity"] != None and indices["unit_rate"] != None and indices["amount"] != None:
        if item[indices["quantity"]] == "" or item[indices["unit_rate"]] == "" or item[indices["amount"]] == "":
            return False
        else:
            return True
    elif indices["item_code"] != None and indices["unit_rate"] != None and indices["amount"] != None:
        if item[indices["item_code"]] == "" or item[indices["unit_rate"]] == "" or item[indices["amount"]] == "":
            return False
        else:
            return True
    elif indices["item_code"] != None and indices["quantity"] != None and indices["amount"] != None:
        if item[indices["item_code"]] == "" or item[indices["quantity"]] == "" or item[indices["amount"]] == "":
            return False
        else:
            return True
    elif indices["item_code"] != None and indices["quantity"] != None and indices["unit_rate"] != None:
        if item[indices["item_code"]] == "" or item[indices["quantity"]] == "" or item[indices["unit_rate"]] == "":
            return False
        else:
            return True