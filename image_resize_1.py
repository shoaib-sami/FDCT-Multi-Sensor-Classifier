from PIL import Image
import os, sys

path = "/media/shoaibmerajsami/SMS/ATR_database_october_2021/Mid Wave class basis crop Images/Truck/"
dirs = os.listdir(path)
dest = '/media/shoaibmerajsami/SMS/ATR_database_october_2021/Mid Wave class basis crop Images/resize_truck_new/'
def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((72,72))
            imResize.save(dest+'resized_'+ item)

resize()