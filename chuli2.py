import os
from PIL import Image

dirname_read="E:\项目\学习1\Mask_RCNN\image\oc\wyh\\"  
dirname_write="E:\项目\学习1\Mask_RCNN\image\oc\yhw\\"
names=os.listdir(dirname_read)
count=0
for name in names:
    img=Image.open(dirname_read+name)
    name=name.split(".")
    if name[-1] == "png":
        name[-1] = "jpg"
        name = str.join(".", name)
        r,g,b,a=img.split()              
        img=Image.merge("RGB",(r,g,b))   
        to_save_path = dirname_write + name
        img.save(to_save_path)
        count+=1
        print(to_save_path, "------conut：",count)
    else:
        continue