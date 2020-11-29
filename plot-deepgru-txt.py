import numpy as np
import cv2
import glob
import imageio

for txtfile in glob.glob("data/LH7/**/*.txt"):
    fname = txtfile.split("/")[-1].split(".")[0]
    folder = "/".join(txtfile.split("/")[:-1])
    images = []
    with open(txtfile) as fp:
        content = fp.readlines()
        
        content_combined = ",".join(content).replace("\n","") 
        
        line_lst = content_combined.split(",")
        line_lst = [float(n) for n in line_lst]
        v = iter(line_lst)
        
        li = [(i, next(v), next(v)) for i in v]  # creates list of tuples
        min_x = min([pnt[0] for pnt in li])-0.1
        min_y = min([pnt[1] for pnt in li])-0.1
        max_x = max([pnt[0] for pnt in li])+0.1
        max_y = max([pnt[1] for pnt in li])+0.1
        
        w_x = abs(max_x-min_x) + 0.1
        w_y = abs(max_y-min_y) + 0.1
        
        for cnt, line in enumerate(content):
            line_lst = line.split("\n")[0].split(",")
            line_lst = [float(n) for n in line_lst]
            v = iter(line_lst)
            
            li = [(i, next(v), next(v)) for i in v]  # creates list of tuples

            
            print(cnt, li)
            
            scale = 100
           
            image = np.ones((int(w_y*100),int(w_x*100),3))*255
            print(min_x, max_x, w_x)
            print(min_y, max_y, w_y)
            for pnt in li:
                if pnt[0] != 0 or pnt[1] != 0:
                    x = int((pnt[0]-min_x)*100)
                    y = int((pnt[1]-min_y)*100)
                    print(x,y)
               
                    image = cv2.circle(image, (x,y), radius=2, color=(0, 0, 255), thickness=-1)
            # cv2.imwrite(f"{fname}-{cnt}.png",image)
            images.append(image.astype(np.uint8))
            

    with imageio.get_writer(f'{folder}/{fname}-skel.gif', mode='I') as writer:
        for image in images:
            writer.append_data(image)

