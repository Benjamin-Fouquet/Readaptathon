import json
from turtle import width
import napari
import numpy as np
import os
import skvideo.io  
from scipy.interpolate import griddata
from skimage.filters import gaussian
import cv2
frames=1000
# for frame in videogen:
#     video.append(frame[...,0:1].astype(np.uint8))
#     if i%100==0:
#         print(i)
#     i+=1

# video[:,950:1130,430:550]=255
points=[]
timestps=[]
keypoints='derivatives-one-skeleton/020302_aha_j15.json'
files=os.listdir(keypoints)
files.sort()
for f in files:
    timestp=int(f.split('_')[1])
    timestps.append(timestp)
    with open(os.path.join(keypoints,f),'r') as f:
        data=json.load(f)
        point=np.array(data['people'][0]['pose_keypoints_2d']).reshape(-1,3)
        point=np.concatenate((np.tile(np.array([timestp]),(point.shape[0],1)),point),axis=1)
        points.append(point)

points=np.stack(points).astype('int')

# video= skvideo.io.vread("02.03.02_AHA_J15.mp4",num_frames=frames,as_grey=False)#[...,0]
video=[]
# videogen = skvideo.io.vreader("02.03.02_AHA_J15.mp4",height=100,width=100)
cap=cv2.VideoCapture("02.03.02_AHA_J15.mp4")
i=0
for i in timestps: 
    #Read only frame that are in timestps
    cap.set(cv2.CAP_PROP_POS_FRAMES,i)
    ret, frame = cap.read()
    if ret==True:
        video.append(frame[...,0:1].astype(np.uint8))
        print(i)        
    else:
        break
cap.release()

video=np.stack(video,0)
video=np.moveaxis(video,1,2)

def convert_points_to_image(video,points):
    """Convert point coordinates to images
    Points are represented as circle, with max radius of 5 pixels for maximum confidence (1)
    """
    size=points.max()
    for i in range(points.shape[0]):
        for label in range(points.shape[1]):
            frame,x,y,confidence=points[i,label,:]
            radius=int(10)
            if frame<video.shape[0]:
                video[frame,int(x-radius):int(x+radius),int(y-radius):int(y+radius)]=255
    return video

def convert_points_lines(video,points):
    """Plot points as lines in images
    """
    body_25b_edges=((1,2),(2,3),(3,4),(1,5),(5,6),(6,7))
    size=points.max()
    
    for i in range(points.shape[0]):
        for edge in body_25b_edges:
            frame,x1,y1,confidence1=points[i,edge[0],:]
            frame,x2,y2,confidence2=points[i,edge[1],:]
            if frame<video.shape[0]:
            #Trace a line between the two points
                video[frame]=trace_line_on_image(video[frame],x1,y1,x2,y2)
    return video

def trace_line_on_image(img,x1,y1,x2,y2,thickness=10):
    """Trace a line between two points on an image with a specific thickness
    """
    #Get the line equation
    if x1>x2:
        x1,x2=x2,x1
        y1,y2=y2,y1
    a=(y2-y1)/(x2-x1)
    b=y1-a*x1
    #Get the line equation
    for x in range(int(x1),int(x2)):
        y=a*x+b
        img[x,int(y-thickness):int(y+thickness)]=255

    return img

def interpolate_points_to_video(video,points):
    """Interpolate points to number of samples in video
    """
    new_size=video.shape[0]
    stop=[i for i,x in enumerate(list(points[:,0,0])) if x>new_size][0]
    coordinates=points[:stop,:,0:1]
    values=points[:stop,:,1:3]
    interp_points=[]
    for lab in range(points.shape[1]):
        print((np.arange(0,frames)).shape)
        interp_points.append(np.concatenate((np.arange(0,frames)[...,None],griddata(coordinates[:,lab],values[:,lab],(np.arange(0,frames)),method='linear',fill_value=0)),-1))
    interp_points=np.stack(interp_points,1)
    return interp_points

points=interpolate_points_to_video(video,points)

for i in range(video.shape[0]):
    nose=points[i,0,1:3]
    nose_x=nose[0]+50
    nose_y=nose[1]-50
    lenght=100
    #Draw a filter on the nose
    patch=video[i,int(nose_x-lenght):int(nose_x+lenght),int(nose_y-lenght):int(nose_y+lenght)]

    blurred_patch=gaussian(patch,sigma=20.01,channel_axis=-1)*255
    video[i,int(nose_x-lenght):int(nose_x+lenght),int(nose_y-lenght):int(nose_y+lenght)]=blurred_patch
    
edges=((1,2),(2,3),(3,4),(1,5),(5,6),(6,7))
print(points[150])
viewer = napari.view_image(video,rotate=-90)
colors=['red','green','blue','yellow','orange','purple','cyan','magenta','pink','brown','black','white']
for i,line in enumerate(edges):
    lab1,lab2=line

    shape=np.stack((points[:,lab1],points[:,lab2]-points[:,lab1]),axis=1)
    viewer.add_vectors(shape,edge_width=15.,edge_color=colors[i],rotate=-90)
napari.run()
