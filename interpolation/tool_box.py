import json
import numpy as np
import os
from scipy.interpolate import griddata

def getT(test_name):
    with open(test_name) as j:
        T = json.load(j)
    return T


def getPosesOld(T):
    """
    Extraction et formatage des poses à partir des json pretraites par les eleves
    """

    P = np.array(T["poses"])
    P = np.array([[xy[:2] for xy in frame] for frame in P])

    # position moyenne de la nuque

    offx, offy = np.mean(P[:, 0], axis=0)

    # normalisation

    for i in range(len(P)):
        for j in range(len(P[i])):
            P[i][j][0] -= offx
            P[i][j][1] -= offy

    # concaténation par pack de frames

    return P

def getPosesRaw(T):

    P = np.array(T["poses"])
    P = np.array([[xy[:2] for xy in frame] for frame in P])

    return P



#<--- Code de Nathan ci-dessous --->
'''
def getPoses(keypoints):
    """
    Extraction et formatage des poses à partir des fichiers OpenPose
    Format des donnees : (N,C,4)   =>   N, label de point, (timestamp, x, y, confidence)
    """
    points=[]
    files=os.listdir(keypoints)
    files.sort()
    for f in files:
        timestp=int(f.split('_')[1])
        with open(os.path.join(keypoints,f),'r') as f:
            data=json.load(f)
            point=np.array(data['people'][0]['pose_keypoints_2d']).reshape(-1,3)
            point=np.concatenate((np.tile(np.array([timestp]),(point.shape[0],1)),point),axis=1)
            points.append(point)
    points=np.stack(points).astype('int')
    return points



def interpolate_points_to_video(points):
    """Interpolate points to number of video frames
    """
    last_frame=points[-1,0,0]
    coordinates=points[...,0:1]
    values=points[...,1:]
    interp_points=[]
    for lab in range(points.shape[1]):
        interp_points.append(np.concatenate((np.arange(0,last_frame)[...,None],griddata(coordinates[:,lab],values[:,lab],(np.arange(0,last_frame)),method='linear',fill_value=0)),-1))
    interp_points=np.stack(interp_points,1)
    return interp_points
'''

# --- modification des fonctions de Nathan par Sarah --- 

# Ici j'ai ajouté une condition si la données est vide (parce-que j'ai des données de pose vides...)
def getPoses(folder: str):
    """
    Extraction et formatage des poses à partir des fichiers OpenPose
    Format des donnees : (N,C,4)   =>   N, label de point, (timestamp, x, y, confidence)
    """
    points = []
    files = os.listdir(folder)
    files.sort()
    for f in files:
        timestp = int(f.split("_")[1])
        with open(os.path.join(folder, f), "r") as f:
            data = json.load(f)

            if (
                len(data["people"]) > 0
            ):  # add something to inform that no data available?
                point = np.array(data["people"][0]["pose_keypoints_2d"]).reshape(-1, 3)
                point = np.concatenate(
                    (np.tile(np.array([timestp]), (point.shape[0], 1)), point), axis=1
                )
                points.append(point)
    points = np.stack(points).astype("int")
    return points

# --- fin des modifications de Sarah ---

def getPosesBM(folder: str,keypoints=['Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist']):
    """
    Extraction et formatage des poses des fichiers OpenPose du dataset BimanualActions
    Format des donnees : (N,C,4)   =>   N, label de point, (timestamp, x, y, confidence)
    """
    points = []
    files = os.listdir(folder)
    files.sort()
    for f in files:
        timestp = int(f.split("_")[1].replace('.json',''))
        with open(os.path.join(folder, f), "r") as f:
            data = json.load(f)[0]
            point=[]
            for k in keypoints:
                x,y,c=[data[k][i] for i in ['x','y','confidence']]
                point.append(np.stack([x,y,c]))
            point=np.concatenate(point)
            points.append(point)
    points = np.stack(points,-1).astype("int")
    return points





# Ici j'ai ajouté un paramètre last_frame pour pouvoir interpoler à un nombre de frame donné 
# (pour pouvoir mettre toutes les vidéos au même nombre de frames)
def interpolate_points_to_video(points, last_frame=None):
    """Interpolate points to number of video frames"""
    if last_frame is None:
        last_frame = points[-1, 0, 0]

    coordinates = points[..., 0:1]
    values = points[..., 1:]
    interp_points = []
    for lab in range(points.shape[1]):
        interp_points.append(
            np.concatenate(
                (
                    np.arange(0, last_frame)[..., None],
                    griddata(
                        coordinates[:, lab],
                        values[:, lab],
                        (np.arange(0, last_frame)),
                        method="linear",
                        fill_value=0,
                    ),
                ),
                -1,
            )
        )
    interp_points = np.stack(interp_points, 1)
    return interp_points
