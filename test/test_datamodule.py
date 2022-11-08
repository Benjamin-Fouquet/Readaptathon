import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datamodules import HackathonDataModule
import torch
import psutil
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

if __name__ == "__main__":
    dm=HackathonDataModule('test/datapath_test',list(range(1,8)),batch_size=1)
    print("Memory Usage:")
    print('Total:    ', round(psutil.virtual_memory().total/1024**3,1), 'GB')
    print('Available before loading :', round(psutil.virtual_memory().available/1024**3,1), 'GB')
    dm.prepare_data()
    dm.setup()
    print('Available after loading :', round(psutil.virtual_memory().available/1024**3,1), 'GB')


    count=0
    #Iterate over all batches
    for i,batch in enumerate(dm.train_dataloader()):
        if i==0:
            print(f'Batch {i} : with {len(batch)} variable(s)')

            if len(batch)==1:
                print(f'\t Variable shape : {batch.shape}')
                data=batch
            else:
                for j,v in enumerate(batch):
                    print(f'\t Variable {j} : {v.shape}')
                data=batch[0]
        count+=1

    print(f'Number of batches : {count}')
    
    #Recover points from data. from (batch_size,num_points*space_dim,N) to (batch_size,num_points,space_dim,N)
    data=data.reshape(data.shape[0],data.shape[1]//3,3,data.shape[2])[:,:,:2,:]
    print(f'Points shape : {data.shape}')
    #Plot points as an animation over the dimension 3 of the skeleton
    body_25b_edges=((0,1),(1,2),(2,3),(0,4),(4,5),(5,6))
    point_names=['Cou','EpauleD','CoudeD','PoignetD','EpauleG','CoudeG','PoignetG']
    fig, ax = plt.subplots()
    ax.set_xlim(-1,0)
    ax.set_ylim(-1,0)
    ax.set_aspect('equal')
    ax.set_title('Skeleton')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid()
    lines = [ax.plot([], [], 'o-', lw=2, label=point_names[i])[0] for i in range(data.shape[1])]
    ax.legend()
    def init():
        for line in lines:
            line.set_data([], [])
        return lines
    def update(i):
        for j,line in enumerate(lines):
            line.set_data(-data[0,j,0,i], -data[0,j,1,i])
        return lines
    
    anim = FuncAnimation(fig, update, init_func=init, frames=data.shape[3], interval=100, blit=True)
    plt.show()


















