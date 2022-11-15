import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datamodules import HackathonDataModule
import torch
import psutil
import holoviews as hv
import panel as pn
import pandas as pd
import numpy as np
import datashader as ds
from holoviews.operation.datashader import datashade
import datashader as ds
import datashader.transfer_functions as tf
from datashader.bundling import connect_edges, hammer_bundle


if __name__ == "__main__":
    dm=HackathonDataModule('test/datapath_test','test/scores.json',list(range(1,8)),batch_size=1)
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
    data=dm.pretrain_ds[0][0]
    
    #Recover points from data. from (batch_size,num_points*space_dim,N) to (batch_size,num_points,space_dim,N)
    data=data.reshape(data.shape[0],data.shape[1]//3,3,data.shape[2])[:,:,:2,:]
    print(f'Points shape : {data.shape}')
    #Plot points over the dimension 3 of the skeleton
    body_25b_edges=((0,1),(1,2),(2,3),(0,4),(4,5),(5,6))
    point_names=['Cou','EpauleD','CoudeD','PoignetD','EpauleG','CoudeG','PoignetG']
    point_ids=[0,1,2,3,4,5,6]
    #Convert points as a dataframe
    df=pd.DataFrame(-data[0].moveaxis(-1,0).reshape(-1,2),columns=['x','y'])
    print(df)
    df['point']=point_names*data.shape[-1]
    df['point_id']=point_ids*data.shape[-1]
    df['frame']=np.array(range(data.shape[-1]))[:,None].repeat(len(point_names),axis=1).reshape(-1)
    print(df)


    hv.extension('bokeh')
    slider=pn.widgets.IntSlider(name='Frame',start=0,end=data.shape[-1]-1,value=0)
    #Same as above but with datashader
    def nodesplot(nodes, name=None, canvas=None, cat=None,cvsopts={}):
        canvas = ds.Canvas(**cvsopts) if canvas is None else canvas
        aggregator=None if cat is None else ds.count_cat(cat)
        agg=canvas.points(nodes,'x','y',aggregator)
        return tf.spread(tf.shade(agg, cmap=["#FF3333"]), px=3, name=name)

    def edgesplot(edges, name=None, canvas=None,cvsopts={}):
        canvas = ds.Canvas(**cvsopts) if canvas is None else canvas
        return tf.shade(canvas.line(edges, 'x','y', agg=ds.count()), name=name)
    
    def graphplot(nodes, edges, name="", canvas=None, cat=None,cvsopts={}):
        if canvas is None:
            xr = nodes.x.min(), nodes.x.max()
            yr = nodes.y.min(), nodes.y.max()
            canvas = ds.Canvas(x_range=xr, y_range=yr, **cvsopts)
            
        np = nodesplot(nodes, name + " nodes", canvas, cat)
        ep = edgesplot(edges, name + " edges", canvas)
        return tf.stack(ep, np, how="over", name=name)

    @pn.depends(slider.param.value)
    def plot_points(frame=0):
        df_frame=df[df['frame']==frame]
        nodes=df_frame[['x','y','point_id']]
        nodes['name']=nodes['point_id']
        nodes['point_id']=nodes['point_id'].astype('category')
        nodes.set_index('name',inplace=True)
        #Convert body_25b_edges to a dataframe with x and y columns as source and target
        edges=pd.DataFrame(body_25b_edges,columns=['source','target'])
        print(nodes)
        print(edges)
        #Set type str to source and target
        cvs=ds.Canvas(plot_width=400,plot_height=400,x_range=(-1.2,0.2),y_range=(-1.2,0.2))
        graph=graphplot(nodes,connect_edges(nodes,edges),cat='point_id',canvas=cvs,cvsopts={'plot_height':400,'plot_width':400})
        return graph
        
        
    accelerations=[]
    for point in range(data.shape[1]):
        acceleration2d=np.linalg.norm(data[0,point,:,1:]-data[0,point,:,0:-1],axis=0)
        accelerations.append(acceleration2d)
        #Create line plot, with the acceleration as y and the frame as x
    def update_slider(x, y):
        slider.value=int(x)
        print(x)
        #return empty RGBPlot
    shaders=[]
    ndoverlay=hv.NdOverlay()
    opts= hv.opts.RGB(width=1000,height=300,xlim=(0,data.shape[-1]-1),xlabel='Frame',ylabel='Acceleration',title=point_names[i])
    #set color blue
    
    # ndoverlay=hv.NdOverlay({point_names[i]:hv.Curve((range(data.shape[-1]),acceleration2d))},kdims='Point')
    #Create an ndoverlay with all the curves
    def curve_acceleration(point):
        return hv.Curve((range(data.shape[-1]),accelerations[point]),label=point_names[point])

    curve_dict={point_names[i]:curve_acceleration(i) for i in range(data.shape[1])}

    ndoverlay=hv.NdOverlay(curve_dict,kdims='Point')
    #Add datashade, with different colors for each point and legend outside
    shaded=datashade(ndoverlay,aggregator=ds.count_cat('Point'),cmap='Category10',legend_position='right').opts(opts)
    stream= hv.streams.Tap(source=shaded,x=0, y=np.nan)
    stream.add_subscriber(update_slider)
    # shaders.append(shaded)
    

    @pn.depends(slider.param.value)
    def plot_acceleration(frame):
        lines=[]
        # for i,shaded in enumerate(shaders):
        #     shaded=hv.VLine(frame).opts(color='red',line_width=2)*shaded
        #     lines.append(shaded)
        
        return shaded*hv.VLine(frame).opts(color='red',line_width=2)
    #Add a legend that shows point_names with corresponding color (Category10)
    legend=pd.DataFrame({'Point':point_names})
    legend['Color']=pd.Series(['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'])
    legend.set_index('Point',inplace=True)
    
    #Convert legend as panel widget
    legend_widget=pn.widgets.DataFrame(legend)
    #Set cell background colours
    
    
    row=pn.Column(slider,plot_points)
    app=pn.Column(row,plot_acceleration)
    app=pn.Row(app,legend_widget).servable()
    app.show()



















