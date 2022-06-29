import time
import os
import numpy as np
import numpy.linalg as la
import pickle
from numbers import Number
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy import integrate
from PIL import Image
import pandas as pd
from threading import Thread
import threading
import multiprocessing
from time import sleep
#Создали датафрейм и записали его в файл
#-----------------------------------------------------------------------------------------------------------------------
def read_tif(filepattern , imrange=(15, 487), outfn = None):
    n = 500*500*(imrange[-1]-imrange[0])
    x = np.zeros(n, dtype='int16')
    y = np.zeros(n, dtype='int16')
    z = np.zeros(n, dtype='int16')
    intensity = np.zeros(n, dtype='float32')
    row = 0
    for iz in range(*imrange):
        path = filepattern.replace('*', '%04d' % iz)
        print('Reading', path)
        if n % 3 ==0:
            print(n)
        image_tiff = Image.open(path)
        # image_tiff.show() # opens the tiff image. this rainbow color tiff
        imarray = np.array(image_tiff)
        # print((imarray < 0).any())
        xv, yv = np.meshgrid(np.arange(500), np.arange(500), indexing='ij')
        x[row:row+250000], y[row:row+250000] = xv.flatten(), yv.flatten()
        z[row:row+250000] = iz
        intensity[row:row+250000] = imarray.flatten()
        row += 250000
#        for i in range(500):
#            for j in range(500):
#                x[row], y[row], z[row] = i, j, n
#                intensity[row] = imarray[i][j]
#                row += 1
        df = pd.DataFrame({'x': x, 'y': y, 'z': z, 'I': intensity})

    print(df)

    if outfn is not None:
        print(f'Saving data to {outfn}...', flush=True, end='')
        #df.to_csv(outfn)
        df.to_pickle(outfn)
        print('Done')

    return df


#df = read_tif(imrange=(15, 487), outfn='ct.csv')


def read_csv(filename):
    df = pd.read_csv(filename, dtype={'x': 'int16', 'y': 'int16', 'z': 'int16', 'I': 'float32'}, index_col=0)
    print(f'Read {df.shape[0]} rows from {filename}')
    print(df)
    return df

def plot_im(df, project='z', x=(0, 500), y=(0, 500), z=(16,483), what='I'):
    plt.clf()
    if project == 'z':
        if not (isinstance(x, tuple) and len(x) == 2 and
                isinstance(y, tuple) and len(y) == 2 and
                isinstance(z, Number)):
            print('Invalid coordinate ranges given z')
            return
        im = df.loc[(df['x'] >= x[0]) & (df['y'] >= y[0]) & (df['x'] < x[1]) & (df['y'] < y[1]) & (df['z'] == z), what].to_numpy()
        if im.size == 0:
            print('Nothing to plot')
            return
        im = np.reshape(im, (y[1]-y[0], x[1]-x[0]), order='F')
        print(im)
        plt.imshow(im, extent=[*x, *y], cmap='jet')#winter  jet
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
    elif project == 'x':
        if not (isinstance(x, Number) and
                isinstance(y, tuple) and len(y) == 2 and
                isinstance(z, tuple) and len(z) == 2):
            print('Invalid coordinate ranges given x')
            return

        im = df.loc[(df['x'] == x) & (df['y'] >= y[0]) & (df['z'] >= z[0]) & (df['y'] < y[1]) & (df['z'] <= z[1]), what].to_numpy()
        if im.size == 0:
            print('Nothing to plot')
            return
        im = np.reshape(im, (z[1]-z[0], y[1]-y[0]))
        print(im)
        plt.imshow(im, extent=[*y, *z], cmap='jet')
        plt.colorbar()
        plt.xlabel('y')
        plt.ylabel('z')
    elif project == 'y':
        # TODO: change the next code
        if not (isinstance(x, tuple) and len(x) == 2 and
                isinstance(y, Number) and
                isinstance(z, tuple) and len(z) == 2):
            print('Invalid coordinate ranges given y')
            return

        im = df.loc[(df['x'] >= x[0]) & (df['y'] == y) & (df['z'] >= z[0]) & (df['x'] < x[1]) & (df['z'] <= z[1]), what].to_numpy()
        if im.size == 0:
            print('Nothing to plot')
            return
        im = np.reshape(im, (z[1]-z[0], x[1]-x[0]))
        print(im)
        plt.imshow(im, extent=[*x, *z], cmap='jet')
        plt.colorbar()
        plt.xlabel('y')
        plt.ylabel('z')

    #plt.show()

def ctnotair(df):
    midl = df.loc[(df['x'] <= 100) & (df['y'] <= 100) & (df['z'] <= 100), 'I'].mean()
    print(midl)
    df1=df['I']- midl
   # df1.to_csv('ctnotair.csv')
    return df1

def save_image(df, path, project='Z', what='I'):#368
    if project=='Z':
        for z in range(338, 483):
            plot_im(df, 'z', x=(0, 500), y=(0, 500), z=z, what=what)
            plt.savefig(path + '\\Z\\Z_' + str(z) + '.png')
            plt.close()
            plt.show()
    elif project =='X':
        for i in range(150, 300):
            plot_im(df, 'x', x=i, y=(0, 500), z=(16, 483), what=what)
            plt.savefig(path + '\\X\\X_' + str(i) + '.png')
            plt.close()
            plt.show()
    elif project =='Y':
        for i in range(150, 300):
            plot_im(df, 'y', x=(0, 500), y=i, z=(16, 483), what=what)
            plt.savefig(path + '\\Y\\Y_' + str(i) + '.png')
            plt.close()
            plt.show()



def dest_density(df, boundaries, ithreshold=2000, mass=None, voxelsize=None, update=False):
    print('Calculating density...', flush=True)
    # if 'D' not in df.columns or  update and (mass is None or voxelsize is None):
    #     print('Needed parameters are not provided!')
    #     return None
    if not update or (mass is None or voxelsize is None):
        print('Needed parameters are not provided or update = False!')
        return None

    iair = df.loc[(df['x'] <= 100) & (df['y'] <= 100) & (df['z'] <= 100), 'I'].mean()

    C = boundaries
    inside = (C[0, 0] * df['x'] + C[0, 1] * df['y'] + C[0, 2] * df['z'] < C[0, 3]) & \
             (C[1, 0] * df['x'] + C[1, 1] * df['y'] + C[1, 2] * df['z'] < C[1, 3]) & \
             (C[2, 0] * df['x'] + C[2, 1] * df['y'] + C[2, 2] * df['z'] < C[2, 3]) & \
             (C[3, 0] * df['x'] + C[3, 1] * df['y'] + C[3, 2] * df['z'] < C[3, 3]) & \
             (C[4, 0] * df['x'] + C[4, 1] * df['y'] + C[4, 2] * df['z'] < C[4, 3]) & \
             (C[5, 0] * df['x'] + C[5, 1] * df['y'] + C[5, 2] * df['z'] < C[5, 3])
    selected = inside & (df['I'] > iair + ithreshold)

    allreco = all([what in df.columns for what in ('D', 'n', 'inside', 'selected')])

    if not allreco or update:
        alf400 = 0.438  # cm^3/g
        N = selected.sum()
        volume = N * (voxelsize ** 3)
        print(f"V = {volume:.1f} mm^3")
        rho = 1000 * mass / volume  # g/cm^3
        S = df.loc[selected, 'I'].mean() - iair
        k = rho / S
        print('k = ', k)
        df['D'] = (df['I'] - iair) * k
        df['n'] = np.sqrt(1.+alf400*df['D'])
        df['inside'] = inside
        df['selected'] = selected
        print(df)
    else:
        rho = df.loc[selected, 'D'].mean()

    print(f'rho = {rho:.4f} g/cm^3')

    return rho

def bild_gistogramm(df, what):
    df = df.hist(column=what,  bins=500)
    # plt.xlim(xmin=0.97)
    # plt.xlim(xmax=1.09)
    plt.xlim(xmin=0.98)
    plt.xlim(xmax=1.09)
    plt.ylim(ymax=0.65*10**6)
    #plt.yscale('log')
    #plt.xscale('log')
    plt.xlabel('n')
    plt.ylabel('количество вокселей')
    print('hist')
    print(df)
    plt.savefig('histogramm.png')
    plt.show()


def bild_plane(point):
    vector1 = []
    vector2 = []
    for i in range(3):
        print(i)
        vector1.append(point[1][i]-point[0][i])
        vector2.append(point[2][i]-point[1][i])
    norm_vector=[vector1[1]*vector2[2]-vector1[2]*vector2[1],
                 vector1[2] * vector2[0] - vector1[0] * vector2[2],
                 vector1[0] * vector2[1] - vector1[1] * vector2[0]
                 ]
    arrayABCD=[norm_vector[0],norm_vector[1],norm_vector[2],
               norm_vector[0]*point[0][0]+norm_vector[1]*point[0][1]+norm_vector[2]*point[0][2]]
    a=math.gcd(math.gcd(arrayABCD[0], arrayABCD[1]),math.gcd(arrayABCD[2], -arrayABCD[3]))
    arrayABCD = [i / a for i in arrayABCD]
    #print(vector1, vector2, norm_vector, arrayABCD)
    return arrayABCD


def var_boundaries(df, boundaries, hvard, side=0, ithreshold=2000):

    assert(side < 6 and side >= 0)

    nboundaries = boundaries.copy()
    for s in range(6):
        nboundaries[s, :] = boundaries[s, :] / la.norm(boundaries[s, :3])

    nboundaries1 = nboundaries.copy()
    nboundaries2 = nboundaries.copy()
    nboundaries1[side, 3] -= hvard
    nboundaries2[side, 3] += hvard

    rho0 = dest_density(df, nboundaries, ithreshold)
    rho1 = dest_density(df, nboundaries1, ithreshold)
    rho2 = dest_density(df, nboundaries2, ithreshold)

    opside = (side + 3) % 6

    r1 = nboundaries[side, 3] * nboundaries[side, :3]
    r2 = nboundaries[opside, 3] * nboundaries[opside, :3]
    a = abs(np.dot(nboundaries[side, :3], (r2-r1)))

    varrho = abs(rho2-rho1)/rho0
    vardist = 2*hvard/a
    print(f'Side {side}: varrho={varrho:.4f}, vardist={vardist:.4f}, varrho/vardist={varrho/vardist:.2f}')

    return varrho, varrho/vardist


def plot1d(df, point1, point2,bhcs, radius=5, step=2, voxelsize=1, what='D', addplot=True):
    print(f'Plotting 1D density distribution between {point1} and {point2} with radius {radius} and step {step}...')
    v = point2 - point1
    v = v / la.norm(v)
    voxels = df.loc[:, ('x', 'y', 'z')].to_numpy()
    cv = np.cross(voxels - point1, v)
    distance = la.norm(cv, axis=1)
    inside = (distance < radius) & df['inside']
    s = (np.dot(voxels[inside, :] - point1, v) // step) * step * voxelsize
    rho = df.loc[inside, what].groupby(s).mean()
    if not addplot:
        plt.clf()
    df1D = pd.DataFrame({'X': 1, 'N': rho})
    df1D.to_csv('1DdestribU.csv')
    plt.plot(rho.index, rho, '-', label=f'{point1}-{point2}, {bhcs}')
    plt.xlabel('X (mm)')
    plt.ylabel('показатель преломления')
    print(df1D)
    plt.legend()
    # plt.ylim((0, 1.1*rho.max()))
    if not addplot:
        plt.show()


def plot1dtomanydf(alldf,point,bhcs,d):
    for i in range(len(alldf)):
        plot1d(alldf[i], point[0], point[1],bhcs[i],radius=5, step=2, voxelsize=voxelsize, what='n', addplot=True)
    plt.savefig(d+"\\1D распределение\\"+d + rf"{point[0]}-{point[1]}" + '.png')
    plt.close()
    plt.show()
def plot1d_all_lauer(alldf,points,bhcs,d):
    for i in range(4):
        for j in range(4):
           # print(alldf[i])
            point = points[j]
            plot1d(alldf[i], point[0], point[1],bhcs[i],radius=5, step=2, voxelsize=voxelsize, what='n', addplot=True)
        plt.savefig(d+"\\1D распределение\\lauer_bhc\\" + d + rf"  bhc{bhcs[i]}" + '.png')
        plt.close()
        plt.show()

def test (df,C):
    inside = df.loc[((C[0, 0] * df['x'] + C[0, 1] * df['y'] + C[0, 2] * df['z'] < C[0, 3]) & \
             (C[1, 0] * df['x'] + C[1, 1] * df['y'] + C[1, 2] * df['z'] < C[1, 3]) & \
             (C[2, 0] * df['x'] + C[2, 1] * df['y'] + C[2, 2] * df['z'] < C[2, 3]) & \
             (C[3, 0] * df['x'] + C[3, 1] * df['y'] + C[3, 2] * df['z'] < C[3, 3]) &  ###верх
             (C[4, 0] * df['x'] + C[4, 1] * df['y'] + C[4, 2] * df['z'] < C[4, 3]) & #лево
             (C[5, 0] * df['x'] + C[5, 1] * df['y'] + C[5, 2] * df['z'] < C[5, 3])), 'I']=0
    plot_im(df, 'z', z=240)
    plt.show()



if __name__ == '__main__':
    timer = time.time()

    updateDataframe = False
    updateDensity = False

    ABCD= [103598., 11968., -1689., 93150602]
    Axyz=[[79,374,66],
          [370,412,66],
          [402,135,66],
          [124,98,66]]


    dirs = ['Aerogel 4-layer CT 030222 1', 'Aerogel 4-layer CT 030222 2',
            'Aerogel 4-layer CT Mo 030222', 'Aerogel 4-layer CT Mo 030222 3frames','Aerogel Zr 6% R (rho=0.2) Mo glass 3.2mm 5 frames',
            'Aerogel Zr 6% U (rho=0.115) Mo glass 3.2 mm 5 frames', 'Aerogel 4-layer Mo  glass 3.2mm 5 frames 130522']


    ### Parameters for "Aerogel 4-layer CT 030222 1"
    # mass = 4.77 # gram
    # d = dirs[0]
    # boundaries = np.array([[-418., 3201., 8., 1164680], [103598., 11968., -1689., 43150602], [0., 0., -1., -33.],
    #                        [629., -4726., - 8., -385680], [-1564., -255., 9., -218332], [0., 0., 1., 482.]])

#############################################################################
    ##Parameters for "Aerogel 4-layer CT Mo 030222"

    # mass = 4.77  # gram
    # d = dirs[2]
    # boundaries = np.array([[3800.,59400.,13.,24570320.], [27800.,-1900.,-913.,10163680.], [0., 0., -1., -33.],
    #                        [-3000.,-56200.,-1081.,-7584440.], [-18800.,200.,227.,-1800520.], [0., 0., 1., 482.]])

    ### Parameters for "Aerogel 4-layer CT Mo 030222 3frames"
    # mass = 4.77  # gram
    # d = dirs[3]
    # boundaries = np.array([[3400.,59000.,627.,24579480.], [56200.,-3000.,-901.,21090760.], [0., 0., -1., -33.],
    #                        [-3400.,-56200.,379.,-7400040.], [-5620.,20.,71.,-564900.], [0., 0., 1., 482.]])

    ### Parameters for "Aerogel Zr 6% R (rho=0.2) Mo glass 3.2mm 5 frames"
    # mass = 1.73  # gram
    # d = dirs[4]
    # boundaries = np.array([[0.,1.,0.,385.], [1.,0.,0.,383.], [0., 0., -1., -337.],
    #                        [0.,-1.,0.,-127.], [-1.,0.,0.,-126.], [0., 0., 1., 482.]])

    ### Parameters for "Aerogel Zr 6% U (rho=0.2) Mo glass 3.2mm 5 frames"
    mass = 0.79  # gram
    d = dirs[5]
    boundaries = np.array([[0., 1., 0., 385.], [1., 0., 0., 383.], [0., 0., -1., -337.],
                           [0., -1., 0., -127.], [-1., 0., 0., -126.], [0., 0., 1., 482.]])

    #####  Aerogel 4-layer Mo  glass 3.2mm 5 frames 130522
    # mass = 4.77  # gram
    # d = dirs[6]
    # boundaries = np.array([[0., 1., 0., 383.], [1., 0., 0., 431.], [0., 0., -1., -33.],
    #                        [0., -1., 0., -75.], [-1., 0., 0., -112.], [0., 0., 1., 482.]])

    voxelsize = 0.08231293 # size of voxel [mm]
    ithreshold = 2000
    hvard = 5
    ###
   # df = pd.read_pickle('Aerogel_4 - layer_CT_030222_1_bhc0.pkl')


    bhcs = ['0', '0.05', '0.1', '0.15']


    alldf= [None]*len(bhcs)
    print(alldf)
    i = 0
    for bhc in bhcs:

        filepattern = rf"D:\Diplom\{d}\raw\reconstruction bhc{bhc}\recon*.tif"
        outfn = d.replace(' ', '_') + f'_bhc{bhc}.pkl'

        if not os.access(outfn, os.R_OK) or updateDataframe:
            df = read_tif(filepattern, imrange=(16, 483), outfn=outfn)

        else:
            print(f'Reading {outfn}...', flush=True, end='')
            df = pd.read_pickle(outfn)
            print('Done')
            print(df)
            alldf[i] = df
            i += 1

        allreco = all([what in df.columns for what in ('D', 'n', 'inside', 'selected')])

        if not allreco or updateDensity:
            dest_density(df, boundaries, ithreshold, mass, voxelsize, update=updateDensity)
            print(f'Saving {outfn}...', flush=True, end='')
            df.to_pickle(outfn)
            print('Done')
        else:
            print('Density data are already in the dataframe')

    bild_gistogramm(alldf[2],'n')
        ##### РАспределение  в каждом слое
    dp1 = np.array([[0, 300, 90], [500, 250, 90]])
    dp2 = np.array([[0, 300, 170], [500, 250, 170]])
    dp3 = np.array([[0, 300, 280], [500, 250, 280]])
    dp4 = np.array([[0, 300, 380], [500, 250, 380]])
    point = [dp1, dp2, dp3, dp4]

    dz1 = np.array([[250, 250, 20], [250, 250, 383]])
    # plot1dtomanydf(alldf, dz1, bhcs, d)
    # plot1d_all_lauer(alldf, point, bhcs, d)
    # plot1dtomanydf(alldf, dp1, bhcs, d)
    # plot1dtomanydf(alldf, dp2, bhcs, d)
    # plot1dtomanydf(alldf, dp3, bhcs, d)
    # plot1dtomanydf(alldf, dp4, bhcs, d)
    #plot1d(alldf[2],dz1[0],dz1[1],bhcs[2], what = 'n', voxelsize=voxelsize, step=3) # распределения по оси Z
   # plot1dtomanydf(alldf[2], dz1, bhcs, d)

    #test(df,boundaries)
    print('------------')
    #plot_im(df, 'y', y=220, what='D')
    #plot_im(df, 'z', z=400)
    #
    plt.show()

    #save_image(alldf[0],d,'Z','D')
    # pool= multiprocessing.Pool(processes=3)
    # p1 = multiprocessing.Process(target=save_image, args=(alldf[0], d, 'X', 'D'))
    # p2 = multiprocessing.Process(target=save_image, args=(alldf[0], d, 'Y', 'D'))
    # p3 = multiprocessing.Process(target=save_image, args=(alldf[0], d, 'Z', 'D'))
    # p1.start()
    # p2.start()
    # p3.start()
    #
    # p1.join()
    # p2.join()
    # p3.join()

    print(f'Processing time: {time.time() - timer:.1f} seconds')