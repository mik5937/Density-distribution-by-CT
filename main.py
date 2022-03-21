# #pandas
# back propogetion
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
    print('11232')
    if project=='Z':
        for z in range(150, 300):
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

    if 'D' not in df.columns or update and (mass is None or voxelsize is None):
        print('Needed parameters are not provided!')
        return None

    iair = df.loc[(df['x'] <= 100) & (df['y'] <= 100) & (df['z'] <= 100), 'I'].mean()

    C = boundaries
    selvoxels = (C[0, 0] * df['x'] + C[0, 1] * df['y'] + C[0, 2] * df['z'] < C[0, 3]) & \
                (C[1, 0] * df['x'] + C[1, 1] * df['y'] + C[1, 2] * df['z'] < C[1, 3]) & \
                (C[2, 0] * df['x'] + C[2, 1] * df['y'] + C[2, 2] * df['z'] < C[2, 3]) & \
                (C[3, 0] * df['x'] + C[3, 1] * df['y'] + C[3, 2] * df['z'] < C[3, 3]) & \
                (C[4, 0] * df['x'] + C[4, 1] * df['y'] + C[4, 2] * df['z'] < C[4, 3]) & \
                (C[5, 0] * df['x'] + C[5, 1] * df['y'] + C[5, 2] * df['z'] < C[5, 3]) & \
                (df['I'] > iair + ithreshold)

    if 'D' not in df.columns or update:
        N = selvoxels.sum()
        volume = N * (voxelsize ** 3)
        print(f"V = {volume:.1f} mm^3")
        rho = 1000 * mass / volume  # g/cm^3
        S = df.loc[selvoxels, 'I'].mean() - iair
        k = rho / S
        print('k = ', k)
        df['D'] = (df['I'] - iair) * k
    else:
        rho = df.loc[selvoxels, 'D'].mean()

    print(f'rho = {rho:.4f} g/cm^3')

    return rho

def bild_gistogramm(df):
    df = df.hist(column='I',  bins=200)
    print('hist')
    print(df)
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



if __name__ == '__main__':
    timer = time.time()

    updateDataframe = False
    updateDensity = False

    #point= [[79,374,66],[370,412,66],[224,392,440]]
    #bild_plane(point)
    ABCD= [103598., 11968., -1689., 93150602]
    Axyz=[[79,374,66],
          [370,412,66],
          [402,135,66],
          [124,98,66]]


    dirs = ['Aerogel 4-layer CT 030222 1', 'Aerogel 4-layer CT 030222 2',
            'Aerogel 4-layer CT Mo 030222', 'Aerogel 4-layer CT Mo 030222 3frames']
    d = dirs[0]

    ### Parameters for "Aerogel 4-layer CT 030222 1"
    mass = 4.77 # gram

    boundaries = np.array([[-418., 3201., 8., 1164680], [103598., 11968., -1689., 43150602], [0., 0., -1., -33.],
                           [629., -4726., - 8., -385680], [-1564., -255., 9., -218332], [0., 0., 1., 482.]])
    # boundaries = np.array([[629., -4726., - 8., -385680],
    #                        [103598., 11968., -1689., 43150602],
    #                        [-418., 3201., 8., 1164680],
    #                        [-1564., -255., 9., -218332],
    #                        [0., 0., -1., -33.], [0., 0., 1., 482.]])
    # boundaries = np.array([[-418., 3201., 8., 1164680], [103598., 11968., -1689., 93150602],
    #                        [629., -4726., - 8., -385680], [-1564., -255., 9., -218332],
    #                        [0., 0., -1., -33.], [0., 0., 1., 482.]])


    voxelsize = 0.08231293 # size of voxel [mm]
    ithreshold = 2000
    hvard = 5
    ###
   # df = pd.read_pickle('Aerogel_4 - layer_CT_030222_1_bhc0.pkl')
   # print(df)

    bhcs = ['0', '0.05', '0.1', '0.15']

    for bhc in bhcs[:1]:
        filepattern = rf"D:\Diplom\{d}\raw\reconstruction bhc{bhc}\recon*.tif"
        outfn = d.replace(' ', '_') + f'_bhc{bhc}.pkl'

        if not os.access(outfn, os.R_OK) or updateDataframe:
            df = read_tif(filepattern, imrange=(16, 483), outfn=outfn)

        else:
            print(f'Reading {outfn}...', flush=True, end='')
            df = pd.read_pickle(outfn)
            print('Done')

        if 'D' not in df.columns or updateDensity:
            dest_density(df, boundaries, ithreshold, mass, voxelsize, update=updateDensity)
            print(f'Saving {outfn}...', flush=True, end='')
            df.to_pickle(outfn)
            print('Done')
        else:
            print('Density data are already in the dataframe')

        for side in range(6):
            varrho, varrho_over_vardist = var_boundaries(df, boundaries, hvard, side, ithreshold)

        #print(df)

    print('------------')
   #  plot_im(df, 'z', z=240)
   # # plot_im(df, 'y', y=200, what='D')
   #  plt.show()


  #  obj = df.loc[ (-10704 * df['x'] + 96302 * df['y'] + 1779 * df['z'] > 8486562)]
   # bildgistogramm(df)
    #df = read_csv('ct.csv')
    #df = read_tif(imrange=(15, 487), outfn='ct.csv')
    #df= dest_density(df,m)

    # pool= multiprocessing.Pool(processes=3)
    # p1 = multiprocessing.Process(target=save_image, args=(df, d, 'X', 'D'))
    # p2 = multiprocessing.Process(target=save_image, args=(df, d, 'Y', 'D'))
    # p3 = multiprocessing.Process(target=save_image, args=(df, d, 'Z', 'D'))
    # p1.start()
    # p2.start()
    # p3.start()
    #
    # p1.join()
    # p2.join()
    # p3.join()

    print(f'Processing time: {time.time() - timer:.1f} seconds')
# df = read_csv('3D.csv')
# df = df.loc[df['I'] > 1.28]
# df = df.loc[df['I'] < 1.30]
# df = df.reset_index(drop=True)
# print(df)
# for i in range(0, 45000):
#     if i % 3 == 0:
#         df.drop(axis=0, index=i, inplace=True)
#
# # df.drop(axis=0,index=(df['I'] > 6500), inplace=True)
# print(df)
# # df.to_csv('3D.csv')
# x = df['x']
# y = df['y']
# z = df['z']
# s = df['I']
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z, s=s)
# plt.show()



# df = read_tif(imrange=(15, 150), outfn='ct.csv')
# midl = df.loc[(df['x'] <= 100) & (df['y'] <= 100) & (df['z'] <= 100), 'I'].mean()
# df['I'] = df['I'] - midl
# rslt_df = []
# rslt_df = df[df['I'] > 6500]
# h = 0.07112526539
# N=len(rslt_df)
# V = N * (h ** 3)
# print(V)
# pho = m / V
# print(pho * 1000, 'g/sm^3')
# S = df.loc[df['I'] > 6500, 'I'].mean()
# k = 1000 * pho / (S)
# print('k=', k)
# k=4.7097355512420514e-05
# # df['I'] = df['I'] * k





# # df = read_csv('density_in_point4.csv')
# # print(df)
# # for x in range(150,200):
# #     plot_im(df, 'z', x=(0,500), y=(0,500), z=x)
#
# #df = read_csv('ct.csv')
# df = read_csv('density_in_point3.csv')
# df1= read_csv('ctnotair.csv')
# #midl = df.loc[(df['x'] <= 100) & (df['y'] <= 100) & (df['z'] <= 100), 'I'].mean()
# #print(df1)
# rslt_df = []
# rslt_df = df1[df1['I'] > 6500]
# #S= df1[df1['I'] > 6500].sum()
#
# print(len(rslt_df))
# #print(midl)
# h=0.07112526539
# #N=len(rslt_df)
# N=5025465
# V=N*(h**3)
# print(V)
# m=2.24
# pho= m/V
# print(pho*1000, 'g/sm^3')
#
# #rslt_df = df[df['I'] > 6500+midl]
# S2= df1.loc[df1['I'] > 6500, 'I'].mean()
# #S= df.loc[df['I'] > 6500+midl, 'I'].mean()
# #print('Sср=',S)
#
# k=1000*pho/(S2)
# print('k=',k)
# df['I']=df1['I']*k
# print(df1)
# print(df)



#df1.to_csv('density_in_point4.csv')


# for x in range(150,200):
#     plot_im(df, 'x', x=x, y=(0,500), z=(150, 300))

#df = read_csv('ct.csv')
#print(df)




# df['№'] = int
# df['X'] = X
# df['Y'] = Y
#
# #print(df.iloc[1])
# df.to_csv('Intensity.csv')
#-----------------------------------------------------------------------------------------------

# Найдем интенсивность воздуха из объема 100х100х100
#-------------------------------------------------------------------------------------------
# df = pd.read_csv('Intensity.csv')
# pho_mid=0
# index=0
# for z in range(100):
#     if z!=0:
#         index+=200500
#
#     for x in range(100):
#         if x != 0:
#             index+=400
#         for y in range(100):
#             if y != 0:
#                 index += 1
#             #print(index)
#             pho_mid += df.iloc[index]['№']
# print(pho_mid/(100*100*100))
#---------------------------------------------------12161.213 - средняя интенсивность в воздухе

#Ищем пиксили с высокой плотностью
#-----------------------------------------------------------------------------

# df = pd.read_csv('Intensity1.csv')
# df1 = pd.DataFrame()
# #df["№-air"]=None
# #print(df.iloc[1])
# pho=[]
# for i in range(117999999+1):
#     pho.append(df.iloc[i]['№']-12198.10957)
#     if i % 1000000==0:
#         print(i)
# #print(pho)
# df1['№-air'] = pho
# df1.to_csv('Intensity2.csv')

#------------------------------------------------------------------------------------

# 0,07112526539- размер 1 стороны вокселя

# найдем сколько точек имеют интенсивность более 6500 (№-air > 6500)
#-------------------------------------------------------------------------------------------
#df = pd.read_csv('Intensity2.csv')
# rslt_df = []
# rslt_df = df[df['№-air'] > 6500]
# print(len(rslt_df))
#-------------------------------------------------------------------------------------------
#1484294- количество точек оъекта


#Посчитаем обьем фигуры V= 534.0619431396051 mm^3
#-------------------------------------------------------------------------------------------
# h=0.07112526539
# N=len(rslt_df)
# V=N*(h**3)
# print(V)
# m=2.24
# pho= m/V
# print(pho*1000, 'g/sm^3')
# pho =4.194270025742049 g/sm^3
#-------------------------------------------------------------------------------------------



# Средняя интенсивность обьекта
#-------------------------------------------------------------------------------------------
# S=0
# for i in range(117999999+1):
#     if df.iloc[i]['№-air'] >6500:
#         S+=(df.iloc[i]['№-air'])
#     if i % 1000000==0:
#         print(i)
# print(S)
# print(S/1484294) # 13676.819949033594- средняя интенсивность обьекта
#-------------------------------------------------------------------------------------------
# df = pd.read_csv('Intensity2.csv')
# df1 = pd.DataFrame()
# dens=[]
# K=4.194270025742049/13676.819949033594
# for i in range(117999999+1):
#
#     dens.append(df.iloc[i]['№-air']*K)
#     if i % 1000000==0:
#         print(i)
# df1['Density'] = dens
# df1.to_csv('density.csv')
#print(df1)



#print(df)
# Intensiviti=25000
# k=0                 [13069 11934 13331 ... 13262 12978 10457]
#  [ 9654 13583 10851
# for n in range(500):
#     for m in range(500):
#         if array[1][m][n] >k:
#             k= array[1][m][n]
#         if array[1][m][n]<Intensiviti:
#             array[1][m][n]=0
# for n in range(500):
#     for m in range(500):
#         if imarray[m][n] >k:
#             k= imarray[m][n]
#         if imarray[m][n]<Intensiviti:
#             imarray[m][n]=0
#         #imarray[m][n]=20000
# print(imarray)
# print(k)
# pillow_image=Image.fromarray(array[1])
# pillow_image.show()
# pillow_image=Image.fromarray(array[0])
# pillow_image.show()
# pillow_image=Image.fromarray(array[2])
# pillow_image.show()
#
#
# def read(fileName):
#     f = open(fileName, "r")
#     return [list(map(int, i.split())) for i in f.readlines()]
#
#
# def write(array, fileName):
#     f = open(fileName, "w")
#     for i in array:
#         print(" ".join(map(str, i)), file=f)
#
#
# fileName = "input.txt"
# a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# write(imarray, fileName)
# #b = read(fileName)
# b = np.array([[1.5, 2, 3], [4, 5, 6]])
# print(b[1][2])
#
#
#
