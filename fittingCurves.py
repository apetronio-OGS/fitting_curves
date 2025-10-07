import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import scipy as sy
from scipy.interpolate import interp1d


def find_coordinate(data_dict, target_value, known="DRm"):
    """
    Trova la coordinata corrispondente a un valore su una retta o spezzata.

    Parameters:
    - data_dict: dizionario con chiavi 'D' e 'DRm', ciascuna associata a una lista di valori
    - target_value: valore noto (D o DRm)
    - known: 'D' se conosci D e vuoi DRm, 'DRm' se conosci DRm e vuoi D

    Returns:
    - valore interpolato corrispondente
    """
    x = data_dict['D']
    y = data_dict['DRm']
    x = np.array(x)
    y = np.array(y)
    if known == 'D':
        interp_fun = interp1d(x, y, kind='linear', fill_value="extrapolate")
        return float(interp_fun(target_value))
    elif known == 'DRm':
        interp_fun = interp1d(y, x, kind='linear', fill_value="extrapolate")
        return float(interp_fun(target_value))
    else:
        raise ValueError("Parametro 'known' deve essere 'D' o 'DRm'") 


print("I'm fitting curves!")

def func(x, a, b, c, d):
    return a * np.exp(-b * x**2  ) + c*x +d 

def func2(x, a, b, c):
    return a / (1 + np.exp(b *(x-c) ))

def func3(x, a, b, c,d):
    return (np.exp(a*(x))-np.exp(-b*(x)))/(np.exp(c*(x))+np.exp(- d*(x)))

def func4(x, a, b, c):
    return a*x*(b+sy.special.erf(c*x))

comp = pd.read_csv("Competitor_table.csv",sep=';',header=None)
print("competitor table",comp)

pa1 = pd.read_csv("jumbo10_H0_data",sep='\t',header=None)
pa2 = pd.read_csv("jumbo16_H0_data",sep='\t',header=None)
pa3 = pd.read_csv("jumbo22_H0_data",sep='\t',header=None)
p1=pa1.to_numpy()
p2=pa2.to_numpy()
p3=pa3.to_numpy()
comp=comp.to_numpy()
print(comp[0:8,1])

print(pa2.shape)
d1=p1[:,0]
d2=p2[:,0]
d3=p3[:,0]

ydata1 = p1[:,2]
ydata2 = p2[:,2]
ydata3 = p3[:,2]

print("data for fitting",d1 ,ydata1)
popt1, pcov1 = curve_fit(func, d1, p1[:,8])
print(popt1)
popt2, pcov2 = curve_fit(func, d2, p2[:,10])
print(popt2)
popt3, pcov3 = curve_fit(func, d3, p3[:,14])
print(popt3)

dd=np.linspace(start=0, stop=d1[-1], num=100, endpoint=False, retstep=False)
plt.figure(1)
plt.plot(dd, func(dd, *popt1), 'b-')#, label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt1))
plt.plot(dd, func(dd, *popt2), 'r-')#, label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt2))
plt.plot(dd, func(dd, *popt3), 'g-')#, label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt3))

 
plt.plot(comp[0:8,1],comp[0:8,0]+5, '-*m')

plt.plot(d1,p1[:,8],'k+',label='original')
plt.plot(d2,p2[:,10],'k+',label='original')
plt.plot(d3,p3[:,14],'k+',label='original')


#plt.legend(loc="upper right")
#plt.title("Pekeris case, monopole source, y=-36m")
plt.xlabel("distance [m]")
plt.ylabel("de-rating %")
#plt.legend
plt.grid()
#plt.ylim([0,10])
#plt.xlim([1,7.5])



#find the intersections
def func0(x):
    return a * np.exp(-b * x**2  ) + c*x +d - dr
dr =5.5

[a,b,c,d] = popt1
root1 = fsolve(func0, x0=2)
print(f"The root is: {root1[0]}")
[a,b,c,d] = popt2
root2 = fsolve(func0, x0=2)
print(f"The root is: {root2[0]}")
[a,b,c,d] = popt3
root3 = fsolve(func0, x0=2)
print(f"The root is: {root3[0]}")

plt.show()
plt.savefig('figure.png',format='png', dpi=300)


plt.figure(2)
plt.plot([8,10,12,14,16,18,20,22],comp[0:8,1],"-+k",label="competitor H=0")
plt.plot([8,10,12,14,16,18,20,22],comp[8:16,1],"-*k",label="competitor Hmax")
plt.plot([10, 16, 22],[1.4741,	2.7857,	3.924],"-+m",label="DR=5.5%")
plt.plot([10, 16, 22],[1.6026,	2.8536,	4.0261],"-+b",label="DR=5%")
plt.plot([10, 16, 22],[1.8762,	3.0149,	4.2857],"-+r",label="DR=4%")
plt.plot([10, 16, 22],[2.0278,	3.1146,	4.46387],"-+g",label="DR=3.5%")
plt.plot([10, 16, 22],[2.1959,	3.235,	4.7086],"-+c",label="DR=3%")
plt.legend(loc="upper left")
plt.legend
plt.xlabel("FAN n.")
plt.ylabel("Distance [m]")
plt.grid()
plt.show()


#H and L
data10 = pd.read_csv("jumbo_10_data",sep='\t')
print("load data 10",data10)

data16 = pd.read_csv("jumbo_16_data0.txt",sep='\t')

print("load data 16",data16)

data20 = pd.read_csv("jumbo_20_data0.txt",sep='\t')

print("load data 20",data20)

data22 = pd.read_csv("jumbo_22_data0.txt",sep='\t')

print("load data 22",data22)


#interpolate to find D3 for each L
dataframes = {
    'data10': data10,
    'data16': data16,
    'data20': data20,
    'data22': data22
}

for name, df in dataframes.items():
    print(f"DataFrame {name}:", df)
    unique = df["L"].unique()
    gruppi = {}

    for valore in unique:
        gruppo_df = df[df['L'] == valore][["N","L", "D", "DRm"]]
        gruppi[valore] = gruppo_df

        print(f"\nGruppo L={valore}:\n", gruppo_df)

        try:
            if gruppo_df.shape[0] < 2:
                print(f"⚠️ Not enough data points to interpolate for L={valore}. Skipping.")
                D3 = gruppo_df['D'].values[0] 
            else:
                D3 = find_coordinate(gruppo_df, 3.0, known="DRm")
            print(f"→ Interpolated D for DRm=3.0: {D3}")
            print(f"→ Corresponding N value: {gruppo_df['N'].values[0]}")
            # Add new row to the original DataFrame
            new_row = {'N': gruppo_df["N"].values[0], 'L': valore, 'D': D3, 'DRm': 3.0}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        except Exception as e:
            print(f"⚠️ Interpolation failed for L={valore}: {e}")
        df.sort_values(by=['D'], inplace=True) 
    dataframes[name] = df
    print("\nUpdated DataFrame with interpolated values:\n", df)


print(dataframes['data20'])

estratti = {}

for df in [data10, data16, data20, data22]:
    unique = df["L"].unique()

    for valore in unique:
        # Filtra il gruppo con L == valore
        gruppo = df[df["L"] == valore]

        # Estrai solo le righe con DRm == 3.0
        filtro = gruppo[gruppo["DRm"] == 3.0]

        # Salva nel dizionario se non è vuoto
        if not filtro.empty:
            chiave = f"{valore}"
            estratti[chiave] = filtro
            print(f"\n🔍 Gruppo L={valore} con DRm=3.0:\n", filtro)
print("estratti",estratti)

data10=data10.to_numpy()
data16=data16.to_numpy()
data20=data20.to_numpy()
data22=data22.to_numpy()

#load Guentner Data
dataG=pd.read_csv("comp_guentner.txt",sep='\t')
print("load data Guentner",dataG)
dataG=dataG.to_numpy()




#Competitor plot
plt.figure(3)
#plt.plot([comp[0,0],comp[8,0]], [comp[0,1],comp[8,1]], label="c-08 fan")
plt.plot([comp[1,0],comp[9,0]], [comp[1,1],comp[9,1]],color='darkgray',linewidth=2, linestyle='--', label="c-10 fan")

#plt.plot([comp[2,0],comp[10,0]], [comp[2,1],comp[10,1]], label="c-12 fan")
#plt.plot([comp[3,0],comp[11,0]], [comp[3,1],comp[11,1]], label="c-14 fan")
plt.plot([comp[4,0],comp[12,0]], [comp[4,1],comp[12,1]],color='gray',linewidth=2, linestyle='--', label="c-16 fan")
#plt.plot([comp[5,0],comp[13,0]], [comp[5,1],comp[13,1]], label="c-18 fan")
plt.plot([comp[6,0],comp[14,0]], [comp[6,1],comp[14,1]], color="lightgrey", linewidth=2, linestyle='--', label="c-20 fan")
plt.plot([comp[7,0],comp[15,0]], [comp[7,1],comp[15,1]],color='dimgray',linewidth=2, linestyle='--', label="c-22 fan")

#Parametric analysis
#H0a=np.array([0,0,0,0,0])
H0a=np.array([0])
#data10H0=np.array([1.4741,1.6026,1.8762,2.0278,2.195])
#dr10=np.array([5.5,5,4,3.5,3])
#only 3%
annotate_=True
if annotate_==True:
    data10H0=np.array([2.195])
    dr10=np.array([3])
    plt.plot(H0a,data10H0,"*b",markersize=6 , label="TK 10 fan")
    for i, txt in enumerate(dr10):
        plt.annotate(f'{txt:.3f}', (H0a[i],data10H0[i]), textcoords="offset points", xytext=(0,5),fontsize=6, ha='center')

    #data16H0=np.array( [2.7857,2.8536,3.014,3.1146,3.235])
    data16H0=np.array( [3.235])
    plt.plot(H0a,data16H0,"*r",markersize=6 , label="TK 16 fan")
    for i, txt in enumerate(dr10):
        plt.annotate(f'{txt:.3f}', (H0a[i],data16H0[i]), textcoords="offset points", xytext=(0,5),fontsize=6, ha='center')

    #data22H0=np.array([3.924,4.0261,4.2857,4.46387,4.7086])
    data22H0=np.array([4.7086])
    plt.plot(H0a,data22H0,"*g",markersize=6 , label="TK 22 fan")
    for i, txt in enumerate(dr10):
        plt.annotate(f'{txt:.3f}', (H0a[i],data22H0[i]), textcoords="offset points", xytext=(0,5),fontsize=6, ha='center')

    data20H0=np.array([4.2])
    #plt.plot([0,3.559],[4.2,2.179],color="orange",marker="*", markersize=6, label="TK 20 fan")
    #for i, txt in enumerate(dr10):
    #    plt.annotate(f'{txt:.3f}', (H0a[i],data20H0[i]), textcoords="offset points", xytext=(0,5),fontsize=6, ha='center')

    #

    plt.plot(data10[:,2],data10[:,1],"*b")
    # Annotate a specific point
    for i, txt in enumerate(data10[:,10]):
        plt.annotate(f'{txt:.3f}', (data10[i,2],data10[i,1]), textcoords="offset points", xytext=(0,5),fontsize=6, ha='center')
    plt.plot(data16[:,2],data16[:,1],"*r")
    for i, txt in enumerate(data16[:,-1]):
        plt.annotate(f'{txt:.3f}', (data16[i,2],data16[i,1]), textcoords="offset points", xytext=(0,5),fontsize=6, ha='center')
    plt.plot(data20[:,2],data20[:,1],"*",color="orange")
    for i, txt in enumerate(data20[:,-1]):
        plt.annotate(f'{txt:.3f}', (data20[i,2],data20[i,1]), textcoords="offset points", xytext=(0,5),fontsize=6, ha='center')
    plt.plot(data22[:,2],data22[:,1],"*g")
    for i, txt in enumerate(data22[:,-1]):
        plt.annotate(f'{txt:.3f}', (data22[i,2],data22[i,1]), textcoords="offset points", xytext=(0,5),fontsize=6, ha='center')
    #PLOT POINT OF 20FANS H3.9m
    x=3.9
    y=2.1
    plt.plot(x,y,"*")
    plt.annotate(10.90, (3.9,2.1), textcoords="offset points", xytext=(0,5),fontsize=6, ha='center')
#
#PLOT LINES
data10all = dataframes['data10']
data10all =data10all[data10all["DRm"] == 3.0]
data10all = data10all.to_numpy()
#
data16all = dataframes['data16']
data16all =data16all[data16all["DRm"] == 3.0]
data16all = data16all.to_numpy()
#
data20all = dataframes['data20']
data20all =data20all[data20all["DRm"] == 3.0]
data20all = data20all.to_numpy()
#data22
data22all = dataframes['data22']
data22all =data22all[data22all["DRm"] == 3.0]
data22all = data22all.to_numpy()

print("data10all",data10all)
print("data16all",data16all)
print("data20all",data20all)
print("data22all",data22all)


plt.plot(data10all[:,2],data10all[:,1],"-",color="blue", label="TK 10 fan")
plt.plot(data16all[:,2],data16all[:,1],"-",color="red", label="TK 16 fan")
plt.plot(data20all[:,2],data20all[:,1],"-",color="orange", label="TK 20 fan")
plt.plot(data22all[:,2],data22all[:,1],"-",color="green", label="TK 22 fan")


#PLOT GUENTNER DATA
gunt=False
if gunt:
    dataG10=dataG[0:6,:]
    print(dataG10)
    plt.plot(dataG10[:,2],2*dataG10[:,1],linewidth=2, label="Guntner -10 fan")
    dataG16=dataG[6:-1,:]
    print(dataG16)
    plt.plot(dataG16[:,2],2*dataG16[:,1],linewidth=2, label="Guntner -16 fan")


plt.legend(loc="upper right")
plt.legend
plt.xlabel("H [m]",fontsize = 12)
plt.ylabel("D [m]",fontsize = 12)
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
plt.grid()
axes=plt.gca()
#axes.set_aspect(0.8)
plt.savefig('studio.png',format='png', dpi=300)
#plt.show()


#D at DRm=3% H=0
fan=[10,16,22]
dr3=[2.1959,3.235,4.7086]
interpDr=interp1d(fan, dr3, kind='quadratic', fill_value="extrapolate")
dr3_20=interpDr(20)
fan=[10,16,20,22]
dr3=[2.1959,3.235,dr3_20,4.7086]
print(dr3_20)

plt.figure(4)
plt.plot(fan,dr3,"-*",color="blue")
plt.legend(loc="upper right")
plt.legend
plt.xlabel("fan ",fontsize = 12)
plt.ylabel("Dr3 [m]",fontsize = 12)
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
plt.grid()
axes=plt.gca()
#axes.set_aspect(0.8)
plt.savefig('DR3.png',format='png', dpi=300)
plt.show()

#interpolat on N fan
x_common = np.linspace(0.00001, 4, num=25)
print("x_common",x_common)
interp1 = interp1d(data10all[:,2], data10all[:,1], kind='linear', bounds_error=False, fill_value='extrapolate')
interp2 = interp1d(data16all[:,2], data16all[:,1], kind='linear', bounds_error=False, fill_value='extrapolate')
interp3 = interp1d(data20all[:,2], data20all[:,1], kind='linear', bounds_error=False, fill_value='extrapolate')
interp4 = interp1d(data22all[:,2], data22all[:,1], kind='linear', bounds_error=False, fill_value='extrapolate')

y_10 = interp1(x_common)
y_16 = interp2(x_common)
y_20 = interp3(x_common)
y_22 = interp4(x_common)

def interpolate_across_N(y1, y2, N1, N2, N_target):
    w = (N_target - N1) / (N2 - N1)
    return (1 - w) * y1 + w * y2

fansi=[12,14,18,20]
y_12 = interpolate_across_N(y_10, y_16, 10, 16, 12)
y_14 = interpolate_across_N(y_10, y_16, 10, 16, 14)
y_18 = interpolate_across_N(y_16, y_20, 16, 20, 18)
#y_22 = interpolate_across_N(y_16, y_20, 16, 20, 22)
plt.figure(5)

plt.plot(x_common, y_12, label=f'N={12}', linestyle='--')
plt.plot(x_common, y_14, label=f'N={14}', linestyle='--')
plt.plot(x_common, y_18, label=f'N={18}', linestyle='--')
#plt.plot(x_common, y_22, label=f'N={22}', linestyle='--')

plt.plot(x_common, y_10,  "-",color="blue", label="TK 10 fan")
plt.plot(x_common, y_16,  "-",color="red", label="TK 16 fan")
plt.plot(x_common, y_20, "-",color="orange", label="TK 20 fan")
plt.plot(x_common, y_22, "-",color="green", label="TK 22 fan")

plt.legend(loc="upper right")
plt.legend
plt.xlabel("H [m]",fontsize = 12)
plt.ylabel("D [m]",fontsize = 12)
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
plt.grid()
plt.title("Interpolated Curve Across Parameter N")
plt.savefig('interpolated_curves.png',format='png', dpi=300)