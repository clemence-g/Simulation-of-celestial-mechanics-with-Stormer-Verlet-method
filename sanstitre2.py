# -*- coding: utf-8 -*-
"""
Created on Wed May  8 18:14:15 2024

@author: User
"""

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

#PROJET DIFFERENCES FINIES

#CODE POUR LE PROJET DIFFERENCES FINIES

#PARTIE AVEC JUPITER ET LE SOLEIL


###########################################################
##################### CONSTANTES ##########################
###########################################################

#Constantes (unités astronomiques : masse solaire, jour, UA)
m_j = 0.000954  #en masse solaire 
m_s = 1 #en masse solaire
G = 0.0002960149122113483 #en UA^3/(ms*jour)
const_pot = G*m_j*m_s

#Temps :
t_max = 30*365
#t_max = 1826250 #le temps max qu'on considère (en jours)
dt = 30 #le pas de temps 
N = int(np.floor(t_max/dt))+1 #nombre d'étapes de calcul
T = N*dt #ça va permettre qu'à chaque pas de temps on a une certaine position qui est enregistrée (temps total)
time = np.linspace(0,t_max,N) #c'est le temps qui passe 



###########################################################
##################### HEUN ################################
###########################################################

#Positions et impulsions : 

q_s = np.zeros((3,N)) 
p_s = np.zeros((3,N))

q_j = np.zeros((3,N))
p_j = np.zeros((3,N))

#Conditions initiales : 
#Les conditions initiales du soleil sont nulles pour le moment car on le prend en heliocentré 

q_s[:,0] = (0,0,0)
p_s[:,0] = (0,0,0)

#Nous avons choisi 

q_j[:,0] = (2.9621859199066, 4.0324519403416,-0.0830242875291) #position initiale de Jupiter à la date du 1 avril 2024
p_j[:,0] = (m_j*(-0.0061745080703), m_j*0.0048273346504, m_j*0.0001180901210) #c'est l'impulsion de jupiter à la date du 1 avril 2024

#On définit nos matrices qui nous seront utiles pour appliquer la méthode de Heun par après : 
#je me demande si c'est pas mieux de les définir dans la boucle finalement et dans la def de heun quoi   

Q_tilde_s = np.zeros((3,N))
P_tilde_s = np.zeros((3,N))

Q_tilde_j = np.zeros((3,N))
P_tilde_j = np.zeros((3,N))

def heun_2() : 
    for i in range(N-1) :

        diff = q_j[:,i] - q_s[:,i]
        norme = np.linalg.norm(q_j[:,i] - q_s[:,i])

        Q_tilde_s[:,i+1] = q_s[:,i] + (p_s[:,i]/m_s)*dt

        Q_tilde_j[:,i+1] = q_j[:,i] + (p_j[:,i]/m_j)*dt

        P_tilde_s[:,i+1] = p_s[:,i] - ((const_pot*diff)/norme**3)*dt

        P_tilde_j[:,i+1] = p_j[:,i] + ((const_pot*diff)/norme**3)*dt


        diff_tilde = Q_tilde_j[:,i+1] - Q_tilde_s[:,i+1]  #la différence de Q_tilde_j et Q_tilde_s est évaluée en i+1
        norme_tilde = np.linalg.norm(Q_tilde_j[:,i+1] - Q_tilde_s[:,i+1])  #cette norme est aussi évaluée en i+1

        #CM (centre de masse)
        cm = (q_s[:,i]*m_s+q_j[:,i]*m_j)/(m_s+m_j)

        q_s [:,i+1] = q_s[:,i] + (p_s[:,i]/m_s + P_tilde_s[:,i+1]/m_s)*dt/2 - cm

        q_j[:,i+1] = q_j[:,i] + (p_j[:,i]/m_j + P_tilde_j[:,i+1]/m_j )*dt/2 - cm

        p_s[:,i+1] = p_s[:,i] + const_pot*(((diff)/norme**3) + (diff_tilde/norme_tilde**3))*dt/2

        p_j[:,i+1] = p_j[:,i] - const_pot*(((diff)/norme**3) + (diff_tilde/norme_tilde**3))*dt/2

heun_2()

fig = plt.figure()
ax = plt.axes(projection='3d')

# defining all 3 axis

xj = q_j[0,:]
yj = q_j[1,:]
zj = q_j[2,:]

xs = q_s[0,:]
ys = q_s[1,:]
zs = q_s[2,:]

# plotting
ax.plot(xj, yj, zj, 'peru', label = "Jupiter")
ax.plot(xs,ys,zs, 'hotpink', label = "Soleil")
ax.set_title("Positions de Jupiter et du Soleil avec la méthode de Heun")
ax.legend()
plt.show()

###########################################################
##################### VERLET ##############################
###########################################################

q_sv = np.zeros((3,N)) 
p_sv = np.zeros((3,N))

q_jv = np.zeros((3,N))
p_jv = np.zeros((3,N))

#Les conditions initiales du soleil sont nulles pour le moment car on le prend en heliocentré 

q_sv[:,0] = (0,0,0)
p_sv[:,0] = (0,0,0)

q_jv[:,0] = (2.9621859199066, 4.0324519403416,-0.0830242875291) #position initiale de Jupiter à la date du 1 avril 2024
p_jv[:,0] = (m_j*(-0.0061745080703), m_j*0.0048273346504, m_j*0.0001180901210) #c'est l'impulsion de jupiter à la date du 1 avril 2024



def SVerlet_2() : 



   for i in range(N-1) : 

        diff_v = q_jv[:,i] - q_sv[:,i]
        norme_v = np.linalg.norm(diff_v)

        #tilde

        p_sv_tilde = p_sv[:,i] + dt/2*(const_pot*(diff_v)/norme_v**3)
        p_jv_tilde = p_jv[:,i] - dt/2*(const_pot*(diff_v)/norme_v**3)  

        cm = (q_sv[:,i]*m_s+q_jv[:,i]*m_j)/(m_s+m_j)
        #q

        q_sv[:,i+1] = q_sv[:,i] + dt*(p_sv_tilde/m_s) - cm
        q_jv[:,i+1] = q_jv[:,i] + dt*(p_jv_tilde/m_j) - cm

        #norme
        diff_v_tilde = q_jv[:,i+1] - q_sv[:,i+1]
        norme_v_tilde = np.linalg.norm(diff_v_tilde)

        #p
        p_sv[:,i+1] = p_sv_tilde + dt/2*(const_pot*(diff_v_tilde)/norme_v_tilde**3)
        p_jv[:,i+1] = p_jv_tilde - dt/2*(const_pot*(diff_v_tilde)/norme_v_tilde**3)


SVerlet_2()

fig = plt.figure()
ax = plt.axes(projection='3d')

# defining all 3 axis

xj = q_jv[0,:]
yj = q_jv[1,:]
zj = q_jv[2,:]

xs = q_sv[0,:]
ys = q_sv[1,:]
zs = q_sv[2,:]      

# plotting
ax.plot(xj, yj, zj, 'peru', label = "Jupiter")
ax.plot(xs,ys,zs, 'hotpink', label = "Soleil")
ax.set_title('Positions de Jupiter et du Soleil avec la méthode de Stormer-Verlet')
ax.legend()
plt.show()

#PARTIE AVEC JUPITER, LE SOLEIL ET SATURNE
###########################################################
##################### CONSTANTES ##########################
###########################################################

#Constantes (en unités astronomiques : masse solaire, jour, UA)
m_sat = 0.0002857 #masse de saturne

G = 0.0002960149122113483 #en UA^3/(ms*jour)

#Temps :
t_max = 1826250 #5000 ans
#t_max = 30*365 #temps pour les tests
dt = 30 #le pas de temps 
N = int(np.floor(t_max/dt))+1 #nombre d'étapes de calcul
time = np.linspace(0,t_max,N) #c'est le temps qui passe

###########################################################
##################### HEUN ################################
###########################################################

#Positions et impulsions : 

q_s = np.zeros((3,N)) 
p_s = np.zeros((3,N))

q_j = np.zeros((3,N))
p_j = np.zeros((3,N))

q_sat = np.zeros((3,N))
p_sat = np.zeros((3,N))

#Conditions initiales
#Les conditions initiales du soleil sont nulles pour que le repère soit iniitialement héliocentré

q_s[:,0] = (0,0,0)
p_s[:,0] = (0,0,0)

#Positions et impulsions de Jupiter et Saturne le 01/04/2024 à minuit

q_j[:,0] = (2.9621859199066, 4.0324519403416,-0.0830242875291)
p_j[:,0] = (m_j*(-0.0061745080703), m_j*0.0048273346504, m_j*0.0001180901210)

q_sat[:,0] = (9.1493153567909,-2.8526967544563,-1.5722157314376)
p_sat[:,0] = (m_sat*0.0015501223838,m_sat*0.0048799887954,m_sat*0.0019488702242)
 
#Calcul par la méthode de Heun

def heun_3() : 
    
    Q_tilde_s = np.zeros((3,N))
    P_tilde_s = np.zeros((3,N))

    Q_tilde_j = np.zeros((3,N))
    P_tilde_j = np.zeros((3,N))

    Q_tilde_sat = np.zeros((3,N))
    P_tilde_sat = np.zeros((3,N))

    for i in range(N-1) :

        diff_j_s = q_j[:,i] - q_s[:,i]
        norme_j_s = np.linalg.norm(diff_j_s)

        diff_sat_s = q_sat[:,i] - q_s[:,i]
        norme_sat_s = np.linalg.norm(diff_sat_s)

        diff_sat_j = q_sat[:,i] - q_j[:,i]
        norme_sat_j = np.linalg.norm(diff_sat_j)


        Q_tilde_s[:,i+1] = q_s[:,i] + (p_s[:,i]/m_s)*dt
        Q_tilde_j[:,i+1] = q_j[:,i] + (p_j[:,i]/m_j)*dt
        Q_tilde_sat[:,i+1] = q_sat[:,i] + (p_sat[:,i]/m_sat)*dt

        P_tilde_s[:,i+1] = p_s[:,i] + dt*(G*m_j*m_s*diff_j_s/norme_j_s**3 + G*m_s*m_sat*diff_sat_s/norme_sat_s**3)
        P_tilde_j[:,i+1] = p_j[:,i] + dt*(-G*m_j*m_s*diff_j_s/norme_j_s**3 + G*m_j*m_sat*diff_sat_j/norme_sat_j**3)
        P_tilde_sat[:,i+1] = p_sat[:,i] + dt*(-G*m_sat*m_j*diff_sat_j/norme_sat_j**3 - G*m_sat*m_s*diff_sat_s/norme_sat_s**3)


        diff_tilde_j_s = Q_tilde_j[:,i+1] - Q_tilde_s[:,i+1]
        norme_tilde_j_s = np.linalg.norm(diff_tilde_j_s)

        diff_tilde_sat_s = Q_tilde_sat[:,i+1] - Q_tilde_s[:,i+1]
        norme_tilde_sat_s = np.linalg.norm(diff_tilde_sat_s)

        diff_tilde_sat_j = Q_tilde_sat[:,i+1] - Q_tilde_j[:,i+1]
        norme_tilde_sat_j = np.linalg.norm(diff_tilde_sat_j)

        #CM
        cm = (q_s[:,i]*m_s+q_j[:,i]*m_j+q_sat[:,i]*m_sat)/(m_s+m_j+m_sat)

        q_s[:,i+1] = q_s[:,i] + dt/2 * (p_s[:,i]/m_s + P_tilde_s[:,i+1]/m_s) - cm
        q_j[:,i+1] = q_j[:,i] + dt/2 * (p_j[:,i]/m_j + P_tilde_j[:,i+1]/m_j) - cm
        q_sat[:,i+1] = q_sat[:,i] + dt/2 * (p_sat[:,i]/m_sat + P_tilde_sat[:,i+1]/m_sat) - cm

        p_s[:,i+1] = p_s[:,i] + dt/2 * ((G*m_j*m_s*diff_j_s/norme_j_s**3 + G*m_sat*m_s * diff_sat_s/norme_sat_s**3)+(G*m_j*m_s*diff_tilde_j_s/norme_tilde_j_s**3+ G*m_sat*m_s*diff_tilde_sat_s/norme_tilde_sat_s**3))

        p_j[:,i+1] = p_j[:,i] + dt/2 * ((-G*m_j*m_s*diff_j_s/norme_j_s**3 + G*m_sat*m_j*diff_sat_j/norme_sat_j**3) + (-G*m_j*m_s*diff_tilde_j_s/norme_tilde_j_s**3 + G*m_sat*m_j*diff_tilde_sat_j/norme_tilde_sat_j**3))

        p_sat[:,i+1] = p_sat[:,i] + dt/2 * ((-G*m_sat*m_j*diff_sat_j/norme_sat_j**3 - G*m_sat*m_s*diff_sat_s/norme_sat_s**3) + (-G*m_sat*m_j*diff_tilde_sat_j/norme_tilde_sat_j**3 - G*m_sat*m_s*diff_tilde_sat_s/norme_tilde_sat_s**3))

heun_3()

#PLOT

fig = plt.figure()
ax = plt.axes(projection='3d')

xj = q_j[0,:]
yj = q_j[1,:]
zj = q_j[2,:]

xs = q_s[0,:]
ys = q_s[1,:]
zs = q_s[2,:]

xsat = q_sat[0,:]
ysat = q_sat[1,:]
zsat = q_sat[2,:]

ax.plot(xs,ys,zs, 'hotpink', label = "Sun")
ax.plot(xj, yj, zj, 'peru', label = "Jupiter")
ax.plot(xsat,ysat,zsat, "mediumturquoise", label = "Saturn")
ax.set_title('Trajectoires de Jupiter, du Soleil et de Saturne avec la méthode de Heun : \n t={} jours and dt = {} jours'.format(t_max,dt), fontsize=12)
ax.legend()
plt.show()

###########################################################
##################### VERLET ##############################
###########################################################

q_sv = np.zeros((3,N)) 
p_sv = np.zeros((3,N))

q_jv = np.zeros((3,N))
p_jv = np.zeros((3,N))

q_satv = np.zeros((3,N))
p_satv = np.zeros((3,N))

#Conditions initiales (idem que pour Heun)

q_sv[:,0] = (0,0,0)
p_sv[:,0] = (0,0,0)

q_jv[:,0] = (2.9621859199066, 4.0324519403416,-0.0830242875291)
p_jv[:,0] = (m_j*(-0.0061745080703), m_j*0.0048273346504, m_j*0.0001180901210)

q_satv[:,0] = (9.1493153567909,-2.8526967544563,-1.5722157314376)
p_satv[:,0] = (m_sat*0.0015501223838,m_sat*0.0048799887954,m_sat*0.0019488702242)

#Calcul par la méthode de Stormer-Verlet

def SVerlet_3() :

    for i in range(N-1) : 

        diff_v_j_s = q_jv[:,i] - q_sv[:,i]
        norme_v_j_s = np.linalg.norm(diff_v_j_s)

        diff_v_sat_s = q_satv[:,i] - q_sv[:,i]
        norme_v_sat_s = np.linalg.norm(diff_v_sat_s)

        diff_v_sat_j = q_satv[:,i] - q_jv[:,i]
        norme_v_sat_j = np.linalg.norm(diff_v_sat_j)


        #tilde

        p_sv_tilde = p_sv[:,i] + dt/2*(G*m_s*m_j*diff_v_j_s/norme_v_j_s**3 + G*m_s*m_sat*diff_v_sat_s/norme_v_sat_s**3)
        p_jv_tilde = p_jv[:,i] + dt/2*(-G*m_j*m_s*diff_v_j_s/norme_v_j_s**3 + G*m_sat*m_j*diff_v_sat_j/norme_v_sat_j**3)
        p_satv_tilde = p_satv[:,i] + dt/2 *(-G*m_sat*m_j*diff_v_sat_j/norme_v_sat_j**3 - G*m_sat*m_s*diff_v_sat_s/norme_v_sat_s**3)

        #CM
        cm = (q_sv[:,i]*m_s+q_jv[:,i]*m_j+q_satv[:,i]*m_sat)/(m_s+m_j+m_sat)

        #q

        q_sv[:,i+1] = q_sv[:,i] + dt*(p_sv_tilde/m_s) - cm
        q_jv[:,i+1] = q_jv[:,i] + dt*(p_jv_tilde/m_j) - cm
        q_satv[:,i+1] = q_satv[:,i] + dt*(p_satv_tilde/m_sat) - cm

        #norme des tildes
        diff_v_tilde_j_s = q_jv[:,i+1] - q_sv[:,i+1]
        norme_v_tilde_j_s = np.linalg.norm(diff_v_tilde_j_s)

        diff_v_tilde_sat_s = q_satv[:,i+1] - q_sv[:,i+1]
        norme_v_tilde_sat_s = np.linalg.norm(diff_v_tilde_sat_s)

        diff_v_tilde_sat_j = q_satv[:,i+1] - q_jv[:,i+1]
        norme_v_tilde_sat_j = np.linalg.norm(diff_v_tilde_sat_j)

        #p
        p_sv[:,i+1] = p_sv_tilde + dt/2*(G*m_s*m_j*diff_v_tilde_j_s/norme_v_tilde_j_s**3 + G*m_s*m_sat*diff_v_tilde_sat_s/norme_v_tilde_sat_s**3)
        p_jv[:,i+1] = p_jv_tilde + dt/2*(-G*m_s*m_j*diff_v_tilde_j_s/norme_v_tilde_j_s**3 + G*m_sat*m_j*diff_v_tilde_sat_j/norme_v_tilde_sat_j**3)
        p_satv[:,i+1] = p_satv_tilde + dt/2*(-G*m_sat*m_s*diff_v_tilde_sat_s/norme_v_tilde_sat_s**3 - G*m_sat*m_j*diff_v_tilde_sat_j/norme_v_tilde_sat_j**3)

#PLOT

SVerlet_3()

fig = plt.figure()
ax = plt.axes(projection='3d')

xvj = q_jv[0,:]
yvj = q_jv[1,:]
zvj = q_jv[2,:]

xvs = q_sv[0,:]
yvs = q_sv[1,:]
zvs = q_sv[2,:]      

xvsat = q_satv[0,:]
yvsat = q_satv[1,:]
zvsat = q_satv[2,:]   

ax.plot(xvs,yvs,zvs, 'hotpink', label = "Soleil")
ax.plot(xvj, yvj, zvj, 'peru', label = "Jupiter")
ax.plot(xvsat, yvsat, zvsat, "mediumturquoise", label = "Saturne")
ax.legend()

ax.set_title('Trajectoires de Jupiter, du Soleil et de SAturne en utilisant Stormer-Verlet : \n t={} jours et dt = {} jours'.format(t_max,dt), fontsize=12)
plt.show()  


###########################################################
##################### INVARIANTS ##########################
###########################################################

#On pense que les invariants sont l'énergie, le moment angulaire et la quantité de mouvement

#Calcul de l'ENERGIE en fonction du temps, mode = "heun" ou mode = "verlet"
def E_3(mode):

    E_tot_list = []

    if mode == "heun":
        qs = q_s
        qj = q_j
        qsat = q_sat

        ps = p_s
        pj = p_j
        psat = p_sat

    elif mode == "verlet":
        qs = q_sv
        qj = q_jv
        qsat = q_satv

        ps = p_sv
        pj = p_jv
        psat = p_satv

    for t in range (N):   

        E_tot = (np.linalg.norm(ps[:,t])**2)/(2*m_s) + (np.linalg.norm(pj[:,t])**2)/(2*m_j) + (np.linalg.norm(psat[:,t])**2)/(2*m_sat) - G*m_s*m_j/np.linalg.norm(qj[:,t]-qs[:,t]) - G*m_sat*m_j/np.linalg.norm(qsat[:,t]-qj[:,t]) - G*m_sat*m_s/np.linalg.norm(qsat[:,t]-qs[:,t])
        E_tot_list.append(E_tot)


    return (E_tot_list)

dataH = E_3("heun")
dataV = E_3("verlet")
fig = plt.figure() 
plt.plot(time,dataH, "purple", label = "Heun")
plt.plot(time,dataV, "steelblue", label = "Verlet")
plt.legend()
plt.xlabel("Temps [Jours]")
plt.ylabel("Energie [Joules]")
plt.title(" Energie totale des 3 corps pour les différentes méthodes")
plt.show()

#QUANTITE DE MOUVEMENT en fonction du temps

def p(mode):
    p_tot_list = []
    
    if mode == "heun": 
        ps = p_s
        pj = p_j
        psat = p_sat
        
    elif mode == "verlet":  
        ps = p_sv
        pj = p_jv
        psat = p_satv
        
    for t in range (N):
        p_tot_list.append(ps[:,t]+pj[:,t]+psat[:,t])
        
        
    return(np.array(p_tot_list))

dataPH = p("heun")
fig = plt.figure()   
plt.plot(time,dataPH[:,0], "coral", label = "$P_{x}$")
plt.plot(time,dataPH[:,1], "slateblue", label = "$P_{y}$")
plt.plot(time,dataPH[:,2], "teal", label = "$P_{z}$")
plt.legend()
plt.title("Total momentum with HEUN \n t = {} and dt = {}".format(t_max,dt))
plt.show()    

dataPV = p("verlet")
fig = plt.figure()   
plt.plot(time,dataPV[:,0], "coral", label = "$P_{x}$")
plt.plot(time,dataPV[:,1], "slateblue", label = "$P_{y}$")
plt.plot(time,dataPV[:,2], "teal", label = "$P_{z}$")
plt.legend()
plt.title("Total momentum with VERLET \n t = {} and dt = {}".format(t_max,dt))
plt.show()

#MOMENT ANGULAIRE TOTAL en fonction du temps
#Rappel : le moment angulaire est le produit vectoriel du vecteur position et du vecteur impulsion
#Le moment angulaire total du système est la somme des moments angulaires de chaque objet dans le système

def L(mode):
    L_list=[]

    if mode == "heun":
        qs = q_s
        qj = q_j
        qsat = q_sat

        ps = p_s
        pj = p_j
        psat = p_sat

    elif mode == "verlet":
        qs = q_sv
        qj = q_jv
        qsat = q_satv

        ps = p_sv
        pj = p_jv
        psat = p_satv

    for t in range (N):
        #calcul du produit vectoriel composante par composante:
        ls = np.cross(qs[:,t], ps[:,t])
        lj = np.cross(qj[:,t], pj[:,t])
        lsat = np.cross(qsat[:,t], psat[:,t])

        L_tot = ls+lj+lsat
        L_list.append(L_tot)

    return np.array(L_list)


dataL = L("heun")
fig = plt.figure()   
plt.plot(time,dataL[:,0], "coral", label="$L_{x}$")
plt.plot(time,dataL[:,1], "slateblue", label="$L_{y}$")
plt.plot(time,dataL[:,2], "teal", label="$L_{z}$")
plt.xlabel("Temps [jours]")
plt.ylabel("Moment angulaire total")
plt.title("Moment angulaire total du système à 3 corps avec la méthode de Heun \n t = {} and dt = {}".format(t_max,dt))
plt.legend()
plt.show()    

dataL = L("verlet")
fig = plt.figure()   
plt.plot(time,dataL[:,0],"coral", label="$L_{x}$")
plt.plot(time,dataL[:,1],"slateblue", label="$L_{y}$")
plt.plot(time,dataL[:,2], "teal", label="$L_{z}$")
plt.xlabel("Temps [jours]")
plt.ylabel("Moment angulaire total")
plt.title("Moment angulaire total du système à 3 corps avec la méthode de Stormer-Verlet \n t = {} and dt = {}".format(t_max,dt))
plt.legend()
plt.show()

###########################################################
##################### ANIMATION ###########################
###########################################################

#Ici choisir la méthode "heun" ou "verlet"
modee = "verlet"

def mode(mode):
    if mode == "heun":
        qs = q_s
        qj = q_j
        qsat = q_sat


    elif mode == "verlet":
        qs = q_sv
        qj = q_jv
        qsat = q_satv

    return qs,qj,qsat
qs,qj,qsat = mode(modee)

# Création de la figure et de l'axe 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Trajectoires initiales
line_sun, = ax.plot(qs[0,0], qs[1,0], qs[2,0], 'hotpink', label='Sun')
line_jupiter, = ax.plot(qj[0,0], qj[1,0], qj[2,0], 'peru', label='Jupiter')
line_saturn, = ax.plot(qsat[0,0], qsat[1,0], qsat[2,0], 'mediumturquoise', label='Saturn')

ax.set_title('Trajectories of the Sun, Jupiter and Saturn with {} \n t = {} days and dt = {} days'.format(modee, t_max, dt))
ax.legend()

#fonction d'animation

def update(frame):
    line_sun.set_data(qs[0,:frame+1], qs[1,:frame+1])  # Mise à jour des coordonnées du Soleil
    line_sun.set_3d_properties(qs[2,:frame+1])

    line_jupiter.set_data(qj[0,:frame+1], qj[1,:frame+1])  # Mise à jour des coordonnées de Jupiter
    line_jupiter.set_3d_properties(qj[2,:frame+1])

    line_saturn.set_data(qsat[0,:frame+1], qsat[1,:frame+1])  # Mise à jour des coordonnées de Saturne
    line_saturn.set_3d_properties(qsat[2,:frame+1])

    return line_sun, line_jupiter, line_saturn

def init():
    ax.set_xlim([-15, 15])  # Ajustez les limites selon vos besoins
    ax.set_ylim([-15, 15])
    ax.set_zlim([-15, 15]) 

    ax.plot([], [], [], 'black')  # Ajustez les limites selon vos besoins
    return line_sun, line_jupiter, line_saturn

# Création de l'animation
ani = FuncAnimation(fig, update, frames=N, init_func=init, interval = 1, blit=True)
plt.show()

#Sert à enregistrer un fichier mp4 de l'animation : le fichier se trouve dans "Ce PC" je pense
#HTML(ani.to_html5_video())
#FFwriter = animation.FFMpegWriter(fps=10)
#ani.save(filename= 'animation-planètes-{}.mp4'.format(modee), writer=FFwriter)