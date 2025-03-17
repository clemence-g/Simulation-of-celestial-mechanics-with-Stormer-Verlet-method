# -*- coding: utf-8 -*-
"""
Created on Sat May  4 17:20:05 2024

@author: User
"""
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


#PROJET DIFFERENCES FINIES

###########################################################
##################### CONSTANTES ##########################
###########################################################

#Constantes (en unités astronomiques : masse solaire, jour, UA)
m_s = 1 #masse du soleil
m_j = 0.000954  #masse de jupiter
m_sat = 0.0002857 #masse de saturne
G = 0.0002960149122113483 #en UA^3/(ms*jour)

#Temps :
#t_max = 1826250 #5000 ans
t_max = 365 *200
dt =  365 #pas de temps 
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

def heun() : 
    
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

heun()


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

def SVerlet() :

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

SVerlet()

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
line_sun, = ax.plot(qs[0,0], qs[1,0], qs[2,0], 'black', label='Sun')
line_jupiter, = ax.plot(qj[0,0], qj[1,0], qj[2,0], 'red', label='Jupiter')
line_saturn, = ax.plot(qsat[0,0], qsat[1,0], qsat[2,0], 'green', label='Saturn')

ax.legend()

#fonction d'animation
#time_text = None
def update(frame):
    line_sun.set_data(qs[0,:frame+1], qs[1,:frame+1])  #Mise à jour des coordonnées du Soleil
    line_sun.set_3d_properties(qs[2,:frame+1])

    line_jupiter.set_data(qj[0,:frame+1], qj[1,:frame+1])  #Mise à jour des coordonnées de Jupiter
    line_jupiter.set_3d_properties(qj[2,:frame+1])

    line_saturn.set_data(qsat[0,:frame+1], qsat[1,:frame+1])  #Mise à jour des coordonnées de Saturne
    line_saturn.set_3d_properties(qsat[2,:frame+1])
    
    ax.set_title(f"Animation of trajectories of the Sun, Jupiter and Saturn with {modee} \n t = {round(frame*dt/365,0)} years \n dt = {dt} days")
   
    return line_sun, line_jupiter, line_saturn,


def init():
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])
    ax.set_zlim([-15, 15]) 

    ax.plot([], [], [], 'black')
    return line_sun, line_jupiter, line_saturn

# Création de l'animation
ani = FuncAnimation(fig, update, frames=N, init_func=init, interval = 1, blit=False)


#Retirer les # pour enregistrer un fichier mp4 de l'animation
HTML(ani.to_html5_video())
FFwriter = animation.FFMpegWriter(fps=10)
ani.save(filename= 'animation-planètes-{}.mp4'.format(modee), writer=FFwriter)
