# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:43:09 2024

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 18:33:48 2024

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

G = 0.0002960149122113483 #en UA^3/(ms*jour)

#Temps :
#t_max = 1826250 #5000 ans
t_max = 365*30
dt =  3 #pas de temps 
N = int(np.floor(t_max/dt))+1 #nombre d'étapes de calcul
time = np.linspace(0,t_max,N) #c'est le temps qui passe

#Masses des planètes

sys_mass = {
            'planete_a' : 1,
            'planete_b': 0.001,
            }


#CONDITIONS INITIALES
#Accessibles sur http://vo.imcce.fr/webservices/miriade/?forms
#01/04/2024 à 00:00, ref héliocentrique, coordonnées rectangulaires

#positions initiales

sys_q0 = {
            'planete_a' : (0,0,0),
            'planete_b': (-0.3327785015450,0.1022269278657,0.0891005567242),
}

#quantités de mouvement initiales

sys_p0 = {
            'planete_a' : (0,0,0),
            'planete_b': (sys_mass["planete_b"]*-0.0159904596657,sys_mass["planete_b"]*-0.0227338671153,sys_mass["planete_b"]*-0.0104871235041),
 }

#Enregistrement des positions et impulsions

q_a = np.zeros((3,N)) 
p_a = np.zeros((3,N))

q_b = np.zeros((3,N)) 
p_b = np.zeros((3,N))


sys_q = {
    "planete_a" : q_a,
    "planete_b" : q_b,
    }

sys_p = {
    "planete_a" : p_a,
    "planete_b" : p_b,
    }

for key in sys_q.keys():
    sys_q[key][:,0] = sys_q0[key]
    sys_p[key][:,0] = sys_p0[key]


###########################################################
##################### VERLET ##############################
###########################################################

#Calcul par la méthode de Stormer-Verlet

def rap(a,b,i):
    
    #calcule le rapport r_ij/|rij|**3
    
    r_ab = a[:,i] - b[:,i]
    norme3 = np.linalg.norm(r_ab)**3
    
    if norme3 != 0:
        return r_ab/norme3
    
    elif norme3 == 0:
        return np.zeros(3)
    
def interaction(planete_1, planete_2,i):
    #attention, quand on appelle la fonction il faut mettre des guillemets, ie : interaction("sun","jupiter",i)
    q1 = sys_q[planete_1]
    q2 = sys_q[planete_2]
    m1 = sys_mass[planete_1]
    m2 = sys_mass[planete_2]

    return G * m1 * m2 * rap(q1, q2, i)


def SVerlet() :

    for i in range(N-1) : 

        #impulsion tilde
        
        p_a_tilde = p_a[:,i]
        
        for key in sys_q.keys():
            if key != "planete_a":
                      p_a_tilde += dt/2*interaction(key,"planete_b",i)
                      
        
        p_b_tilde = p_b[:,i]
        
        for key in sys_q.keys():
            if key != "planete_b":
                      p_b_tilde += dt/2*interaction(key,"planete_b",i)

                                   
        #CM
        cm1 = 0
        for key in sys_q.keys():
            cm1 += sys_q[key][:,i]*sys_mass[key]
            
        sum_mass = 0
        for key in sys_q.keys():
            sum_mass += sys_mass[key]
            
        cm = cm1/sum_mass

        #q
        #mise à jour de la position
        q_a[:,i+1] = q_a[:,i] + dt*(p_a_tilde/sys_mass['planete_a']) - cm
        q_b[:,i+1] = q_b[:,i] + dt*(p_b_tilde/sys_mass["planete_b"]) - cm
        
 
        #p
        #mise à jour de l'impulsion

        p_a[:,i+1] = p_a_tilde
        
        for key in sys_q.keys():
            if key != "planete_a":
                      p_a[:,i+1] += dt/2*interaction(key,"planete_a",i+1)
                
        p_b[:,i+1] = p_b_tilde
        
        for key in sys_q.keys():
            if key != "planete_b":
                      p_b[:,i+1] += dt/2*interaction(key,"planete_b",i+1)
                      

SVerlet()

###########################################################
##################### ANIMATION ###########################
###########################################################


# Création de la figure et de l'axe 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Trajectoires initiales
line_a, = ax.plot(q_a[0,0], q_a[1,0], q_a[2,0], label='Planete a')
line_b, = ax.plot(q_b[0,0], q_b[1,0], q_b[2,0], label='Planete b')



ax.legend()

#fonction d'animation

def update(frame):
    
    #Mise à jour des coordonnées des planètes
    
    line_a.set_data(q_a[0,:frame+1], q_a[1,:frame+1])
    line_a.set_3d_properties(q_a[2,:frame+1])
    
    line_b.set_data(q_b[0,:frame+1], q_b[1,:frame+1])  
    line_b.set_3d_properties(q_b[2,:frame+1])    
    
    
    ax.set_title(f"Two body problem calculated with Verlet Method \n t = {round(frame*dt/365,0)} years \n dt = {dt} days")
   
    return line_a, line_b


def init():
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5]) 

    ax.plot([], [], [], 'black')
    return line_a, line_b

# Création de l'animation
ani = FuncAnimation(fig, update, frames=N, init_func=init, interval = 30, blit=False)

#blit=True : Seuls les éléments qui changent sont redessinés : plus rapide mais moins fiable car 3D (ie: zoom ne marche pas)
#interval = 30 -> intervalle en milliseconde entre les frames (ici: ne pas mettre plus petit que 1)

#Retirer les # pour enregistrer un fichier mp4 de l'animation
#HTML(ani.to_html5_video())
#FFwriter = animation.FFMpegWriter(fps=10)
#ani.save(filename= 'animation-planètes.mp4', writer=FFwriter)
