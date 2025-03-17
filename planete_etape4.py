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
t_max = 365
dt =  1 #pas de temps 
N = int(np.floor(t_max/dt))+1 #nombre d'étapes de calcul
time = np.linspace(0,t_max,N) #c'est le temps qui passe

#Masses des planètes

sys_mass = {
            'sun' : 1,
            'mercury': 1.137 * 10**-7,
            'venus': 2.4478383 *10**-6,	
            'earth': 3.0034896 * 10**-6,
            'moon' : 3.7*10**-8,
            'mars': 3.2271514 * 10**-7,
            'jupiter':0.000954,
            'saturn':0.0002857,
            'uranus':4.365785 * 10**-5,
            'neptune': 5.150314 * 10**-5
            }


#CONDITIONS INITIALES
#Accessibles sur http://vo.imcce.fr/webservices/miriade/?forms
#01/04/2024 à 00:00, ref héliocentrique, coordonnées rectangulaires

#positions initiales

sys_q0 = {
            'sun' : (0,0,0),
            'mercury': (-0.3327785015450,0.1022269278657,0.0891005567242),
            'venus': (0.6354456214219,	-0.3070593176010,-0.1783689808518),	
            'earth': (-0.9794601405886,-0.1815443077220,-0.0786877614111),
            'moon' : (-0.9795752842881,-0.1838039347675,-0.0799137153938),
            'mars': (0.9398508267644,-0.9228243743591,-0.4486361467002),
            'jupiter': (2.9621859199066, 4.0324519403416,-0.0830242875291),
            'saturn': (9.1493153567909,-2.8526967544563,-1.5722157314376),
            'uranus':(11.9866561269502,14.2650107445850,6.0780181428544),
            'neptune': (29.8563952124881,-1.1179477681737,-1.2008534619168)
            }

#quantités de mouvement initiales

sys_p0 = {
            'sun' : (0,0,0),
            'mercury': (sys_mass["mercury"]*-0.0159904596657,sys_mass["mercury"]*-0.0227338671153,sys_mass["mercury"]*-0.0104871235041),
            'venus': (sys_mass["venus"]*0.0096965327418,sys_mass["venus"]*0.0162749712740,sys_mass["venus"]*0.0067097080096),	
            'earth': (sys_mass["earth"]*0.0031204802390,sys_mass["earth"]*-0.015532070788,sys_mass["earth"]*-0.0067327866431),
            'moon' : (sys_mass["moon"]*0.0037079902825,sys_mass["moon"]*-0.0155195079017,sys_mass["moon"]*-0.0067422299670),
            'mars': (sys_mass["mars"]*0.0108438608442,sys_mass["mars"]*0.0097921726251,sys_mass["mars"]*0.0041988885774),
            'jupiter': (sys_mass['jupiter']*(-0.0061745080703), sys_mass['jupiter']*0.0048273346504, sys_mass['jupiter']*0.0001180901210),
            'saturn': (sys_mass['saturn']*0.0015501223838,sys_mass['saturn']*0.0048799887954,sys_mass['saturn']*0.0019488702242),
            'uranus':(sys_mass['uranus']*-0.0031466147003,sys_mass['uranus']*0.0020246456904,sys_mass['uranus']*0.0009312082542),
            'neptune': (sys_mass['neptune']*0.0001319903611,sys_mass['neptune']*0.0029264143507,sys_mass['neptune']*0.0011945870713)
            }

#Enregistrement des positions et impulsions

q_s = np.zeros((3,N)) 
p_s = np.zeros((3,N))

q_me = np.zeros((3,N)) 
p_me = np.zeros((3,N))

q_v = np.zeros((3,N)) 
p_v = np.zeros((3,N))

q_e = np.zeros((3,N)) 
p_e = np.zeros((3,N))

q_mo = np.zeros((3,N)) 
p_mo = np.zeros((3,N))

q_ma = np.zeros((3,N)) 
p_ma = np.zeros((3,N))

q_j = np.zeros((3,N))
p_j = np.zeros((3,N))

q_sat = np.zeros((3,N))
p_sat = np.zeros((3,N))

q_u = np.zeros((3,N)) 
p_u = np.zeros((3,N))

q_n = np.zeros((3,N)) 
p_n = np.zeros((3,N))


sys_q = {
    "sun" : q_s,
    "mercury" : q_me,
    "venus" : q_v,
    "earth" : q_e,
    "moon" : q_mo,
    "mars" : q_ma,
    "jupiter": q_j,
    "saturn": q_sat,
    "uranus" : q_u,
    "neptune" : q_n
    }

sys_p = {
    "sun" : p_s,
    "mercury" : p_me,
    "venus" : p_v,
    "earth" :p_e,
    "moon" : p_mo,
    "mars" : p_ma,
    "jupiter": p_j,
    "saturn": p_sat,
    "uranus" : p_u,
    "neptune" : p_n
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
    #attention, quand on appelle la fonciton il faut mettre des guillemets, ie : interaction("sun","jupiter",i)
    q1 = sys_q[planete_1]
    q2 = sys_q[planete_2]
    m1 = sys_mass[planete_1]
    m2 = sys_mass[planete_2]

    return G * m1 * m2 * rap(q1, q2, i)


def SVerlet() :

    for i in range(N-1) : 

        #impulsion tilde
        
        p_s_tilde = p_s[:,i]
        
        for key in sys_q.keys():
            if key != "sun":
                      p_s_tilde += dt/2*interaction(key,"sun",i)
                      
        
        p_me_tilde = p_me[:,i]
        
        for key in sys_q.keys():
            if key != "mercury":
                      p_me_tilde += dt/2*interaction(key,"mercury",i)

        p_v_tilde = p_v[:,i]
        
        for key in sys_q.keys():
            if key != "venus":
                      p_v_tilde += dt/2*interaction(key,"venus",i)
        
        p_e_tilde = p_e[:,i]
        
        for key in sys_q.keys():
            if key != "earth":
                      p_e_tilde += dt/2*interaction(key,"earth",i)
        
        p_mo_tilde = p_mo[:,i]
        
        for key in sys_q.keys():
            if key != "moon":
                      p_mo_tilde += dt/2*interaction(key,"moon",i)

        p_ma_tilde = p_ma[:,i]
        
        for key in sys_q.keys():
            if key != "mars":
                      p_ma_tilde += dt/2*interaction(key,"mars",i)
                      
        p_j_tilde = p_j[:,i]
        
        for key in sys_q.keys():
            if key != "jupiter":
                      p_j_tilde += dt/2*interaction(key,"jupiter",i)

        p_sat_tilde = p_sat[:,i]
        
        for key in sys_q.keys():
            if key != "saturn":
                      p_sat_tilde += dt/2*interaction(key,"saturn",i)

        p_u_tilde = p_u[:,i]
        
        for key in sys_q.keys():
            if key != "uranus":
                      p_u_tilde += dt/2*interaction(key,"uranus",i)   
                     
        p_n_tilde = p_n[:,i]
        
        for key in sys_q.keys():
            if key != "neptune":
                      p_n_tilde += dt/2*interaction(key,"neptune",i)
                                   
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
        q_s[:,i+1] = q_s[:,i] + dt*(p_s_tilde/sys_mass['sun']) - cm
        q_me[:,i+1] = q_me[:,i] + dt*(p_me_tilde/sys_mass["mercury"]) - cm
        q_v[:,i+1] = q_v[:,i] + dt*(p_v_tilde/sys_mass["venus"]) - cm
        q_e[:,i+1] = q_e[:,i] + dt*(p_e_tilde/sys_mass["earth"]) - cm 
        q_mo[:,i+1] = q_mo[:,i] + dt*(p_mo_tilde/sys_mass["moon"]) - cm        
        q_ma[:,i+1] = q_ma[:,i] + dt*(p_ma_tilde/sys_mass["mars"]) - cm
        q_j[:,i+1] = q_j[:,i] + dt*(p_j_tilde/sys_mass["jupiter"]) - cm
        q_sat[:,i+1] = q_sat[:,i] + dt*(p_sat_tilde/sys_mass["saturn"]) - cm
        q_u[:,i+1] = q_u[:,i] + dt*(p_u_tilde/sys_mass["uranus"]) - cm
        q_n[:,i+1] = q_n[:,i] + dt*(p_n_tilde/sys_mass["neptune"]) - cm
        
 
        #p
        #mise à jour de l'impulsion

        p_s[:,i+1] = p_s_tilde
        
        for key in sys_q.keys():
            if key != "sun":
                      p_s[:,i+1] += dt/2*interaction(key,"sun",i+1)
                
        p_me[:,i+1] = p_me_tilde
        
        for key in sys_q.keys():
            if key != "mercury":
                      p_me[:,i+1] += dt/2*interaction(key,"mercury",i+1)

        p_v[:,i+1] = p_v_tilde
        
        for key in sys_q.keys():
            if key != "venus":
                      p_v[:,i+1] += dt/2*interaction(key,"venus",i+1)

        p_e[:,i+1] = p_e_tilde
        
        for key in sys_q.keys():
            if key != "earth":
                      p_e[:,i+1] += dt/2*interaction(key,"earth",i+1)
                     
        p_mo[:,i+1] = p_mo_tilde
        
        for key in sys_q.keys():
            if key != "moon":
                      p_mo[:,i+1] += dt/2*interaction(key,"moon",i+1)

        p_ma[:,i+1] = p_ma_tilde
        
        for key in sys_q.keys():
            if key != "mars":
                      p_ma[:,i+1] += dt/2*interaction(key,"mars",i+1)

        p_j[:,i+1] = p_j_tilde
        
        for key in sys_q.keys():
            if key != "jupiter":
                      p_j[:,i+1] += dt/2*interaction(key,"jupiter",i+1)

        p_sat[:,i+1] = p_sat_tilde
        
        for key in sys_q.keys():
            if key != "saturn":
                      p_sat[:,i+1] += dt/2*interaction(key,"saturn",i+1)
                      
        p_u[:,i+1] = p_u_tilde
        
        for key in sys_q.keys():
            if key != "uranus":
                      p_u[:,i+1] += dt/2*interaction(key,"uranus",i+1)
                      
        p_n[:,i+1] = p_n_tilde
        
        for key in sys_q.keys():
            if key != "neptune":
                      p_n[:,i+1] += dt/2*interaction(key,"neptune",i+1)
                      

SVerlet()

###########################################################
##################### ANIMATION ###########################
###########################################################


# Création de la figure et de l'axe 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Trajectoires initiales
line_s, = ax.plot(q_s[0,0], q_s[1,0], q_s[2,0], label='Sun')
line_me, = ax.plot(q_me[0,0], q_me[1,0], q_me[2,0], label='Mercury')
line_v, = ax.plot(q_v[0,0], q_v[1,0], q_v[2,0], label='Venus')
line_e, = ax.plot(q_e[0,0], q_e[1,0], q_e[2,0], label='Earth')
line_mo, = ax.plot(q_mo[0,0], q_mo[1,0], q_mo[2,0], label='Moon')
line_ma, = ax.plot(q_ma[0,0], q_ma[1,0], q_ma[2,0], label='Mars')
line_j, = ax.plot(q_j[0,0], q_j[1,0], q_j[2,0], label='Jupiter')
line_sat, = ax.plot(q_sat[0,0], q_sat[1,0], q_sat[2,0], label='Saturn')
line_u, = ax.plot(q_u[0,0], q_u[1,0], q_u[2,0], label='Uranus')
line_n, = ax.plot(q_n[0,0], q_n[1,0], q_n[2,0], label='Neptune')


ax.legend()

#fonction d'animation

def update(frame):
    
    #Mise à jour des coordonnées des planètes
    
    line_s.set_data(q_s[0,:frame+1], q_s[1,:frame+1])
    line_s.set_3d_properties(q_s[2,:frame+1])
    
    line_me.set_data(q_me[0,:frame+1], q_me[1,:frame+1])  
    line_me.set_3d_properties(q_me[2,:frame+1])    

    line_v.set_data(q_v[0,:frame+1], q_v[1,:frame+1])  
    line_v.set_3d_properties(q_v[2,:frame+1])
    
    line_e.set_data(q_e[0,:frame+1], q_e[1,:frame+1])  
    line_e.set_3d_properties(q_e[2,:frame+1])
    
    line_mo.set_data(q_mo[0,:frame+1], q_mo[1,:frame+1])  
    line_mo.set_3d_properties(q_mo[2,:frame+1])

    line_ma.set_data(q_ma[0,:frame+1], q_ma[1,:frame+1]) 
    line_ma.set_3d_properties(q_ma[2,:frame+1])    
    
    line_j.set_data(q_j[0,:frame+1], q_j[1,:frame+1])
    line_j.set_3d_properties(q_j[2,:frame+1])

    line_sat.set_data(q_sat[0,:frame+1], q_sat[1,:frame+1]) 
    line_sat.set_3d_properties(q_sat[2,:frame+1])
    
    line_u.set_data(q_u[0,:frame+1], q_u[1,:frame+1])
    line_u.set_3d_properties(q_u[2,:frame+1])
    
    line_n.set_data(q_n[0,:frame+1], q_n[1,:frame+1])  
    line_n.set_3d_properties(q_n[2,:frame+1])
    
    
    ax.set_title(f"Motion of the solar system calculated with Verlet Method \n t = {round(frame*dt/365,0)} years \n dt = {dt} days")
   
    return line_s, line_me, line_v, line_e, line_mo, line_ma, line_j, line_sat, line_u, line_n


def init():
    ax.set_xlim([-30, 30])
    ax.set_ylim([-30, 30])
    ax.set_zlim([-30, 30]) 

    ax.plot([], [], [], 'black')
    return line_s, line_me, line_v, line_e, line_mo, line_ma, line_j, line_sat, line_u, line_n

# Création de l'animation
ani = FuncAnimation(fig, update, frames=N, init_func=init, interval = 30, blit=False)

#blit=True : Seuls les éléments qui changent sont redessinés : plus rapide mais moins fiable car 3D (ie: zoom ne marche pas)
#interval = 30 -> intervalle en milliseconde entre les frames (ici: ne pas mettre plus petit que 1)

#Retirer les # pour enregistrer un fichier mp4 de l'animation
#HTML(ani.to_html5_video())
#FFwriter = animation.FFMpegWriter(fps=10)
#ani.save(filename= 'animation-planètes.mp4', writer=FFwriter)