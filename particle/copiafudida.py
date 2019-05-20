#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esta classe deve conter todas as suas implementações relevantes para seu filtro de partículas
"""

from pf import Particle, draw_random_sample
import numpy as np
import inspercles # necessário para o a função nb_lidar que simula o laser
import math
from scipy import stats

largura = 775 # largura do mapa
altura = 748  # altura do mapa

# Robo
robot = Particle(largura/2, altura/2, math.pi/4, 1.0)

# Nuvem de particulas
particulas = []

num_particulas = 1000


# Os angulos em que o robo simulado vai ter sensores
angles = np.linspace(0.0, 2*math.pi, num=16, endpoint=False)

# Lista mais longa
movimentos_longos = [[-10, -10, 0], [-10, 10, 0], [-10,0,0], [-10, 0, 0],
              [0,0,math.pi/12.0], [0, 0, math.pi/12.0], [0, 0, math.pi/12],[0,0,-math.pi/4],
              [-5, 0, 0],[-5,0,0], [-5,0,0], [-10,0,0],[-10,0,0], [-10,0,0],[-10,0,0],[-10,0,0],[-15,0,0],
              [0,0,-math.pi/4],[0, 10, 0], [0,10,0], [0, 10, 0], [0,10,0], [0,0,math.pi/8], [0,10,0], [0,10,0], 
              [0,10,0], [0,10,0], [0,10,0],[0,10,0],
              [0,0,-math.radians(90)],
              [math.cos(math.pi/3)*10, math.sin(math.pi/3),0],[math.cos(math.pi/3)*10, math.sin(math.pi/3),0],[math.cos(math.pi/3)*10, math.sin(math.pi/3),0],
              [math.cos(math.pi/3)*10, math.sin(math.pi/3),0]]

# Lista curta
movimentos_curtos = [[-10, -10, 0], [-10, 10, 0], [-10,0,0], [-10, 0, 0]]

movimentos_relativos = [[0, -math.pi/3],[10, 0],[10, 0], [10, 0], [10, 0],[15, 0],[15, 0],[15, 0],[0, -math.pi/2],[10, 0],
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
                       [0, -math.pi/2], 
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
                       [0, -math.pi/2], 
                       [10,0], [0, -math.pi/4], [10,0], [10,0], [10,0],
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0],
                       [10,0], [10, 0], [10, 0], [10, 0], [10, 0], [10, 0]]



movimentos = movimentos_relativos

def cria_particulas(minx=0, miny=0, maxx=largura, maxy=altura, n_particulas=num_particulas):
    particle_cloud = []
    for i in range(n_particulas):
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        theta = np.random.uniform(0, math.pi)
        prob = Particle(x, y, theta, w=1.0) # A prob. w vai ser normalizada depois
        particle_cloud.append(prob)
    return particle_cloud
    
def move_particulas(particulas, movimento):
    for particula in particulas:  
      distancia, rotacao = movimento
      movimento[0] = stats.norm.rvs(loc=distancia, scale=1)
      movimento[1] = stats.norm.rvs(loc=rotacao, scale=math.radians(1))
      particula.move_relative(movimento)

    return None
    
def leituras_laser_evidencias(robot, particulas):
    prob_total = 0

    leitura_robo = inspercles.nb_lidar(robot, angles)
    for particula in particulas:
      #  Reset das probs para cada particula e set da leitura
      soma_probs = 0
      leitura_particula = inspercles.nb_lidar(particula, angles)

      #  Soma das probs
      for angulo in leitura_robo.keys():
        soma_probs += stats.norm.pdf(leitura_particula[angulo], leitura_robo[angulo], scale=8)

      #  Append na prob total
      prob_total += soma_probs
      particula.w = soma_probs

    #  Inverso da prob total
    for particula in particulas:
      particula.normalize(prob_total)
    
def reamostrar(particulas, n_particulas = num_particulas):

    particulas_pesos = [p.w for p in particulas]
    novas_particulas = draw_random_sample(particulas, particulas_pesos, 1000)

    #  Reamostragem com incertezas
    for particula in novas_particulas:
      particula.w = 1
      particula.x += stats.norm.rvs(scale = 6)
      particula.y += stats.norm.rvs(scale = 6)
      particula.theta += stats.norm.rvs(scale = math.radians(7))

    return novas_particulas