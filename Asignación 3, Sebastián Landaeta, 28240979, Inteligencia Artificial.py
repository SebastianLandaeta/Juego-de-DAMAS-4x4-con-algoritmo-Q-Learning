# Importar librerías.
import pygame
import time
import numpy as np
import pickle
import os

# Clase que maneja todo lo relacionado al algoritmo Q-Learning
class QLearning:
    # Constructor de la clase
    def __init__(self, alpha=0.7, gamma=0.95, epsilon=0.5, epsilon_decay=0.995, q_table_file="q_table.pkl"):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table_file = q_table_file
        self.load_q_table()
    
    # Guarda la tabla Q en un archivo
    def save_q_table(self):
        with open(self.q_table_file, "wb") as f:
            pickle.dump(self.q_table, f)

    # Carga el archivo con los datos de las partidas pasadas
    def load_q_table(self):
        if os.path.exists(self.q_table_file):
            with open(self.q_table_file, "rb") as f:
                self.q_table = pickle.load(f)
                print("Tabla Q cargada exitosamente.")
        else:
            print("No se encontró una tabla Q previa, iniciando desde cero.")

    # Convierte el tablero (lista de listas) en una tupla de tuplas
    def get_state_key(self, tablero):
        return tuple(tuple(fila) for fila in tablero)

    # Selecciona una acción usando la política epsilon-greedy
    def choose_action(self, state, acciones_validas):
        if not acciones_validas:
            return None
        
        acciones_validas = list(acciones_validas)
        if np.random.uniform(0, 1) < self.epsilon:
            return acciones_validas[np.random.randint(len(acciones_validas))]
        else:
            q_values = [self.q_table.get((state, accion), 0) for accion in acciones_validas]
            return acciones_validas[np.argmax(q_values)]

   # Actualiza el valor Q de una acción usando la ecuación de Bellman
    def update_q_value(self, estado_anterior, accion, recompensa, nuevo_estado):
        if estado_anterior is None or accion is None:
            return
        old_value = self.q_table.get((estado_anterior, accion), 0)
        max_future = max([self.q_table.get((nuevo_estado, a), 0) for a in self.get_possible_actions(nuevo_estado)] or [0])
        new_value = old_value + self.alpha * (recompensa + self.gamma * max_future - old_value)
        self.q_table[(estado_anterior, accion)] = new_value

    # Genera todas las acciones válidas para el estado actual
    def get_possible_actions(self, state):
        acciones = []
        tablero = [list(fila) for fila in state]
        for x in range(4):
            for y in range(4):
                if tablero[x][y] == 'I':
                    for dx in [-1, 1]:
                        for dy in [-1, 1]:
                            if 0 <= x+dx < 4 and 0 <= y+dy < 4 and tablero[x+dx][y+dy] is None:
                                acciones.append((x, y, x+dx, y+dy))
                            if 0 <= x+2*dx < 4 and 0 <= y+2*dy < 4 and tablero[x+dx][y+dy] == 'H' and tablero[x+2*dx][y+2*dy] is None:
                                acciones.append((x, y, x+2*dx, y+2*dy))
        return acciones

    # Reduce epsilon multiplicándolo por epsilon_decay.
    def update_epsilon(self):
        self.epsilon = max(0.1, self.epsilon * self.epsilon_decay)

# Clase que maneja todo lo relacionado al propio juego
class DamasJuego:
    # Constructor de la clase
    def __init__(self):
        pygame.init()
        self.tamano_casilla = 100 
        self.ancho_tablero = 4
        self.altura_extra = 50
        self.tablero = [[None for _ in range(4)] for _ in range(4)]
        self.pieza_seleccionada = None
        self.turno_humano = True # El humano juega primero
        self.ronda = 1
        self.ia_qlearning = QLearning()
        self.estado_anterior_ia = None
        self.accion_anterior_ia = None
        self.screen = pygame.display.set_mode((self.tamano_casilla * 4, self.tamano_casilla * 4 + self.altura_extra))
        pygame.display.set_caption("Juego de DAMAS")
        self.fuente = pygame.font.SysFont(None, 40)
        self.inicializar_tablero()

    # Configuración del tablero
    def inicializar_tablero(self):
        self.tablero = [[None for _ in range(4)] for _ in range(4)]
        self.tablero[3][1] = 'H'
        self.tablero[3][3] = 'H'
        self.tablero[0][0] = 'I'
        self.tablero[0][2] = 'I'
        self.dibujar_tablero()

    def dibujar_tablero(self):
        for i in range(4):
            for j in range(4):
                # Dibujar casilla
                color = (0, 0, 0) if (i + j) % 2 == 0 else (255, 255, 255)
                pygame.draw.rect(self.screen, color, (j * self.tamano_casilla, i * self.tamano_casilla, 
                                                    self.tamano_casilla, self.tamano_casilla))
                
                # Resaltar casilla seleccionada con borde verde
                if self.pieza_seleccionada == (i, j):
                    pygame.draw.rect(self.screen, (0, 255, 0), 
                                (j * self.tamano_casilla, i * self.tamano_casilla, 
                                    self.tamano_casilla, self.tamano_casilla), 5)  # Grosor del borde: 5px
                
                # Dibujar fichas
                if self.tablero[i][j] == 'H':
                    pygame.draw.circle(self.screen, (200, 200, 200), # Las fichas humanas son grises
                                    (j * self.tamano_casilla + 50, i * self.tamano_casilla + 50), 33)
                elif self.tablero[i][j] == 'I':
                    pygame.draw.circle(self.screen, (255, 0, 0), # Las fichas de la IA son rojas
                                    (j * self.tamano_casilla + 50, i * self.tamano_casilla + 50), 33)
        
        # Barra inferior
        pygame.draw.rect(self.screen, (50, 50, 50), (0, 400, 400, 50))
        texto_ronda = self.fuente.render(f"Ronda: {self.ronda}", True, (255, 255, 255))
        self.screen.blit(texto_ronda, (10, 410))
        pygame.display.flip()

    # Mueve las fichas del humano
    def mover_ficha(self, i, j):
        if self.pieza_seleccionada:
            x, y = self.pieza_seleccionada
            if (i, j) == (x, y):
                self.pieza_seleccionada = None
                self.dibujar_tablero()
                return

            if self.es_movimiento_valido(x, y, i, j):
                self.procesar_movimiento(x, y, i, j)
                if self.juego_terminado():
                    self.finalizar_juego()
                else:
                    self.turno_humano = False
                    time.sleep(1)
                    self.ia_juega()

    def procesar_movimiento(self, x, y, i, j):
        if abs(x - i) == 2:
            x_medio, y_medio = (x + i) // 2, (y + j) // 2
            self.tablero[x_medio][y_medio] = None
        self.tablero[i][j] = self.tablero[x][y]
        self.tablero[x][y] = None
        self.pieza_seleccionada = None
        self.dibujar_tablero()

    def es_movimiento_valido(self, x, y, i, j):
        if self.tablero[i][j] is not None:
            return False
        dx = abs(x - i)
        dy = abs(y - j)
        if dx == 1 and dy == 1:
            return True
        if dx == 2 and dy == 2:
            x_medio, y_medio = (x + i) // 2, (y + j) // 2
            return self.tablero[x_medio][y_medio] == 'I' if self.turno_humano else self.tablero[x_medio][y_medio] == 'H'
        return False
    
    # Mueve las fichas de la IA
    def ia_juega(self):
        estado_actual = self.ia_qlearning.get_state_key(self.tablero)
        acciones_validas = self.ia_qlearning.get_possible_actions(estado_actual)
        
        if self.accion_anterior_ia is not None:
            recompensa = self.calcular_recompensa()
            self.ia_qlearning.update_q_value(self.estado_anterior_ia, self.accion_anterior_ia, recompensa, estado_actual)
        
        accion = self.ia_qlearning.choose_action(estado_actual, acciones_validas)
        
        if accion:
            x, y, i, j = accion
            self.procesar_movimiento(x, y, i, j)
            self.estado_anterior_ia = estado_actual
            self.accion_anterior_ia = accion
            self.turno_humano = True
            self.ronda += 1
            self.ia_qlearning.update_epsilon()
            if self.juego_terminado():
                self.finalizar_juego()

    # Sistema de recompensas
    def calcular_recompensa(self):
        tablero_anterior = [list(fila) for fila in self.estado_anterior_ia]
        recompensa = 0
        
        # Captura de fichas
        piezas_ia_antes = sum(fila.count('I') for fila in tablero_anterior)
        piezas_ia_ahora = sum(fila.count('I') for fila in self.tablero)
        recompensa += (piezas_ia_ahora - piezas_ia_antes) * 5
        
        # Avance estratégico
        for i, fila in enumerate(self.tablero):
            if 'I' in fila:
                recompensa += (3 - i) * 0.5  # Priorizar moverse hacia abajo
        
        return recompensa

    def finalizar_juego(self):
        if sum(fila.count('H') for fila in self.tablero) == 0:
            mensaje = "¡IA ha ganado!"
            recompensa_final = 20
        else:
            mensaje = "¡Humano ha ganado!"
            recompensa_final = -20
        
        if self.estado_anterior_ia is not None and self.accion_anterior_ia is not None:
            clave = (self.estado_anterior_ia, self.accion_anterior_ia)
            # Si la clave no existe, inicialízala con 0
            if clave not in self.ia_qlearning.q_table:
                self.ia_qlearning.q_table[clave] = 0
            self.ia_qlearning.q_table[clave] += recompensa_final
        
        self.mostrar_mensaje(mensaje)
        self.ia_qlearning.save_q_table()
        time.sleep(2)
        self.reiniciar_juego()
    
    # Al terminar una partida, se crea otra 2 segundos despues
    def reiniciar_juego(self):
        self.inicializar_tablero()
        self.ronda = 1
        self.turno_humano = True

    def mostrar_mensaje(self, mensaje):
        self.screen.fill((0, 0, 0))
        
        # Renderizar el texto
        texto = self.fuente.render(mensaje, True, (255, 255, 255))
        
        # Calcular posición centrada
        pos_x = (self.screen.get_width() - texto.get_width()) // 2
        pos_y = (self.screen.get_height() - self.altura_extra - texto.get_height()) // 2  # Centrar en área del tablero
        
        # Dibujar texto
        self.screen.blit(texto, (pos_x, pos_y))
        
        pygame.display.flip()

    def juego_terminado(self):
        return (sum(fila.count('H') for fila in self.tablero) == 0 or 
                sum(fila.count('I') for fila in self.tablero) == 0 or 
                self.ronda >= 64)

# Función principal
if __name__ == "__main__":
    juego = DamasJuego()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                i, j = y // juego.tamano_casilla, x // juego.tamano_casilla
                if juego.turno_humano and juego.tablero[i][j] == 'H':
                    juego.pieza_seleccionada = (i, j)
                else:
                    juego.mover_ficha(i, j)
        juego.dibujar_tablero()
    pygame.quit()