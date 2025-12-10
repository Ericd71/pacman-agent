# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
#
# Implementación de un equipo ofensivo coordinado para el Pacman Capture the Flag:
# - Dos agentes ofensivos coordinados (DualOffensiveAgent).
# - Uso de A* con coste personalizado para evitar ghosts y colisiones entre compañeros.
# - Roles dinámicos: "distractor" (atrae y entretiene al defensor) y "cleaner" (optimiza comer comida).
# - Modo especial de "endgame defense" para cerrar partidas cuando conviene.

import random
import math
import time

import contest.util as util
from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='DualOffensiveAgent', second='DualOffensiveAgent', num_training=0):
    """
    Crea el equipo de Pacman.

    En este proyecto usamos siempre dos agentes coordinados DualOffensiveAgent,
    que se reparten dinámicamente los roles en ataque.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class OffensiveAStarAgent(CaptureAgent):
    """
    Agente ofensivo base que usa A* para buscar caminos.

    Comportamiento general:
    - Decide si está en modo ataque normal o en modo "volver a casa".
    - Entra al lado enemigo por la frontera más segura.
    - Busca comida relativamente segura, alejándose de ghosts y del compañero.
    - Si está perseguido o lleva demasiada comida, vuelve a casa.
    - Puede entrar en un modo especial de defensa de endgame (en la subclase).

    Este agente sirve como base para DualOffensiveAgent, que añade coordinación global.
    """

    # ======================
    #  INICIALIZACIÓN
    # ======================

    def register_initial_state(self, game_state):
        """
        Inicializa el agente: calcula distancias de laberinto y memoria de comida defendida.
        """
        CaptureAgent.register_initial_state(self, game_state)

        # Posición inicial (utilizada como fallback si falla todo)
        self.start = game_state.get_agent_position(self.index)

        # Precalcular todas las distancias del laberinto
        self.distancer.get_maze_distances()

        # Parámetros de comportamiento ofensivo
        self.max_carry = 6           # comida máxima que se tolera llevar antes de querer volver
        self.return_threshold = 0.6  # umbral del score de retorno

        # Flag de debug (imprime logs si está a True)
        self.debug = False

        # Estado para detectar loops / atascos
        self.position_history = []
        self.stuck_mode = False

        # Cooldown anti-bucle
        self.unstuck_cooldown = 0
        self.last_stuck_center = None

        # Memoria de comida que defendemos / última comida propia comida
        try:
            self.last_defending_food = self.get_food_you_are_defending(game_state).as_list()
        except Exception:
            self.last_defending_food = []
        self.last_eaten_food_pos = None

    # ======================
    #  DEBUG / LOGGING
    # ======================

    def dbg(self, *args):
        """
        Impresión condicionada por la bandera de debug.
        """
        if getattr(self, "debug", False):
            print(f"[OFF-{self.index}]", *args)

    # ========================================
    #  MEMORIA DE COMIDA DEFENDIDA (LOST FOOD)
    # ========================================

    def update_defending_food_memory(self, game_state):
        """
        Detecta comida propia (del lado que defendemos) que ha desaparecido entre turnos.

        Equivalente al manejo de 'lost food' del DefensiveReflexAgent del baseline.
        Guarda en self.last_eaten_food_pos una casilla donde acaban de comer.
        """
        try:
            current = self.get_food_you_are_defending(game_state).as_list()
        except Exception:
            return

        if getattr(self, "last_defending_food", None) is not None:
            before = set(self.last_defending_food)
            now = set(current)
            eaten = before - now
            if eaten:
                # Normalmente solo habrá una casilla comida
                self.last_eaten_food_pos = list(eaten)[0]
                self.dbg(f"Lost food at {self.last_eaten_food_pos}")
        self.last_defending_food = current

    # ======================
    #  UTILIDADES DE LADO
    # ======================

    def is_on_own_side(self, game_state, position):
        """
        Devuelve True si 'position' está en nuestro propio lado del mapa.
        """
        walls = game_state.get_walls()
        mid_x = walls.width // 2
        if self.red:
            return position[0] < mid_x
        else:
            return position[0] >= mid_x

    # ============================
    #  INFO TEAMMATE / DEFENSORES
    # ============================

    def _get_team_indices(self, game_state):
        """
        Devuelve los índices de los agentes del propio equipo.
        (Compatibilidad con getTeam / get_team según versión de capture_agents).
        """
        fn = getattr(self, "getTeam", None) or getattr(self, "get_team", None)
        if fn is None:
            return [self.index]
        return fn(game_state)

    def get_teammate_pos(self, game_state):
        """
        Devuelve la posición del compañero (o None si no se puede obtener).
        """
        indices = self._get_team_indices(game_state)
        for i in indices:
            if i != self.index:
                mate_state = game_state.get_agent_state(i)
                return mate_state.get_position()
        return None

    def get_defender_info(self, game_state, my_pos):
        """
        Devuelve (defensor_más_cercano, distancia_desde_my_pos) o (None, None)
        si no hay ghosts defensores visibles.

        Solo cuenta enemigos que:
        - NO son pacman (ghosts)
        - Tienen posición conocida
        - NO están asustados
        """
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        defenders = [
            e for e in enemies
            if not e.is_pacman and e.get_position() is not None and e.scared_timer == 0
        ]

        if not defenders:
            return None, None

        defender = min(defenders, key=lambda d: self.get_maze_distance(my_pos, d.get_position()))
        dist = self.get_maze_distance(my_pos, defender.get_position())
        return defender, dist

    # ============================
    #  DETECCIÓN DE "STUCK"
    # ============================

    def update_stuck_state(self, game_state):
        """
        Guarda las últimas posiciones y marca stuck_mode=True
        si llevamos varios turnos moviéndonos en una zona muy pequeña
        (bucle típico izquierda-derecha / arriba-abajo).
        """
        pos = game_state.get_agent_position(self.index)
        self.position_history.append(pos)
        if len(self.position_history) > 8:
            self.position_history.pop(0)

        # Por defecto no estamos atascados en este turno
        self.stuck_mode = False

        if len(self.position_history) < 6:
            return

        xs = [p[0] for p in self.position_history]
        ys = [p[1] for p in self.position_history]
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)

        # Si todo sucede en un cuadrado de 2x2, consideramos que estamos en bucle
        if x_range <= 1 and y_range <= 1:
            self.stuck_mode = True
            self.dbg("STUCK DETECTED: small oscillation zone")

            # Activamos cooldown y guardamos el centro aproximado del bucle
            cx = sum(xs) // len(xs)
            cy = sum(ys) // len(ys)
            self.last_stuck_center = (cx, cy)

            # Cooldown mínimo para evitar volver a entrar en el mismo patrón
            if self.unstuck_cooldown < 20:
                self.unstuck_cooldown = 20

    def reset_stuck_state(self):
        """
        Resetea el historial y la bandera de bucle.
        """
        self.position_history = []
        self.stuck_mode = False

    def tick_unstuck_cooldown(self):
        """
        Disminuye el cooldown anti-bucle en cada turno.
        """
        if self.unstuck_cooldown > 0:
            self.unstuck_cooldown -= 1

    # ============================
    #  MODO DEFENSA DE ENDGAME
    # ============================

    def get_home_boundary_positions(self, game_state):
        """
        Devuelve todas las posiciones de la frontera en nuestro lado (columna home_x).
        """
        walls = game_state.get_walls()
        mid_x = walls.width // 2
        if self.red:
            home_x = mid_x - 1
        else:
            home_x = mid_x
        return [(home_x, y) for y in range(walls.height) if not walls[home_x][y]]

    def get_boundary_closest_to(self, game_state, point):
        """
        Devuelve la casilla de frontera de nuestro lado más cercana a 'point'.

        Útil cuando estamos asustados:
        intentamos colocarnos entre el invasor y nuestra base.
        """
        boundary = self.get_home_boundary_positions(game_state)
        if not boundary:
            return point
        return min(boundary, key=lambda b: self.get_maze_distance(point, b))

    def endgame_defense_action(self, game_state):
        """
        Comportamiento de defensa agresiva en endgame.

        Prioridad:
          1) Perseguir invaders visibles.
          2) Si no se ven, ir a la última comida propia comida (lost food).
          3) Proteger cápsulas que defendemos.
          4) Patrullar la frontera, repartiendo parte alta/baja entre los dos
             y situándose cerca de donde queda comida propia.
        """
        my_pos = game_state.get_agent_position(self.index)
        my_state = game_state.get_agent_state(self.index)
        scared = my_state.scared_timer > 0

        # 0) Actualizar memoria de comida defendida (lost food)
        self.update_defending_food_memory(game_state)

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [e for e in enemies if e.is_pacman and e.get_position() is not None]

        # 1) Invaders visibles: perseguimos al más cercano
        if invaders:
            target_invader = min(invaders, key=lambda e: self.get_maze_distance(my_pos, e.get_position()))
            inv_pos = target_invader.get_position()

            if not scared:
                # No asustado → ir directo al invasor
                goal = inv_pos
            else:
                # Asustado → intentamos cortar por la frontera entre él y nuestra base
                goal = self.get_boundary_closest_to(game_state, inv_pos)

            next_step = self.a_star_search_next_step(game_state, my_pos, goal, mode='defend')
            if next_step:
                return self.get_action_from_path(my_pos, next_step)

        # 2) Lost food en nuestro lado
        eaten_pos = getattr(self, "last_eaten_food_pos", None)
        if eaten_pos is not None:
            if my_pos == eaten_pos:
                # Hemos llegado al punto → limpiamos la marca
                self.last_eaten_food_pos = None
            else:
                next_step = self.a_star_search_next_step(game_state, my_pos, eaten_pos, mode='defend')
                if next_step:
                    return self.get_action_from_path(my_pos, next_step)

        # 3) Proteger cápsulas que defendemos
        try:
            defended_capsules = self.get_capsules_you_are_defending(game_state)
        except Exception:
            defended_capsules = []

        if defended_capsules:
            best_cap = min(defended_capsules, key=lambda c: self.get_maze_distance(my_pos, c))
            next_step = self.a_star_search_next_step(game_state, my_pos, best_cap, mode='defend')
            if next_step:
                return self.get_action_from_path(my_pos, next_step)

        # 4) Patrulla de frontera
        walls = game_state.get_walls()
        mid_y = walls.height // 2

        home_boundary = self.get_home_boundary_positions(game_state)
        if not home_boundary:
            return Directions.STOP

        team = self._get_team_indices(game_state)
        candidates = list(home_boundary)

        # Repartir parte superior e inferior entre los dos agentes
        if len(team) >= 2:
            upper_defender = min(team)
            if self.index == upper_defender:
                cand = [p for p in home_boundary if p[1] >= mid_y]
            else:
                cand = [p for p in home_boundary if p[1] < mid_y]
            if cand:
                candidates = cand

        # Si queda comida que defendemos, patrullar cerca de su “banda” vertical
        try:
            food_def = self.get_food_you_are_defending(game_state).as_list()
        except Exception:
            food_def = []

        if food_def:
            avg_y = sum(p[1] for p in food_def) / float(len(food_def))

            def patrol_score(p):
                return abs(p[1] - avg_y) * 2 + self.get_maze_distance(my_pos, p)
        else:
            def patrol_score(p):
                return self.get_maze_distance(my_pos, p)

        best = min(candidates, key=patrol_score)
        next_step = self.a_star_search_next_step(game_state, my_pos, best, mode='defend')
        if next_step:
            return self.get_action_from_path(my_pos, next_step)

        return Directions.STOP

    # ======================
    #  DECISIÓN DE ROL LOCAL
    # ======================

    def decide_role(self, game_state):
        """
        Decide si este agente debe ser 'distractor' o 'cleaner' de forma local.

        Prioridad:
          1) Si el defensor está muy cerca de mí -> distractor.
          2) Si el defensor está claramente más cerca del compañero -> cleaner.
          3) Si está claramente más cerca de mí que del compañero -> distractor.
          4) Si no hay defensor o está muy lejos -> se usa la posición vertical como desempate.
        """
        my_pos = game_state.get_agent_position(self.index)
        my_state = game_state.get_agent_state(self.index)

        defender, dist_def = self.get_defender_info(game_state, my_pos)

        # 0) No hay defensor visible → ambos cleaners
        if defender is None:
            return 'cleaner'

        # Información del compañero
        mate_pos = self.get_teammate_pos(game_state)
        mate_dist = None
        if mate_pos is not None:
            mate_dist = self.get_maze_distance(mate_pos, defender.get_position())

        # 1) Defensor muy cerca de mí → distractor
        if dist_def is not None and dist_def <= 3:
            return 'distractor'

        # 2) Comparativa de distancias al defensor
        if dist_def is not None and mate_dist is not None:
            if mate_dist + 1 < dist_def:
                return 'cleaner'
            elif dist_def + 1 < mate_dist:
                return 'distractor'

        # 3) Si llevo bastante comida y el defensor está lejos → más perfil cleaner
        if my_state.num_carrying >= getattr(self, "max_carry", 6) // 2:
            if dist_def is not None and dist_def >= 6:
                return 'cleaner'

        # 4) Empate o situación neutral: usamos posición vertical como desempate
        walls = game_state.get_walls()
        mid_y = walls.height // 2
        if my_pos[1] > mid_y:
            return 'distractor'
        else:
            return 'cleaner'

    # ======================
    #  LÓGICA PRINCIPAL
    # ======================

    def choose_action(self, game_state):
        """
        Decide la acción en cada turno.

        Flujo general:
        - Actualizar memoria de comida defendida.
        - Actualizar detección de bucles y cooldown.
        - Si no hay acciones legales (aparte de STOP), nos paramos.
        - Si estamos en endgame y la política lo indica → modo defensa absoluta.
        - Calcular rol (distractor / cleaner).
        - Calcular score de retorno.
        - Elegir target y modo ('return', 'entry', 'collect', 'distract').
        - Usar A* para ir hacia el target con coste personalizado.
        """
        # Actualizar memoria de comida defendida y estado de “atasco”
        self.update_defending_food_memory(game_state)
        self.update_stuck_state(game_state)
        self.tick_unstuck_cooldown()

        # Acciones legales, excluyendo STOP
        actions = game_state.get_legal_actions(self.index)
        actions = [action for action in actions if action != Directions.STOP]
        if not actions:
            return Directions.STOP

        # Modo de defensa de endgame (en este proyecto, lo controla la subclase)
        if hasattr(self, "should_endgame_defend") and self.should_endgame_defend(game_state):
            return self.endgame_defense_action(game_state)

        my_pos = game_state.get_agent_position(self.index)
        food_list = self.get_food(game_state).as_list()

        # Rol dinámico local
        role = self.decide_role(game_state)

        # Durante el cooldown, este agente no puede ser distractor
        if self.unstuck_cooldown > 0 and role == 'distractor':
            self.dbg(f"COOLDOWN activo ({self.unstuck_cooldown}) → forzando role=CLEANER")
            role = 'cleaner'

        self.current_role = role

        # Estado de persecución
        being_pursued = self.is_being_pursued(game_state)

        # Score de retorno dinámico
        return_score = self.compute_return_score(game_state, my_pos, food_list)
        threshold = getattr(self, "return_threshold", 0.6)

        # 1) Si estamos perseguidos o el score de retorno es alto → volver a casa
        if being_pursued or return_score >= threshold:
            target = self.get_safest_home_position(game_state, my_pos)
            mode = 'return'
            if being_pursued:
                self.dbg("DECISION: Being pursued → RETURN MODE")
            else:
                self.dbg(f"DECISION: return_score={return_score:.2f} >= {threshold} → RETURN MODE")

        # 2) Ataque normal con roles
        else:
            if role == 'distractor':
                target = self.get_distractor_target(game_state, my_pos)
                mode = 'distract'
                self.dbg("DECISION: ROLE = DISTRACTOR → DISTRACT MODE")
            else:
                # role == 'cleaner'
                if self.is_on_own_side(game_state, my_pos):
                    target = self.get_safest_entry_position(game_state, my_pos)
                    mode = 'entry'
                    self.dbg("DECISION: CLEANER + Own side → ENTRY MODE")
                else:
                    if food_list:
                        target = self.get_closest_safe_food(game_state, my_pos, food_list)
                        mode = 'collect'
                        self.dbg("DECISION: CLEANER + Enemy side → COLLECT MODE")
                    else:
                        self.dbg("DECISION: No food left → STOP")
                        return Directions.STOP

        self.dbg(f"Target ({mode.upper()}): {target}")

        # Si se detecta bucle en modo ofensivo, forzamos vuelta a casa
        if getattr(self, "stuck_mode", False) and mode in ('entry', 'collect', 'distract'):
            self.dbg("STUCK in offensive mode → switching to RETURN MODE")
            target = self.get_safest_home_position(game_state, my_pos)
            mode = 'return'
            self.reset_stuck_state()

        # A* hacia el target
        next_step = self.a_star_search_next_step(game_state, my_pos, target, mode)
        if next_step:
            action = self.get_action_from_path(my_pos, next_step)
            self.dbg(f"Next step: {next_step} → Action: {action}")
            return action
        else:
            self.dbg("No path found by A* → STOP")
            return Directions.STOP

    # ======================
    #  DETECTAR PERSECUCIÓN
    # ======================

    def is_being_pursued(self, game_state):
        """
        Devuelve True si hay un ghost enemigo no asustado a distancia <= 2.
        """
        my_pos = game_state.get_agent_position(self.index)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        active_enemies = [
            e for e in enemies
            if not e.is_pacman and e.get_position() is not None and e.scared_timer == 0
        ]

        for enemy in active_enemies:
            enemy_pos = enemy.get_position()
            distance = self.get_maze_distance(my_pos, enemy_pos)
            if distance <= 2:
                self.dbg(f"Being pursued: enemy at {enemy_pos} dist={distance}")
                return True
        return False

    # ======================
    #  COMIDA "SEGURA"
    # ======================

    def get_closest_safe_food(self, game_state, my_pos, food_list):
        """
        Selecciona una comida teniendo en cuenta:
        - Distancia desde el agente a la comida.
        - Distancia desde la comida a los ghosts (queremos que sea grande).
        - Distancia desde la comida al compañero (queremos separarnos).
        """
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        active_enemies = [
            e for e in enemies
            if not e.is_pacman and e.get_position() is not None and e.scared_timer == 0
        ]

        mate_pos = self.get_teammate_pos(game_state)

        def food_score(food):
            """
            Score bajo = mejor.
            Combina:
            - d_agent: distancia desde el agente a la comida (queremos pequeño).
            - min_de: distancia mínima desde la comida a los ghosts (queremos grande).
            - d_mate: distancia desde el compañero a la comida (queremos grande).
            """
            d_agent = self.get_maze_distance(my_pos, food)
            if active_enemies:
                d_enemies = [self.get_maze_distance(food, e.get_position()) for e in active_enemies]
                min_de = min(d_enemies)
            else:
                min_de = 999

            if mate_pos is not None:
                d_mate = self.get_maze_distance(mate_pos, food)
            else:
                d_mate = 999

            return d_agent - 2 * min_de - 0.5 * d_mate

        best_food = min(food_list, key=food_score)
        self.dbg(f"Chosen food: {best_food}")
        return best_food

    # ======================
    #  TARGET DEL DISTRACTOR
    # ======================

    def get_distractor_target(self, game_state, my_pos):
        """
        Target para el rol 'distractor'.

        Objetivo:
        - Mantener una distancia cómoda al defensor (kiteo).
        - Ser claramente el objetivo más cercano del defensor frente al compañero.
        - Evitar meterse en callejones si no es necesario.
        """
        defender, dist_def = self.get_defender_info(game_state, my_pos)
        if defender is None:
            # Si no hay defensor, no tiene sentido distraer
            return my_pos

        dpos = defender.get_position()
        dpx, dpy = int(dpos[0]), int(dpos[1])

        mate_pos = self.get_teammate_pos(game_state)
        walls = game_state.get_walls()

        IDEAL = 4  # distancia ideal al defensor

        # Candidatos alrededor del defensor a esa distancia aproximada
        candidate_positions = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            tx = dpx + dx * IDEAL
            ty = dpy + dy * IDEAL
            if (
                0 <= tx < walls.width
                and 0 <= ty < walls.height
                and not walls[int(tx)][int(ty)]
            ):
                candidate_positions.append((int(tx), int(ty)))

        # Si no hay candidatos "perfectos", nos quedamos cerca del defensor
        if not candidate_positions:
            return (dpx, dpy)

        def score(pos):
            """
            Score bajo = mejor para distraer.
            Queremos:
            - |d_me - IDEAL| pequeño (mantener distancia ideal al defensor).
            - d_curr pequeño (no alejarnos demasiado de la posición actual).
            - d_mate grande (que el defensor tenga más cerca a este agente que al compañero).
            """
            d_me = self.get_maze_distance(pos, (dpx, dpy))
            d_curr = self.get_maze_distance(my_pos, pos)

            if mate_pos is not None:
                d_mate = self.get_maze_distance(mate_pos, (dpx, dpy))
            else:
                d_mate = 999

            return abs(d_me - IDEAL) * 5 + d_curr - d_mate * 3

        best = min(candidate_positions, key=score)
        self.dbg(f"DISTRACTOR target: {best}")
        return best

    # ======================
    #  FRONTERA / ENTRY
    # ======================

    def get_enemy_boundary_positions(self, game_state):
        """
        Devuelve la columna de frontera del lado enemigo
        (la casilla inmediatamente al otro lado de nuestra frontera).
        """
        walls = game_state.get_walls()
        mid_x = walls.width // 2
        if self.red:
            enemy_x = mid_x
        else:
            enemy_x = mid_x - 1
        return [(enemy_x, y) for y in range(walls.height) if not walls[enemy_x][y]]

    def get_safest_home_position(self, game_state, my_pos):
        """
        Para volver a casa:
        se elige la posición de frontera de nuestro lado que esté
        lo más lejos posible de los ghosts enemigos y razonablemente cerca del agente.
        """
        home_positions = self.get_home_boundary_positions(game_state)
        if not home_positions:
            return self.start

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        active_enemies = [
            e for e in enemies
            if not e.is_pacman and e.get_position() is not None and e.scared_timer == 0
        ]

        def home_score(pos):
            d_me = self.get_maze_distance(my_pos, pos)
            if active_enemies:
                d_enemies = [self.get_maze_distance(pos, e.get_position()) for e in active_enemies]
                min_de = min(d_enemies)
            else:
                min_de = 999
            return d_me - 2 * min_de

        best_home = min(home_positions, key=home_score)
        self.dbg(f"Safest home: {best_home}")
        return best_home

    def get_safest_entry_position(self, game_state, my_pos):
        """
        Determina el punto de entrada al lado enemigo (frontera enemiga).

        Criterios:
        - Estar lo más lejos posible de los ghosts defensores.
        - No estar demasiado lejos del agente.
        - Estar razonablemente separado del compañero.
        """
        entry_positions = self.get_enemy_boundary_positions(game_state)
        if not entry_positions:
            return my_pos

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        defenders = [
            e for e in enemies
            if not e.is_pacman and e.get_position() is not None and e.scared_timer == 0
        ]

        mate_pos = self.get_teammate_pos(game_state)

        def entry_score(pos):
            d_me = self.get_maze_distance(my_pos, pos)
            if defenders:
                d_def = [self.get_maze_distance(pos, e.get_position()) for e in defenders]
                min_dd = min(d_def)
            else:
                min_dd = 999

            sep_term = 0
            if mate_pos is not None:
                sep = abs(pos[0] - mate_pos[0]) + abs(pos[1] - mate_pos[1])
                sep_term = -0.5 * sep

            return d_me - 3 * min_dd + sep_term

        best_entry = min(entry_positions, key=entry_score)
        self.dbg(f"Safest entry: {best_entry}")
        return best_entry

    # ======================
    #  A* + COSTE / HEURÍSTICA
    # ======================

    def a_star_search_next_step(self, game_state, start, goal, mode):
        """
        A* estándar que devuelve solo el siguiente paso desde 'start' hacia 'goal',
        usando get_cost y heuristic personalizadas para cada 'mode'.
        """
        walls = game_state.get_walls()
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))

        open_set = util.PriorityQueue()
        open_set.push(start, 0)
        came_from = {}
        cost_so_far = {start: 0}

        nodes_expanded = 0
        MAX_NODES = 5000
        start_time_search = time.time()
        TIME_LIMIT_SEARCH = 0.15

        current = start

        while not open_set.is_empty():
            current = open_set.pop()
            nodes_expanded += 1

            if nodes_expanded > MAX_NODES:
                self.dbg("A*: node limit reached")
                return None
            if time.time() - start_time_search > TIME_LIMIT_SEARCH:
                self.dbg("A*: time limit reached")
                return None
            if current == goal:
                break

            for next_pos in self.get_successors(current, walls):
                new_cost = cost_so_far[current] + self.get_cost(game_state, next_pos, mode)
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(next_pos, goal, mode)
                    open_set.push(next_pos, priority)
                    came_from[next_pos] = current

        if current != goal and goal not in came_from:
            self.dbg("A*: goal not reached")
            return None

        # Reconstruir camino desde goal hasta start
        path = [goal]
        current_step = goal
        steps = 0
        MAX_PATH_STEPS = 1000

        while current_step != start and steps < MAX_PATH_STEPS:
            current_step = came_from.get(current_step, start)
            path.append(current_step)
            steps += 1

        if current_step != start:
            self.dbg("A*: failed to reconstruct full path")
            return None

        path.reverse()
        if len(path) > 1:
            return path[1]
        else:
            return None

    def get_cost(self, game_state, position, mode):
        """
        Función de coste para A*.

        Incluye:
        - Coste base (1).
        - Penalización por estar cerca de ghosts enemigos (solo en lado enemigo).
        - Repulsión entre compañeros (para no ir pegados).
        - En modo 'return', favorece avanzar hacia la frontera de casa.
        """
        cost = 1

        # Penalización por enemigos en el lado enemigo
        if not self.is_on_own_side(game_state, position):
            enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
            active_enemies = [
                e for e in enemies
                if not e.is_pacman and e.get_position() is not None and e.scared_timer == 0
            ]

            for enemy in active_enemies:
                enemy_pos = enemy.get_position()
                distance = self.get_maze_distance(position, enemy_pos)
                if distance <= 1:
                    return math.inf   # Nunca pisar al lado del ghost
                elif distance == 2:
                    cost += 50
                elif 3 <= distance <= 4:
                    cost += 10

        # Repulsión entre compañeros (evitar ir pegados) si no estamos en modo stuck
        mate_pos = self.get_teammate_pos(game_state)
        if mate_pos is not None and not self.stuck_mode:
            d_tm = abs(position[0] - mate_pos[0]) + abs(position[1] - mate_pos[1])
            if d_tm <= 1:
                cost += 30
            elif d_tm == 2:
                cost += 15
            elif d_tm == 3:
                cost += 5

        # En modo 'return' favorecemos estar más cerca de casa
        if mode == 'return':
            home_positions = self.get_home_boundary_positions(game_state)
            if home_positions:
                min_home_distance = min(
                    self.get_maze_distance(position, pos) for pos in home_positions
                )
                cost += min_home_distance * 0.2

        return cost

    def get_successors(self, position, walls):
        """
        Devuelve las casillas vecinas alcanzables (no paredes) desde 'position'.
        """
        successors = set()
        x, y = position
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < walls.width and 0 <= ny < walls.height and not walls[nx][ny]:
                successors.add((nx, ny))
        return list(successors)

    def heuristic(self, a, b, mode):
        """
        Heurística de A*: distancia Manhattan.

        En modo 'return' se penaliza un poco más la distancia (factor 1.2)
        para priorizar la vuelta rápida a casa.
        """
        (x1, y1) = a
        (x2, y2) = b
        distance = abs(x1 - x2) + abs(y1 - y2)
        if mode == 'return':
            return distance * 1.2
        else:
            return distance

    # ======================
    #  RETURN SCORE
    # ======================

    def get_nearest_active_enemy_distance(self, game_state, my_pos):
        """
        Distancia al ghost enemigo no asustado más cercano.
        Si no hay ninguno visible, devuelve None.
        """
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        active_enemies = [
            e for e in enemies
            if not e.is_pacman and e.get_position() is not None and e.scared_timer == 0
        ]

        if not active_enemies:
            return None

        dists = [self.get_maze_distance(my_pos, e.get_position()) for e in active_enemies]
        return min(dists)

    def get_nearest_food_distance(self, game_state, my_pos, food_list):
        """
        Distancia a la comida más cercana. Si no hay comida, devuelve None.
        """
        if not food_list:
            return None
        dists = [self.get_maze_distance(my_pos, f) for f in food_list]
        return min(dists)

    def compute_return_score(self, game_state, my_pos, food_list):
        """
        Calcula un score de "volver a casa" en función de:
        - Comida que lleva el agente.
        - Cercanía del enemigo.
        - Cercanía de la siguiente comida.

        Score alto implica más probabilidad de que el agente quiera volver.
        """
        my_state = game_state.get_agent_state(self.index)
        carrying_food = my_state.num_carrying

        # Normalizamos comida sobre max_carry (por ejemplo, 3/6 = 0.5)
        load_factor = 0.0
        if self.max_carry > 0:
            load_factor = min(1.0, carrying_food / float(self.max_carry))

        # Enemigo más cercano
        enemy_dist = self.get_nearest_active_enemy_distance(game_state, my_pos)
        if enemy_dist is None:
            enemy_term = 0.0
        else:
            enemy_term = max(0.0, (6.0 - enemy_dist) / 6.0)

        # Comida más cercana
        food_dist = self.get_nearest_food_distance(game_state, my_pos, food_list)
        if food_dist is None:
            food_term = 0.0
        else:
            food_term = 1.0 / (1.0 + food_dist)

        # Combinación lineal de factores
        return_score = 0.5 * load_factor + 0.4 * enemy_term - 0.3 * food_term

        self.dbg(
            f"RETURN_SCORE | pos={my_pos} | carry={carrying_food} "
            f"| load_factor={load_factor:.2f} | enemy_dist={enemy_dist} enemy_term={enemy_term:.2f} "
            f"| food_dist={food_dist} food_term={food_term:.2f} "
            f"| score={return_score:.2f}"
        )

        return return_score

    # ======================
    #  UTILIDADES DE ACCIÓN
    # ======================

    def get_action_from_path(self, current_pos, next_pos):
        """
        Convierte dos posiciones consecutivas en una acción de Directions.
        """
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]
        if dx == 1:
            return Directions.EAST
        elif dx == -1:
            return Directions.WEST
        elif dy == 1:
            return Directions.NORTH
        elif dy == -1:
            return Directions.SOUTH
        else:
            return Directions.STOP

    def get_maze_distance(self, pos1, pos2):
        """
        Envoltorio sobre self.distancer.get_distance.

        Devuelve math.inf si algo falla o si no se puede calcular la distancia.
        """
        try:
            d = self.distancer.get_distance(pos1, pos2)
            if d is None:
                return math.inf
            return d
        except Exception:
            return math.inf


class DualOffensiveAgent(OffensiveAStarAgent):
    """
    Agente ofensivo coordinado.

    Los dos agentes del equipo son de esta clase. Añade una capa de coordinación global
    por encima de OffensiveAStarAgent:

    - Decide para todo el equipo quién es DISTRACTOR y quién es CLEANER.
    - El agente más cercano al defensor se asigna como DISTRACTOR.
    - El otro se comporta como CLEANER.
    - También decide cuándo activar el modo de defensa absoluta de endgame.
    """

    # ======================
    #  INICIALIZACIÓN
    # ======================

    def register_initial_state(self, game_state):
        """
        Inicialización de la subclase: reaprovecha la lógica base y ajusta parámetros.
        """
        super().register_initial_state(game_state)
        # Parámetros intermedios entre agente agresivo y conservador
        self.max_carry = 6
        self.return_threshold = 0.45
        self.debug = False

    # ======================
    #  COORDINACIÓN GLOBAL
    # ======================

    def _get_team_indices(self, game_state):
        """
        Compatibilidad con getTeam / get_team según versión de capture_agents.

        Sobrescribimos para usarlo también desde esta subclase.
        """
        fn = getattr(self, "getTeam", None) or getattr(self, "get_team", None)
        if fn is None:
            return [self.index]
        return fn(game_state)

    def get_defender_and_dist_for_index(self, game_state, agent_index):
        """
        Devuelve (defensor_más_cercano, distancia) vistos desde el agente 'agent_index'.

        Solo cuenta enemies ghosts no asustados con posición conocida.
        """
        my_pos = game_state.get_agent_position(agent_index)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        defenders = [
            e for e in enemies
            if not e.is_pacman and e.get_position() is not None and e.scared_timer == 0
        ]
        if not defenders:
            return None, None

        defender = min(defenders, key=lambda d: self.get_maze_distance(my_pos, d.get_position()))
        dist = self.get_maze_distance(my_pos, defender.get_position())
        return defender, dist

    def decide_global_roles(self, game_state):
        """
        Decide los roles para todo el equipo.

        Lógica:
        - Si no hay defensor visible → ambos cleaners.
        - Si hay defensor:
            * El agente más cercano al defensor → DISTRACTOR.
            * El otro → CLEANER.

        Devuelve un diccionario {indice_agente: 'distractor'/'cleaner'}.
        """
        team = self._get_team_indices(game_state)
        roles = {idx: 'cleaner' for idx in team}

        if len(team) < 2:
            return roles

        any_idx = team[0]
        defender, _ = self.get_defender_and_dist_for_index(game_state, any_idx)

        if defender is None:
            # Sin defensor visible, todos cleaners
            return roles

        # Distancia al defensor para cada agente
        dists = {}
        for idx in team:
            _, dist = self.get_defender_and_dist_for_index(game_state, idx)
            if dist is None:
                dist = 999
            dists[idx] = dist

        # El más cercano al defensor será el distractor
        distractor_idx = min(dists, key=dists.get)
        for idx in team:
            roles[idx] = 'distractor' if idx == distractor_idx else 'cleaner'

        return roles

    # ======================
    #  LÓGICA PRINCIPAL
    # ======================

    def choose_action(self, game_state):
        """
        Versión coordinada de choose_action:

        - Actualiza memoria de comida defendida y estado de atascos.
        - Si está activado el modo endgame_defend → defensa absoluta.
        - Calcula los roles globales para los dos agentes.
        - Usa la lógica base de OffensiveAStarAgent,
          pero empleando el rol global en lugar del rol local.
        """
        # Actualizar memorias y estado de bucle
        self.update_defending_food_memory(game_state)
        self.update_stuck_state(game_state)
        self.tick_unstuck_cooldown()

        # Acciones legales sin STOP
        actions = game_state.get_legal_actions(self.index)
        actions = [action for action in actions if action != Directions.STOP]
        if not actions:
            return Directions.STOP

        # Modo de defensa de endgame
        if self.should_endgame_defend(game_state):
            if self.debug:
                print(f"[COORD-{self.index}] ENDGAME DEFENSE MODE ACTIVATED")
            return self.endgame_defense_action(game_state)

        my_pos = game_state.get_agent_position(self.index)
        food_list = self.get_food(game_state).as_list()

        # Roles globales para el equipo
        roles = self.decide_global_roles(game_state)
        role = roles.get(self.index, 'cleaner')

        # Si el agente está en cooldown, no se permite el rol de distractor
        if self.unstuck_cooldown > 0 and role == 'distractor':
            if self.debug:
                print(f"[COORD-{self.index}] COOLDOWN={self.unstuck_cooldown} → role override DISTRACTOR→CLEANER")
            role = 'cleaner'

        self.current_role = role

        # Estado de persecución y score de retorno
        being_pursued = self.is_being_pursued(game_state)
        return_score = self.compute_return_score(game_state, my_pos, food_list)
        threshold = getattr(self, "return_threshold", 0.45)

        # 1) Prioridad: volver a casa si conviene
        if being_pursued or return_score >= threshold:
            target = self.get_safest_home_position(game_state, my_pos)
            mode = 'return'
            if self.debug:
                if being_pursued:
                    print(f"[COORD-{self.index}] DECISION: Being pursued → RETURN MODE")
                else:
                    print(f"[COORD-{self.index}] DECISION: return_score={return_score:.2f} >= {threshold} → RETURN MODE")

        # 2) Modo normal: DISTRACTOR vs CLEANER
        else:
            if role == 'distractor':
                target = self.get_distractor_target(game_state, my_pos)
                mode = 'distract'
                if self.debug:
                    print(f"[COORD-{self.index}] ROLE = DISTRACTOR → DISTRACT MODE")
            else:
                # role == 'cleaner'
                if self.is_on_own_side(game_state, my_pos):
                    target = self.get_safest_entry_position(game_state, my_pos)
                    mode = 'entry'
                    if self.debug:
                        print(f"[COORD-{self.index}] ROLE = CLEANER + Own side → ENTRY MODE")
                else:
                    if food_list:
                        target = self.get_closest_safe_food(game_state, my_pos, food_list)
                        mode = 'collect'
                        if self.debug:
                            print(f"[COORD-{self.index}] ROLE = CLEANER + Enemy side → COLLECT MODE")
                    else:
                        if self.debug:
                            print(f"[COORD-{self.index}] No food left → STOP")
                        return Directions.STOP

        if self.debug:
            print(f"[COORD-{self.index}] Target ({mode.upper()}): {target}")

        # Si detectamos bucle en modo ofensivo, forzamos retorno
        if getattr(self, "stuck_mode", False) and mode in ('entry', 'collect', 'distract'):
            print(f"[COORD-{self.index}] STUCK in {mode} → FORCED RETURN")
            target = self.get_safest_home_position(game_state, my_pos)
            mode = 'return'
            self.reset_stuck_state()

        # A* hacia el target
        next_step = self.a_star_search_next_step(game_state, my_pos, target, mode)
        if next_step:
            action = self.get_action_from_path(my_pos, next_step)
            if self.debug:
                print(f"[COORD-{self.index}] Next step: {next_step} → Action: {action}")
            return action
        else:
            if self.debug:
                print(f"[COORD-{self.index}] No path found by A* → STOP")
            return Directions.STOP

    # ======================
    #  POLÍTICA DE ENDGAME
    # ======================

    def should_endgame_defend(self, game_state):
        """
        Decide si se activa el modo de defensa absoluta de endgame.

        Criterios:
        - Queda muy poca comida propia.
        - Queda poco tiempo y no vamos perdiendo.
        - Vamos claramente ganando y queda tiempo medio.
        """
        food_def = self.get_food_you_are_defending(game_state).as_list()
        remaining = len(food_def)

        # Tiempo restante en acciones (no en frames)
        time_left = game_state.data.timeleft / 4.0
        score = self.get_score(game_state)

        # 1) Pocas comidas que defender → prioridad máxima a defender
        if remaining <= 3:
            return True

        # 2) Poco tiempo y no vamos perdiendo → cerrar partida
        if time_left < 80 and score >= 0:
            return True

        # 3) Ventaja clara y tiempo medio → preferimos controlar el marcador
        if score > 5 and time_left < 200:
            return True

        # Si vamos perdiendo, evitamos turtlear
        return False
