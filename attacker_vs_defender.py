import pygame
import sys
from dataclasses import dataclass, field
from typing import ClassVar, Tuple, Iterable, Optional, Dict, List
from enum import Enum
import copy
from datetime import datetime
import argparse
import random
import numpy as np
import importlib
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from deap import base, creator, tools, algorithms

class UnitType(Enum):
    AI = 0
    Virus = 1
    Tech = 2
    Firewall = 3
    Program = 4

class Player(Enum):
    Attacker = 0
    Defender = 1
    def next(self) -> 'Player':
        return Player.Defender if self is Player.Attacker else Player.Attacker

class GameType(Enum):
    AttackerVsComp = 1
    CompVsDefender = 2
    def __str__(self):
        if self == GameType.AttackerVsComp:
            return "Attacker vs Comp"
        elif self == GameType.CompVsDefender:
            return "Comp vs Defender"
        else:
            return super().__str__()

@dataclass(slots=True)
class Unit:
    player: Player = Player.Attacker
    type: UnitType = UnitType.Program
    health: int = 5
    damage_table: ClassVar[list[list[int]]] = [
        [3, 3, 3, 1, 3],  # AI
        [5, 1, 4, 1, 4],  # Virus
        [1, 4, 1, 1, 1],  # Tech
        [1, 1, 1, 1, 1],  # Firewall
        [3, 3, 3, 1, 3],  # Program
    ]
    repair_table: ClassVar[list[list[int]]] = [
        [0, 1, 1, 0, 0],  # AI
        [0, 0, 0, 0, 0],  # Virus
        [3, 0, 0, 3, 3],  # Tech
        [0, 0, 0, 0, 0],  # Firewall
        [0, 0, 0, 0, 0],  # Program
    ]

    def is_alive(self) -> bool:
        return self.health > 0

    def mod_health(self, health_delta: int):
        self.health += health_delta
        if self.health < 0:
            self.health = 0
        elif self.health > 5:
            self.health = 5

    def to_string(self) -> str:
        p = self.player.name.lower()[0]
        t = self.type.name.upper()[0]
        return f"{p}{t}{self.health}"

    def __str__(self) -> str:
        return self.to_string()

    def damage_amount(self, target: 'Unit') -> int:
        amount = self.damage_table[self.type.value][target.type.value]
        if target.health - amount < 0:
            return target.health
        return amount

    def repair_amount(self, target: 'Unit') -> int:
        amount = self.repair_table[self.type.value][target.type.value]
        if target.health + amount > 5:
            return 5 - target.health
        return amount

@dataclass(slots=True)
class Coord:
    row: int = 0
    col: int = 0

    def col_string(self) -> str:
        coord_char = '?'
        if self.col < 5:
            coord_char = "01234"[self.col]
        return str(coord_char)

    def row_string(self) -> str:
        coord_char = '?'
        if self.row < 5:
            coord_char = "ABCDE"[self.row]
        return str(coord_char)

    def to_string(self) -> str:
        return self.row_string() + self.col_string()

    def __str__(self) -> str:
        return self.to_string()

    def clone(self) -> 'Coord':
        return copy.copy(self)

    def iter_range(self, dist: int) -> Iterable['Coord']:
        for row in range(self.row - dist, self.row + 1 + dist):
            for col in range(self.col - dist, self.col + 1 + dist):
                yield Coord(row, col)

    def iter_adjacent(self) -> Iterable['Coord']:
        yield Coord(self.row - 1, self.col)
        yield Coord(self.row, self.col - 1)
        yield Coord(self.row + 1, self.col)
        yield Coord(self.row, self.col + 1)

    @classmethod
    def from_string(cls, s: str) -> Optional['Coord']:
        s = s.strip()
        for sep in " ,.:;-_":
            s = s.replace(sep, "")
        if len(s) == 2:
            coord = Coord()
            coord.row = "ABCDE".find(s[0:1].upper())
            coord.col = "01234".find(s[1:2].lower())
            return coord
        else:
            return None

@dataclass(slots=True)
class CoordPair:
    src: Coord = field(default_factory=Coord)
    dst: Coord = field(default_factory=Coord)

    def to_string(self) -> str:
        return self.src.to_string() + " " + self.dst.to_string()

    def __str__(self) -> str:
        return self.to_string()

    def clone(self) -> 'CoordPair':
        return copy.copy(self)

    def iter_rectangle(self) -> Iterable[Coord]:
        for row in range(self.src.row, self.dst.row + 1):
            for col in range(self.src.col, self.dst.col + 1):
                yield Coord(row, col)

    @classmethod
    def from_string(cls, s: str) -> Optional['CoordPair']:
        s = s.strip()
        for sep in " ,.:;-_":
            s = s.replace(sep, "")
        if len(s) == 4:
            coords = CoordPair()
            coords.src.row = "ABCDE".find(s[0:1].upper())
            coords.src.col = "01234".find(s[1:2].lower())
            coords.dst.row = "ABCDE".find(s[2:3].upper())
            coords.dst.col = "01234".find(s[3:4].lower())
            return coords
        else:
            return None

    @classmethod
    def from_dim(cls, dim: int) -> 'CoordPair':
        return CoordPair(Coord(0, 0), Coord(dim - 1, dim - 1))

@dataclass(slots=True)
class Options:
    dim: int = 5
    max_depth: Optional[int] = 4
    min_depth: Optional[int] = 2
    max_time: Optional[float] = 5.0
    game_type: GameType = GameType.CompVsDefender
    alpha_beta: bool = True
    max_turns: Optional[int] = 150
    heuristic: Optional[int] = 0

@dataclass(slots=True)
class Stats:
    evaluations_per_depth: Dict[int, int] = field(default_factory=dict)
    total_seconds: float = 0.0

@dataclass(slots=True)
class Game:
    board: List[List[Optional[Unit]]] = field(default_factory=list)
    next_player: Player = Player.Attacker
    turns_played: int = 0
    options: Options = field(default_factory=Options)
    stats: Stats = field(default_factory=Stats)
    _attacker_has_ai: bool = True
    _defender_has_ai: bool = True
    h_score: int = -2000000000
    states_evaluated: int = 0
    selected_coord: Optional[Coord] = None
    possible_moves: List[Coord] = field(default_factory=list)

    def __post_init__(self):
        dim = self.options.dim
        self.board = [[None for _ in range(dim)] for _ in range(dim)]
        md = dim - 1
        self.set(Coord(0, 0), Unit(player=Player.Defender, type=UnitType.AI))
        self.set(Coord(1, 0), Unit(player=Player.Defender, type=UnitType.Tech))
        self.set(Coord(0, 1), Unit(player=Player.Defender, type=UnitType.Tech))
        self.set(Coord(2, 0), Unit(player=Player.Defender, type=UnitType.Firewall))
        self.set(Coord(0, 2), Unit(player=Player.Defender, type=UnitType.Firewall))
        self.set(Coord(1, 1), Unit(player=Player.Defender, type=UnitType.Program))
        self.set(Coord(md, md), Unit(player=Player.Attacker, type=UnitType.AI))
        self.set(Coord(md - 1, md), Unit(player=Player.Attacker, type=UnitType.Virus))
        self.set(Coord(md, md - 1), Unit(player=Player.Attacker, type=UnitType.Virus))
        self.set(Coord(md - 2, md), Unit(player=Player.Attacker, type=UnitType.Program))
        self.set(Coord(md, md - 2), Unit(player=Player.Attacker, type=UnitType.Program))
        self.set(Coord(md - 1, md - 1), Unit(player=Player.Attacker, type=UnitType.Firewall))

    def clone(self) -> 'Game':
        new = copy.copy(self)
        new.board = copy.deepcopy(self.board)
        return new

    def is_empty(self, coord: Coord) -> bool:
        return self.board[coord.row][coord.col] is None

    def get(self, coord: Coord) -> Optional[Unit]:
        if self.is_valid_coord(coord):
            return self.board[coord.row][coord.col]
        else:
            return None

    def set(self, coord: Coord, unit: Optional[Unit]):
        if self.is_valid_coord(coord):
            self.board[coord.row][coord.col] = unit

    def remove_dead(self, coord: Coord):
        unit = self.get(coord)
        if unit is not None and not unit.is_alive():
            self.set(coord, None)
            if unit.type == UnitType.AI:
                if unit.player == Player.Attacker:
                    self._attacker_has_ai = False
                else:
                    self._defender_has_ai = False

    def mod_health(self, coord: Coord, health_delta: int):
        target = self.get(coord)
        if target is not None:
            target.mod_health(health_delta)
            self.remove_dead(coord)

    def is_valid_move(self, coords: CoordPair, bot) -> bool:
        if not self.is_valid_coord(coords.src) or not self.is_valid_coord(coords.dst):
            return False

        if self.get(coords.src) is None or self.get(coords.src).player != self.next_player:
            return False

        if not self.is_adjacent(coords):
            return False

        if self.is_in_combat(coords) and ((self.board[coords.src.row][coords.src.col].type == UnitType.AI) or (self.board[coords.src.row][coords.src.col].type == UnitType.Firewall) or (self.board[coords.src.row][coords.src.col].type == UnitType.Program)):
            return False

        if (self.board[coords.src.row][coords.src.col].type == UnitType.Tech or
            self.board[coords.src.row][coords.src.col].type == UnitType.Virus):
            return True

        if (self.board[coords.src.row][coords.src.col].player == Player.Attacker and coords.src.col < coords.dst.col) or (self.board[coords.src.row][coords.src.col].player == Player.Attacker and coords.src.row < coords.dst.row):
            
            return False

        if (self.board[coords.src.row][coords.src.col].player == Player.Defender and coords.src.col > coords.dst.col) or (self.board[coords.src.row][coords.src.col].player == Player.Defender and coords.src.row > coords.dst.row):

            return False
        return True

    def is_valid_action(self, coords: CoordPair) -> bool:
        unit = self.get(coords.src)
        if unit is None or unit.player != self.next_player:
            return False

        elif not self.is_valid_coord(coords.src) or not self.is_valid_coord(coords.dst):
            return False

        else:
            return True

    def is_in_combat(self, coords: CoordPair) -> bool:
        for i in coords.src.iter_adjacent():
            adjacent_unit = self.get(i)
            if (adjacent_unit is not None) and (adjacent_unit.player != self.get(coords.src).player):
                return True
        return False

    def perform_move(self, coords: CoordPair) -> Tuple[bool, str, int]:
        if self.is_empty(coords.dst) and self.is_valid_move(coords, False):
            self.set(coords.dst, self.get(coords.src))
            self.set(coords.src, None)
            return (True, "", 0)

        elif not self.is_empty(coords.dst) and self.is_valid_action(coords):
            success, message, actionType = self.action(coords)
            if success:
                return (True, message, actionType)
            else:
                return (False, message, -1)
        else:
            return (False, "", -1)

    def action(self, coords: CoordPair) -> Tuple[bool, str, int]:
        target = self.get(coords.dst)
        source = self.get(coords.src)
        if coords.dst == coords.src:
            self.self_destruct(coords)
            return (True, f"{target.to_string()} self-destructed", 1)
        elif self.is_adjacent(coords):
            if self.is_ally(coords.dst):
                if target.health == 5:
                    return (False, f"{target.to_string()} already has max health", -1)
                else:
                    if self.repair(coords) == 0:
                        return (False, f"{source.to_string()} cannot repair {target.to_string()}", -1)
                    return (True, f"{target.to_string()} was repaired by {source.to_string()}", 2)
            else:
                self.attack(coords)
                return (True, f"{target.to_string()} was attacked by {source.to_string()}", 3)
        else:
            return (False, "", -1)

    def self_destruct(self, coords: CoordPair):
        unit = self.get(coords.dst)
        for coord in coords.dst.iter_range(1):
            if self.get(coord) is not None:
                self.get(coord).mod_health(-2)
                self.remove_dead(coord)
            else:
                continue
        unit.mod_health(-unit.health)
        self.remove_dead(coords.dst)

    def repair(self, coords: CoordPair) -> int:
        source = self.get(coords.src)
        target = self.get(coords.dst)
        amount = source.repair_amount(target)
        target.mod_health(amount)
        return amount

    def attack(self, coords: CoordPair):
        source = self.get(coords.src)
        target = self.get(coords.dst)
        target.mod_health(-source.damage_amount(target))
        source.mod_health(-target.damage_amount(source))
        self.remove_dead(coords.dst)
        self.remove_dead(coords.src)

    def is_adjacent(self, coords: CoordPair) -> bool:
        for coord in coords.src.iter_adjacent():
            if coord == coords.dst:
                return True
        return False

    def is_ally(self, target: Coord) -> bool:
        for coord, unit in self.player_units(self.next_player):
            if coord == target:
                return True
        return False

    def next_turn(self):
        self.next_player = self.next_player.next()
        self.turns_played += 1

    def to_string(self) -> str:
        dim = self.options.dim
        output = ""
        output += f"Next player: {self.next_player.name}\n"
        output += f"Turns played: {self.turns_played}\n"
        coord = Coord()
        output += "\n   "
        for col in range(dim):
            coord.col = col
            label = coord.col_string()
            output += f"{label:^3} "
        output += "\n"
        for row in range(dim):
            coord.row = row
            label = coord.row_string()
            output += f"{label}: "
            for col in range(dim):
                coord.col = col
                unit = self.get(coord)
                if unit is None:
                    output += " .  "
                else:
                    output += f"{str(unit):^3} "
            output += "\n"
        return output

    def __str__(self) -> str:
        return self.to_string()

    def is_valid_coord(self, coord: Coord) -> bool:
        dim = self.options.dim
        return 0 <= coord.row < dim and 0 <= coord.col < dim

    def human_turn(self):
        if self.selected_coord is not None:
            selected_unit = self.get(self.selected_coord)
            if selected_unit is None or selected_unit.player != self.next_player:
                self.selected_coord = None
                self.possible_moves = []
                return

            self.possible_moves = [
                move.dst for move in self.generate_moves() if move.src == self.selected_coord
            ]
        else:
            self.possible_moves = []

        self.next_turn()
        print("\n" + str(self))

    def computer_turn(self) -> Optional[CoordPair]:
        best_sequence = self.optimize_move_sequence()
        best_move = best_sequence[0]
        if isinstance(best_move, CoordPair):
            success, result, actionType = self.perform_move(best_move)
            if success:
                print(f"Computer {self.next_player.name}: {result if result else f'move from {best_move.src.to_string()} to {best_move.dst.to_string()}'}")
                self.next_turn()
                print("\n" + str(self))
                move = (best_move, actionType)
                return move
        return None

    def suggest_move(self) -> Optional[CoordPair]:
        start_time = datetime.now()
        alpha_beta = self.options.alpha_beta
        maxPlayer = True
        alpha = -2000000000
        beta = 2000000000
        depth = self.depth()

        for i in range(depth):
            self.stats.evaluations_per_depth[i + 1] = 0

        score, move = self.minimax(depth, maxPlayer, alpha_beta, alpha, beta)
        self.h_score = score

        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        self.stats.total_seconds = elapsed_seconds

        print(f"Evals per depth: ", end='')
        for k in sorted(self.stats.evaluations_per_depth.keys()):
            print(f"{depth}={self.stats.evaluations_per_depth[k]} ", end='')
            depth -= 1
        print()
        self.states_evaluated = sum(self.stats.evaluations_per_depth.values())
        return move

    def depth(self):
        depth = 0
        if self.turns_played < self.options.max_turns * 0.5:
            depth = self.options.min_depth
        else:
            depth = self.options.max_depth
        return depth

    def minimax(self, depth, maxPlayer, alpha_beta, alpha, beta) -> Tuple[int, Optional[CoordPair]]:
        if depth == 0 or self.is_finished():
            self.stats.evaluations_per_depth[depth] = self.stats.evaluations_per_depth.get(depth, 0) + 1
            return self.evaluate(), None

        moves = list(self.generate_moves())
        self.stats.evaluations_per_depth[depth] += len(moves)
        best_move = None

        if maxPlayer:
            best_score = -2000000000
            for move in moves:
                new_game = self.apply_move(move)
                if new_game is None:
                    continue
                score, _ = new_game.minimax(depth - 1, False, alpha_beta, alpha, beta)
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, score)
                if alpha_beta and beta <= alpha:
                    break
        else:
            best_score = 2000000000
            for move in moves:
                new_game = self.apply_move(move)
                if new_game is None:
                    continue
                score, _ = new_game.minimax(depth - 1, True, alpha_beta, alpha, beta)
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, score)
                if alpha_beta and beta <= alpha:
                    break

        return best_score, best_move

    def evaluate(self):
        if self.options.heuristic == 0:
            e = self.e0()
        elif self.options.heuristic == 1:
            e = self.e1()
        else:
            e = self.e2()
        return e

    def e0(self) -> int:
        VP1 = TP1 = FP1 = PP1 = AIP1 = 0
        VP2 = TP2 = FP2 = PP2 = AIP2 = 0

        for _, unit in self.player_units(self.next_player):
            if unit.type == UnitType.Virus:
                VP1 += 1
            elif unit.type == UnitType.Tech:
                TP1 += 1
            elif unit.type == UnitType.Firewall:
                FP1 += 1
            elif unit.type == UnitType.Program:
                PP1 += 1
            elif unit.type == UnitType.AI:
                AIP1 += 1

        for _, unit in self.player_units(self.next_player.next()):
            if unit.type == UnitType.Virus:
                VP2 += 1
            elif unit.type == UnitType.Tech:
                TP2 += 1
            elif unit.type == UnitType.Firewall:
                FP2 += 1
            elif unit.type == UnitType.Program:
                PP2 += 1
            elif unit.type == UnitType.AI:
                AIP2 += 1

        heuristic_value = (3 * (VP1 + TP1 + FP1 + PP1) + 9999 * AIP1) - (3 * (VP2 + TP2 + FP2 + PP2) + 9999 * AIP2)
        return heuristic_value

    def e1(self):
        weights = {
            UnitType.AI: 10000,
            UnitType.Virus: 3,
            UnitType.Tech: 1,
            UnitType.Firewall: 5,
            UnitType.Program: 1
        }

        value_p1 = 0
        value_p2 = 0

        for row in self.board:
            for cell in row:
                if cell:
                    unit_type = cell.type
                    player = cell.player

                    if player == Player.Attacker:
                        value_p1 += weights[unit_type]
                    else:
                        value_p2 += weights[unit_type]

        return value_p1 - value_p2

    def e2(self) -> int:
        score = 0
        for src, unit in self.player_units(self.next_player):
            for dst in src.iter_adjacent():
                target = self.get(dst)
                if target is None:
                    continue
                if not self.is_ally(dst):
                    score += (unit.health - target.health) + (unit.damage_amount(target) - target.damage_amount(unit))
        return score

    def random_move(self) -> CoordPair:
        move_candidates = list(self.generate_moves())
        random.shuffle(move_candidates)
        for move in move_candidates:
            if move.src != move.dst:
                return move
        return CoordPair()  # Return a default or valid CoordPair if no valid move is found

    def generate_moves(self) -> Iterable[CoordPair]:
        move = CoordPair()
        for src, unit in self.player_units(self.next_player):
            move.src = src
            for dst in src.iter_adjacent():
                move.dst = dst
                if self.is_valid_move(move, True) and self.is_empty(move.dst):
                    yield move.clone()
            move.dst = src
            if self.is_valid_action(move) and not self.is_empty(move.dst) and unit.type != UnitType.AI:
                yield move.clone()

            for dst in src.iter_adjacent():
                move.dst = dst
                if self.is_valid_action(move) and not self.is_empty(move.dst):
                    yield move.clone()

    def apply_move(self, move: CoordPair) -> Optional['Game']:
        new_game = self.clone()
        success, _, _ = new_game.perform_move(move)
        if success:
            return new_game
        else:
            return None

    def player_units(self, player: Player) -> Iterable[Tuple[Coord, Unit]]:
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None and unit.player == player:
                yield coord, unit

    def is_finished(self) -> bool:
        return self.has_winner() is not None

    def has_winner(self) -> Optional[Player]:
        if not self._attacker_has_ai:
            return Player.Defender
        if not self._defender_has_ai:
            return Player.Attacker
        if self.options.max_turns is not None and self.turns_played >= self.options.max_turns:
            return Player.Defender
        return None

    def optimize_move_sequence(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("individual", tools.initRepeat, creator.Individual, self.random_move, n=5)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def evaluate(individual):
            return self.evaluate_sequence(individual),

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        population = toolbox.population(n=100)
        ngen = 10
        cxpb = 0.5
        mutpb = 0.2

        for gen in range(ngen):
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            population[:] = offspring

        best_ind = tools.selBest(population, 1)[0]
        return best_ind

    def fuzzy_logic(self):
        health = ctrl.Antecedent(np.arange(0, 11, 1), 'health')
        damage = ctrl.Consequent(np.arange(0, 11, 1), 'damage')

        health.automf(3)

        damage['low'] = fuzz.trimf(damage.universe, [0, 0, 5])
        damage['medium'] = fuzz.trimf(damage.universe, [0, 5, 10])
        damage['high'] = fuzz.trimf(damage.universe, [5, 10, 10])

        rule1 = ctrl.Rule(health['poor'], damage['low'])
        rule2 = ctrl.Rule(health['average'], damage['medium'])
        rule3 = ctrl.Rule(health['good'], damage['high'])

        damage_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
        damage_sim = ctrl.ControlSystemSimulation(damage_ctrl)

        return damage_sim

    def evaluate_move(self, unit):
        damage_sim = self.fuzzy_logic()
        fitness = 0

        if unit.type == UnitType.Virus:
            damage_sim.input['health'] = unit.health
            damage_sim.compute()
            predicted_damage = damage_sim.output['damage']
            fitness += predicted_damage

        return fitness

    def evaluate_sequence(self, sequence):
        temp_game = self.clone()
        fitness = 0
        for move in sequence:
            success, result, actionType = temp_game.perform_move(move)
            if success:
                unit = temp_game.get(move.dst)
                if unit:
                    fitness += self.evaluate_move(unit)
                if temp_game.has_winner() == self.next_player:
                    fitness += 1000
                if actionType == 3:
                    fitness += 10
                if actionType == 2:
                    fitness += 5
            else:
                fitness -= 10
            if temp_game.is_finished():
                break
        return fitness

class GameTrace:
    def __init__(self, filename):
        self.file = open(filename, 'w')

    def write_parameters(self, options):
        self.file.write(f"Timeout: {options.max_time} seconds\n")
        self.file.write(f"Max Turns: {options.max_turns}\n")
        self.file.write(f"Alpha-Beta: {'On' if options.alpha_beta else 'Off'}\n")
        self.file.write(f"Play Mode: {str(options.game_type)}\n")
        if options.game_type == "attacker":
            self.file.write("Attacker: H & Defender: AI\n")
        else:
            self.file.write("Attacker: AI & Defender: H\n")
        self.file.write(f"Heuristic used by AI: e{options.heuristic}\n")
        self.file.flush()

    def write_board(self, game):
        self.file.write(str(game) + "\n")
        self.file.flush()

    def write_action(self, game, turn, player, action, time):
        self.file.write(f"Turn #{turn} - {player.next().name}\n")
        coords, actionType = action
        string = ""
        match actionType:
            case 0:
                string = "move"
            case 1:
                string = "self-destruct"
            case 2:
                string = "repair"
            case 3:
                string = "attack"

        self.file.write(f"Action: {string} from {coords.src} to {coords.dst}\n")
        if time is not None:
            self.file.write(f"Time for this action: {time} sec\n")
            self.file.write(f"Heuristic score: {game.h_score}\n")
            self.file.write(f"Cumulative evals: {game.states_evaluated}\n")
            depth = game.depth()
            self.file.write("Cumulative evals by depth: ")
            for k in sorted(game.stats.evaluations_per_depth.keys()):
                self.file.write(f"{depth}={game.stats.evaluations_per_depth[k]} ")
                depth -= 1
            self.file.write("\n")
            depth = game.depth()
            self.file.write("Cumulative % evals by depth: ")
            for k in sorted(game.stats.evaluations_per_depth.keys()):
                percentage = (game.stats.evaluations_per_depth[k] / game.states_evaluated) * 100
                self.file.write(f"{depth}={percentage:.1f}% ")
                depth -= 1
            self.file.write("\n")

        self.file.write("\n")
        self.file.flush()

    def write_game_result(self, winner, turns_played):
        self.file.write(f"The winner of the game ({winner.name}) wins in {turns_played} turns\n")
        self.file.flush()

    def close(self):
        self.file.close()

def main():
    parser = argparse.ArgumentParser(
        prog='attacker_vs_defender',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max_time', type=float, help='maximum search time', default=5)
    parser.add_argument('--max_turns', type=int, help='maximum turns before end of game', default=100)
    parser.add_argument('--alpha_beta', type=bool, help='alpha-beta on/off', default=False)
    parser.add_argument('--game_type', type=str, choices=["attacker", "defender"], default="defender",
                        help='game type: attacker|defender')
    args = parser.parse_args()

    filename = f"gameTrace-{'true' if args.alpha_beta else 'false'}-{args.max_time}-{args.max_turns}.txt"
    trace = GameTrace(filename)

    if args.game_type == "attacker":
        game_type = GameType.AttackerVsComp
    else:
        game_type = GameType.CompVsDefender

    options = Options(game_type=game_type)
    options.max_time = args.max_time
    options.max_turns = args.max_turns
    options.alpha_beta = args.alpha_beta
    trace.write_parameters(options)

    game = Game(options=options)
    end = False
    stats = Stats()

    print()
    print(game)
    trace.write_board(game)

    while not end:
        end = game.is_finished()
        winner = game.has_winner()
        if winner is not None:
            trace.write_game_result(winner, game.turns_played)
            trace.close()
            print(f"{winner.name} wins!\n{game.turns_played} turns played")
            break
        if game.options.game_type == GameType.AttackerVsComp and game.next_player == Player.Attacker:
            game.human_turn()
        elif game.options.game_type == GameType.CompVsDefender and game.next_player == Player.Defender:
            game.human_turn()
        else:
            player = game.next_player
            move = game.computer_turn()
            trace.write_action(game, game.turns_played, game.next_player, move, game.stats.total_seconds)
            trace.write_board(game)
            if stats.total_seconds > options.max_time:
                trace.write_board(game)
                trace.write_game_result(game.next_player.next(), game.turns_played)
                trace.close()
                print(f"{game.next_player.name} took too long! Winner is {game.next_player.next().name}")
                break
            if move is not None:
                continue
            else:
                print("Computer doesn't know what to do!!!")
                exit(1)
    trace.close()

import pygame
import sys
import time
import os
from PIL import Image

WIDTH, HEIGHT = 800, 600  # Updated dimensions to cover a standard screen size
INFO_PANEL_WIDTH = 300
GRID_SIZE = 5
CELL_SIZE = (WIDTH - 200) // GRID_SIZE
FPS = 30

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BEIGE = (37, 65, 23)
CHOCOLATE_BROWN = (138, 154, 91)
YELLOW = (255, 255, 0)

def set_utf8_code_page():
    os.system("chcp 65001 > nul")

def draw_grid(screen):
    for x in range(0, WIDTH - INFO_PANEL_WIDTH, CELL_SIZE):
        for y in range(0, HEIGHT, CELL_SIZE):
            pygame.draw.rect(screen, BLACK, (x, y, CELL_SIZE, CELL_SIZE), 1)


def draw_units(screen, game):
    font = pygame.font.Font(None, 36)
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            unit = game.get(Coord(row, col))
            if unit:
                color = BEIGE if unit.player == Player.Defender else CHOCOLATE_BROWN
                pygame.draw.rect(screen, color, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                text = font.render(f'{unit.type.name[0]}{unit.health}', True, WHITE)
                text_rect = text.get_rect(center=(col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2))
                screen.blit(text, text_rect)
            if game.selected_coord and game.selected_coord.row == row and game.selected_coord.col == col:
                pygame.draw.rect(screen, YELLOW, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE), 3)
            if Coord(row, col) in game.possible_moves:
                pygame.draw.rect(screen, (0, 0, 255), (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE), 3)


def draw_stats(screen, game):
    font = pygame.font.Font(None, 24)
    stats_x = WIDTH - 200
    stats_y = 50
    spacing = 30

    max_turns_text = font.render(f"Max Turns: {game.options.max_turns}", True, BLACK)
    screen.blit(max_turns_text, (stats_x, stats_y))

    alpha_beta_text = font.render(f"Alpha-Beta: {'On' if game.options.alpha_beta else 'Off'}", True, BLACK)
    screen.blit(alpha_beta_text, (stats_x, stats_y + spacing))

    play_mode_text = font.render(f"Play Mode: {str(game.options.game_type)}", True, BLACK)
    screen.blit(play_mode_text, (stats_x, stats_y + 2 * spacing))

    attacker_defender_text = font.render(f"Attacker: H & Defender: AI", True, BLACK)
    screen.blit(attacker_defender_text, (stats_x, stats_y + 3 * spacing))

    heuristic_text = font.render(f"Heuristic used by AI: e{game.options.heuristic}", True, BLACK)
    screen.blit(heuristic_text, (stats_x, stats_y + 4 * spacing))

    next_player_text = font.render(f"Next player: {game.next_player.name}", True, BLACK)
    screen.blit(next_player_text, (stats_x, stats_y + 5 * spacing))

    turns_played_text = font.render(f"Turns played: {game.turns_played}", True, BLACK)
    screen.blit(turns_played_text, (stats_x, stats_y + 6 * spacing))

    if game.is_finished():
        winner_text = font.render(f"Winner: {game.has_winner().name}", True, BLACK)
        screen.blit(winner_text, (stats_x, stats_y + 7 * spacing))

def handle_click(game, grid_x, grid_y):
    coord = Coord(grid_y, grid_x)
    if game.selected_coord is None:
        if game.get(coord) and game.get(coord).player == game.next_player:
            game.selected_coord = coord
            game.possible_moves = [
                move.dst for move in game.generate_moves() if move.src == game.selected_coord
            ]
    else:
        if coord in game.possible_moves:
            mv = CoordPair(game.selected_coord, coord)
            success, result, actionType = game.perform_move(mv)
            if success:
                print(f"Player {game.next_player.name}: {result}")
                game.next_turn()
        game.selected_coord = None
        game.possible_moves = []

def draw_menu(screen):
    try:
        # Load the images
        attacker_image = pygame.image.load('Attacker.png')
        defender_image = pygame.image.load('defender.png')

        # Resize the images
        attacker_icon = pygame.transform.scale(attacker_image, (150, 200))
        defender_icon = pygame.transform.scale(defender_image, (150, 200))

        # Get rectangles for positioning
        rect_attacker = attacker_icon.get_rect(center=(1.9*WIDTH // 4 - 50, HEIGHT // 2))
        rect_defender = defender_icon.get_rect(center=(3.6* WIDTH // 4 + 50, HEIGHT // 2))

        # Fill the screen with a background color
        screen.fill((245, 245, 220))  # Beige color

        # Blit the images onto the screen
        screen.blit(attacker_icon, rect_attacker)
        screen.blit(defender_icon, rect_defender)

        # Draw title text
        font = pygame.font.Font(None, 50)
        title_text = font.render("Attacker Vs Defender", True, BLACK)
        title_rect = title_text.get_rect(center=(1.4*WIDTH // 2, 50))
        screen.blit(title_text, title_rect)

        # Draw labels
        font_small = pygame.font.Font(None, 35)
        attacker_label = font_small.render("Attacker", True, BLACK)
        defender_label = font_small.render("Defender", True, BLACK)

        attacker_label_rect = attacker_label.get_rect(center=(1.9*WIDTH // 4 - 50, HEIGHT // 2 + 100))
        defender_label_rect = defender_label.get_rect(center=(3.6* WIDTH // 4 + 50, HEIGHT // 2 + 100))

        screen.blit(attacker_label, attacker_label_rect)
        screen.blit(defender_label, defender_label_rect)

        return rect_attacker, rect_defender

    except pygame.error as e:
        print(f"Unable to load image: {e}")
        sys.exit(1)

def draw_info_panel(screen, game, move_result):
    font = pygame.font.Font(None, 35)
    info_panel_width = INFO_PANEL_WIDTH+200
    info_panel_height = HEIGHT
    info_panel_x = WIDTH-200
    info_panel_y = 0

    # Draw the beige background for the info panel
    pygame.draw.rect(screen, (245, 245, 220), (info_panel_x, info_panel_y, info_panel_width, info_panel_height))

    # Render the text on top of the beige background
    info_text = [
        f"Max Turns: {game.options.max_turns}",
        #f"Alpha-Beta: {'On' if game.options.alpha_beta else 'Off'}",
        f"Play Mode: {game.options.game_type}",
        #f"Attacker: H & Defender: AI" if game.options.game_type == GameType.AttackerVsComp else "Attacker: AI & Defender: H",
        #f"Heuristic used by AI: e{game.options.heuristic}",
        f"Next player: {game.next_player.name}",
        f"Turns played: {game.turns_played}",
    ]

    y_offset = 12
    for line in info_text:
        text_surface = font.render(line, True, (0, 0, 0))
        screen.blit(text_surface, (info_panel_x + 10, y_offset))
        y_offset += 60

def draw_winner(screen, winner):
    font = pygame.font.Font(None, 50)
    if winner == Player.Attacker:
        text_winner = font.render('Attacker Wins!', True, BLACK)
        winner_image = pygame.image.load('haha.png')  # Load the attacker image
    else:
        text_winner = font.render('Defender Wins!', True, BLACK)
        winner_image = pygame.image.load('haha.png')  # Load the defender image

    # Scale the image if necessary
    winner_image = pygame.transform.scale(winner_image, (200, 250))
    winner_image_rect = winner_image.get_rect(center=(1.4*WIDTH // 2, HEIGHT // 2 -100))  # Position the image above the text

    text_rect = text_winner.get_rect(center=(1.4*WIDTH // 2, HEIGHT // 2 + 50))
    screen.fill((245, 245, 220))
    
    # Blit the image onto the screen
    screen.blit(winner_image, winner_image_rect)
    # Blit the text onto the screen
    screen.blit(text_winner, text_rect)
    
    pygame.display.flip()


def show_splash_screen(screen):
    try:
        splash_image = Image.open('waiting-7579_256.gif')
        frames = []

        for frame in range(0, splash_image.n_frames):
            splash_image.seek(frame)
            # Resize each frame
            resized_frame = splash_image.resize((150, 150))
            frame_image = pygame.image.fromstring(
                resized_frame.tobytes(), resized_frame.size, resized_frame.mode
            )
            frames.append(frame_image)

        start_time = time.time()
        frame_duration = splash_image.info['duration'] / 600.0  # Convert milliseconds to seconds

        gif_width, gif_height = 150, 150
        gif_x = (WIDTH + INFO_PANEL_WIDTH - gif_width) // 2
        gif_y = (HEIGHT - gif_height) // 2

        while time.time() - start_time < 1:  # Display splash screen for 3 seconds
            for frame in frames:
                screen.fill((245, 245, 220))  # Fill screen with beige color
                frame = pygame.transform.scale(frame, (gif_width, gif_height))
                screen.blit(frame, (gif_x, gif_y))
                pygame.display.flip()
                time.sleep(frame_duration)

    except pygame.error as e:
        print(f"Unable to load splash image: {e}")
        sys.exit(1)


def main_pygame():
    print("Initializing Pygame")
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WIDTH + INFO_PANEL_WIDTH, HEIGHT))
    pygame.display.set_caption("Attacker vs Defender")
    blank_surface = pygame.Surface((1, 1))
    pygame.display.set_icon(blank_surface)
    clock = pygame.time.Clock()
    print("Pygame initialized")

    show_splash_screen(screen)

    game_ended = False
    move_result = ""

    rect_attacker, rect_defender = draw_menu(screen)
    pygame.display.flip()

    running = True
    game_started = False
    game = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and not game_ended:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if not game_started:
                    if rect_attacker.collidepoint(mouse_x, mouse_y):
                        options = Options(game_type=GameType.AttackerVsComp)
                        game = Game(options=options)
                        game_started = True
                        print("Game started: Attacker vs Comp")
                    elif rect_defender.collidepoint(mouse_x, mouse_y):
                        options = Options(game_type=GameType.CompVsDefender)
                        game = Game(options=options)
                        game_started = True
                        print("Game started: Comp vs Defender")
                else:
                    grid_x, grid_y = mouse_x // CELL_SIZE, mouse_y // CELL_SIZE
                    handle_click(game, grid_x, grid_y)
            elif event.type == pygame.KEYDOWN and game_ended:
                if event.key == pygame.K_RETURN:
                    running = False

        if game_started and not game_ended:
            screen.fill(WHITE)
            draw_grid(screen)
            draw_units(screen, game)
            draw_info_panel(screen, game, move_result)
            pygame.display.flip()

            clock.tick(FPS)

            if game.options.game_type == GameType.AttackerVsComp and game.next_player == Player.Defender:
                move = game.computer_turn()
                if move is not None:
                    move_result = f"Computer {game.next_player.name} moved"
                    print(move_result)
                    pygame.time.wait(500)
            elif game.options.game_type == GameType.CompVsDefender and game.next_player == Player.Attacker:
                move = game.computer_turn()
                if move is not None:
                    move_result = f"Computer {game.next_player.name} moved"
                    print(move_result)
                    pygame.time.wait(500) 

            winner = game.has_winner()
            if winner is not None:
                game_ended = True
                draw_winner(screen, winner)
                pygame.display.flip()
                print(f"{winner.name} wins!\n{game.turns_played} turns played")
        elif game_ended:
            draw_winner(screen, winner)
            pygame.display.flip()
        else:
            rect_attacker, rect_defender = draw_menu(screen)
            pygame.display.flip()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main_pygame()
