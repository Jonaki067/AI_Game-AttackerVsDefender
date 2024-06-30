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
    max_turns: Optional[int] = 300
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
            if not bot:
                print("The attacker’s AI, Firewall and Program can only move up or left")
            return False

        if (self.board[coords.src.row][coords.src.col].player == Player.Defender and coords.src.col > coords.dst.col) or (self.board[coords.src.row][coords.src.col].player == Player.Defender and coords.src.row > coords.dst.row):
            if not bot:
                print("The defender’s AI, Firewall and Program can only move down or right")
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
                if target.health == 9:
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
        mv = self.suggest_move()
        if mv is not None:
            success, result, actionType = self.perform_move(mv)
            if success:
                print(f"Computer {self.next_player.name}: {result if result else f'move from {mv.src.to_string()} to {mv.dst.to_string()}'}")
                self.next_turn()
                print("\n" + str(self))
                move = (mv, actionType)
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

    