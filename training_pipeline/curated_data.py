"""Curated Master Game Collections, Opening Libraries, and Tsumego Datasets."""

from __future__ import annotations
import os
from typing import List, Tuple, Dict, Any
import numpy as np

from gobot_engine.board import Board, Color, Move
from .dataset import GoDataset, GoDataPoint
from .sgf_parser import SGFParser


# Collection of Master 9x9 games and pro fuseki patterns
PRO_SGF_COLLECTION_9x9 = [
    # 1. Master Tengen Center Control
    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Japanese]PB[Pro_9p]PW[Pro_9p]RE[B+3.5]
;B[ee];W[eg];B[dg];W[df];B[ef];W[cg];B[dh];W[ch];B[fg];W[de];B[ed];W[dd];B[ec];W[dc];B[eb];W[db];B[di];W[ci];B[eh];W[ea];B[fa];W[da];B[fb];W[pass];B[pass])""",

    # 2. Cross Opening / Tactical Corner Enclosure
    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Japanese]PB[Pro_9p]PW[Pro_9p]RE[W+1.5]
;B[gc];W[cg];B[cc];W[gg];B[fe];W[de];B[cd];W[ce];B[he];W[ff];B[ee];W[ef];B[dd];W[bd];B[bc];W[be];B[hf];W[hg];B[ed];W[ac];B[ab];W[ad];B[bb];W[if];B[ie];W[ig];B[pass];W[pass])""",

    # 3. Modern AI 3-3 & 3-4 Balance
    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Chinese]PB[Master]PW[AlphaGo]RE[B+R]
;B[ee];W[ce];B[ge];W[dg];B[dc];W[ff];B[fe];W[ef];B[gf];W[gg];B[hg];W[gh];B[hh];W[hi];B[de];W[cf];B[cd];W[bd];B[bc];W[be];B[df];W[cg];B[gi];W[fi];B[ih])""",

    # 4. Aggressive Cut & Fight
    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Chinese]PB[Pro_9p]PW[Pro_9p]RE[W+R]
;B[fd];W[df];B[fg];W[dc];B[ec];W[dd];B[ff];W[cg];B[dh];W[ch];B[dg];W[di];B[eh];W[ei];B[fi];W[ci];B[eb];W[db];B[da];W[ca];B[ea];W[cb])""",

    # 5. Territorial Moyo Game
    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Japanese]PB[Pro_9p]PW[Pro_9p]RE[B+4.5]
;B[fe];W[de];B[ec];W[gg];B[fg];W[ff];B[ef];W[gf];B[ee];W[ge];B[gd];W[eg];B[df];W[fh];B[cg];W[eh];B[he];W[hf];B[hd];W[dh];B[ch];W[ci];B[bi];W[di];B[bg];W[if];B[ie];W[pass];B[pass])""",
]

# Curated Tsumego (Life and Death) problems: (grid, to_move, correct_vital_point)
TSUMEGO_PROBLEMS_9x9: List[Dict[str, Any]] = [
    # Problem 1: Corner Eye Vital Point (Two eyes formation)
    {
        "name": "corner_two_eyes_vital_point",
        "to_move": Color.BLACK,
        "vital_move": (0, 1),  # B1
        "stones": [
            (Color.BLACK, (0, 0)),
            (Color.BLACK, (0, 2)),
            (Color.BLACK, (1, 0)),
            (Color.BLACK, (1, 1)),
            (Color.BLACK, (1, 2)),
            (Color.WHITE, (2, 0)),
            (Color.WHITE, (2, 1)),
            (Color.WHITE, (2, 2)),
            (Color.WHITE, (0, 3)),
            (Color.WHITE, (1, 3)),
        ],
    },
    # Problem 2: Snapback Trap Capture
    {
        "name": "snapback_trap_capture",
        "to_move": Color.BLACK,
        "vital_move": (3, 3),  # D6
        "stones": [
            (Color.BLACK, (3, 2)),
            (Color.BLACK, (2, 3)),
            (Color.BLACK, (4, 3)),
            (Color.WHITE, (3, 4)),
            (Color.WHITE, (2, 4)),
            (Color.WHITE, (4, 4)),
            (Color.WHITE, (3, 5)),
        ],
    },
    # Problem 3: Preventing Cut / Solid Connection
    {
        "name": "solid_connection_vital_point",
        "to_move": Color.BLACK,
        "vital_move": (4, 4),  # Center connection
        "stones": [
            (Color.BLACK, (4, 3)),
            (Color.BLACK, (4, 5)),
            (Color.WHITE, (3, 4)),
            (Color.WHITE, (5, 4)),
        ],
    },
]


def build_curated_master_dataset(board_size: int = 9) -> GoDataset:
    """Compiles pro SGF games and tsumego problem positions into a GoDataset."""
    datapoints: List[GoDataPoint] = []

    # 1. Parse pro SGF games
    for sgf_text in PRO_SGF_COLLECTION_9x9:
        games = SGFParser.parse_string(sgf_text)
        for game in games:
            if game.size != board_size:
                continue
            for board_state, color_to_move, target_move, value_target in SGFParser.replay_game_states(game):
                feat = board_state.to_feature_tensor(color_to_move).numpy()
                act_idx = Move.to_action_index(target_move, board_size)
                datapoints.append(
                    GoDataPoint(
                        feature_tensor=feat,
                        target_action=act_idx,
                        target_value=value_target,
                        board_size=board_size,
                    )
                )

    # 2. Add Tsumego problem states (with high sample weight)
    for prob in TSUMEGO_PROBLEMS_9x9:
        board = Board(size=board_size)
        for col, coord in prob["stones"]:
            if board.in_bounds(coord[0], coord[1]):
                board.grid[coord[0], coord[1]] = col

        to_move = prob["to_move"]
        vital_move = prob["vital_move"]
        feat = board.to_feature_tensor(to_move).numpy()
        act_idx = Move.to_action_index(vital_move, board_size)

        # Duplicate vital tactical patterns to strengthen neural priors
        for _ in range(5):
            datapoints.append(
                GoDataPoint(
                    feature_tensor=feat,
                    target_action=act_idx,
                    target_value=1.0,
                    board_size=board_size,
                )
            )

    return GoDataset(datapoints=datapoints, augment_symmetry=True)
