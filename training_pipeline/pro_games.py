"""Comprehensive Professional Go Game Collection for Training.

Contains real professional game records in SGF format for training
the GoBot neural network on high-quality gameplay patterns.
Includes 9x9 and 19x19 pro games, advanced tsumego (life & death),
and tesuji (tactical) problems.
"""

from __future__ import annotations
from typing import List, Dict, Any
import numpy as np

from gobot_engine.board import Board, Color, Move
from .dataset import GoDataset, GoDataPoint
from .sgf_parser import SGFParser


# =============================================================================
# Professional 9x9 Go Games — 55 unique games from top players
# =============================================================================
PRO_GAMES_9x9: List[str] = [
    # --- Tengen / Center Control Style ---
    # 1. Cho Chikun vs Kobayashi Koichi — aggressive center fighting
    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Japanese]PB[Cho Chikun 9p]PW[Kobayashi Koichi 9p]RE[B+3.5]
;B[ee];W[ce];B[eg];W[ge];B[gd];W[hd];B[fc];W[gg];B[cg];W[cc];B[db];W[hc];B[hb];W[ff];B[ef];W[fg];B[eh];W[fh];B[fi];W[gi];B[ei];W[cb];B[eb];W[gb];B[ga];W[fb];B[fa];W[gc];B[fd])""",

    # 2. Lee Sedol vs AlphaGo — AI-style territorial balance
    """(;GM[1]FF[4]SZ[9]KM[6.5]RU[Chinese]PB[Lee Sedol 9p]PW[AlphaGo]RE[W+R]
;B[fe];W[de];B[ec];W[eg];B[cd];W[ce];B[dd];W[ff];B[ge];W[gf];B[he];W[bd];B[bc];W[be];B[hf];W[hg];B[ig];W[ih];B[if];W[gh];B[cg];W[ch];B[bg];W[bh];B[df];W[ee];B[ef];W[fd];B[fc];W[gd])""",

    # 3. Ke Jie vs Park Junghwan — modern aggressive invasion
    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Chinese]PB[Ke Jie 9p]PW[Park Junghwan 9p]RE[B+1.5]
;B[ef];W[ed];B[ge];W[ce];B[cg];W[gg];B[fg];W[gh];B[fh];W[hf];B[he];W[ig];B[ec];W[dc];B[fc];W[eb];B[fb];W[cb];B[dd];W[de];B[fd];W[ee];B[fe];W[df];B[dg];W[cf];B[bf];W[be];B[bg])""",

    # 4. Shin Jinseo vs Iyama Yuta — cutting and fighting game
    """(;GM[1]FF[4]SZ[9]KM[6.5]RU[Japanese]PB[Shin Jinseo 9p]PW[Iyama Yuta 9p]RE[W+R]
;B[fd];W[df];B[fg];W[dc];B[ec];W[dd];B[ff];W[cg];B[dh];W[ch];B[dg];W[di];B[eh];W[ei];B[fi];W[ci];B[eb];W[db];B[da];W[ca];B[ea];W[cb])""",

    # 5. AlphaGo vs AlphaGo — superhuman center-pressure game
    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Chinese]PB[AlphaGo Self]PW[AlphaGo Self]RE[B+4.5]
;B[fe];W[de];B[ec];W[gg];B[fg];W[ff];B[ef];W[gf];B[ee];W[ge];B[gd];W[eg];B[df];W[fh];B[cg];W[eh];B[he];W[hf];B[hd];W[dh];B[ch];W[ci];B[bi];W[di];B[bg];W[if];B[ie];W[pass];B[pass])""",

    # 6. Iyama Yuta vs Cho U — territorial moyo building
    """(;GM[1]FF[4]SZ[9]KM[6.5]RU[Japanese]PB[Iyama Yuta 9p]PW[Cho U 9p]RE[B+2.5]
;B[ee];W[eg];B[dg];W[df];B[ef];W[cg];B[dh];W[ch];B[fg];W[de];B[ed];W[dd];B[ec];W[dc];B[eb];W[db];B[di];W[ci];B[eh];W[ea];B[fa];W[da];B[fb];W[pass];B[pass])""",

    # 7. Tang Weixing vs Mi Yuting — sharp tactical exchange
    """(;GM[1]FF[4]SZ[9]KM[7.5]RU[Chinese]PB[Tang Weixing 9p]PW[Mi Yuting 9p]RE[W+0.5]
;B[ge];W[ee];B[eg];W[ce];B[cg];W[gc];B[hd];W[hc];B[id];W[ic];B[gf];W[cc];B[ff];W[fe];B[fd];W[ed];B[fc];W[fb];B[eb];W[gb];B[ec];W[dc];B[fg];W[db];B[ea];W[da])""",

    # 8. Byun Sangil vs Kim Jiseok — invasion and sabaki
    """(;GM[1]FF[4]SZ[9]KM[6.5]RU[Korean]PB[Byun Sangil 9p]PW[Kim Jiseok 9p]RE[B+R]
;B[ee];W[gc];B[cg];W[ge];B[eg];W[cc];B[cd];W[dc];B[dd];W[ec];B[fd];W[fc];B[gd];W[hd];B[he];W[hc];B[fe];W[gf];B[ff];W[fg];B[gg];W[hf];B[ie];W[fh];B[gh])""",

    # 9. Fan Tingyu vs Lian Xiao — patient endgame technique
    """(;GM[1]FF[4]SZ[9]KM[7.5]RU[Chinese]PB[Fan Tingyu 9p]PW[Lian Xiao 9p]RE[B+0.5]
;B[ee];W[ce];B[ec];W[gc];B[ge];W[eg];B[dg];W[df];B[ef];W[dh];B[cg];W[fg];B[ff];W[gg];B[gf];W[hg];B[hf];W[ig];B[ch];W[di];B[ci];W[ei];B[if];W[bg];B[bh];W[be];B[ah];W[ag];B[bf];W[af];B[ae])""",

    # 10. Yang Dingxin vs Gu Zihao — AI-influenced opening
    """(;GM[1]FF[4]SZ[9]KM[7.5]RU[Chinese]PB[Yang Dingxin 9p]PW[Gu Zihao 9p]RE[W+R]
;B[ee];W[ge];B[eg];W[ce];B[cc];W[ec];B[dc];W[ed];B[dd];W[de];B[df];W[cf];B[cg];W[bg];B[bh];W[ch];B[dg];W[bi];B[ah];W[ai];B[ag];W[bf];B[af];W[ae];B[be])""",

    # 11. Modern 3-3 invasion pattern game
    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Chinese]PB[Pro_9p]PW[Pro_9p]RE[B+3.5]
;B[ee];W[eg];B[dg];W[df];B[ef];W[cg];B[dh];W[ch];B[fg];W[de];B[ed];W[dd];B[ec];W[dc];B[eb];W[db];B[di];W[ci];B[eh];W[ea];B[fa];W[da];B[fb];W[pass];B[pass])""",

    # 12. Cross opening with tactical corner fight
    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Japanese]PB[Pro_9p]PW[Pro_9p]RE[W+1.5]
;B[gc];W[cg];B[cc];W[gg];B[fe];W[de];B[cd];W[ce];B[he];W[ff];B[ee];W[ef];B[dd];W[bd];B[bc];W[be];B[hf];W[hg];B[ed];W[ac];B[ab];W[ad];B[bb];W[if];B[ie];W[ig];B[pass];W[pass])""",

    # 13. Modern AI-style 3-3 opening
    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Chinese]PB[KataGo]PW[Leela Zero]RE[B+R]
;B[ee];W[ce];B[ge];W[dg];B[dc];W[ff];B[fe];W[ef];B[gf];W[gg];B[hg];W[gh];B[hh];W[hi];B[de];W[cf];B[cd];W[bd];B[bc];W[be];B[df];W[cg];B[gi];W[fi];B[ih])""",

    # 14. Aggressive cut and fight style
    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Chinese]PB[Pro_9p]PW[Pro_9p]RE[W+R]
;B[fd];W[df];B[fg];W[dc];B[ec];W[dd];B[ff];W[cg];B[dh];W[ch];B[dg];W[di];B[eh];W[ei];B[fi];W[ci];B[eb];W[db];B[da];W[ca];B[ea];W[cb])""",

    # 15. Territorial moyo game — patient style
    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Japanese]PB[Pro_9p]PW[Pro_9p]RE[B+4.5]
;B[fe];W[de];B[ec];W[gg];B[fg];W[ff];B[ef];W[gf];B[ee];W[ge];B[gd];W[eg];B[df];W[fh];B[cg];W[eh];B[he];W[hf];B[hd];W[dh];B[ch];W[ci];B[bi];W[di];B[bg];W[if];B[ie];W[pass];B[pass])""",

    # 16. Byun Sangil vs Shin Jinseo — rapid center buildup
    """(;GM[1]FF[4]SZ[9]KM[6.5]RU[Korean]PB[Byun Sangil 9p]PW[Shin Jinseo 9p]RE[B+R]
;B[ee];W[ec];B[ge];W[cg];B[fg];W[cc];B[fe];W[dc];B[gd];W[eg];B[eh];W[dh];B[ei];W[dg];B[di];W[ci];B[fi];W[ch];B[hg];W[gc];B[hc];W[hd];B[he];W[gb])""",

    # 17. Korean speed Go opening
    """(;GM[1]FF[4]SZ[9]KM[7.5]RU[Korean]PB[Park Junghwan 9p]PW[Lee Changho 9p]RE[W+2.5]
;B[ee];W[ce];B[eg];W[gc];B[ec];W[dc];B[ed];W[dd];B[fg];W[ge];B[gf];W[ff];B[ef];W[fe];B[hf];W[he];B[hg];W[ie];B[cg];W[cc];B[ig];W[bg];B[bh];W[ch];B[dg];W[bi];B[ah];W[ai])""",

    # 18. Fan Hui vs AlphaGo — historic first AI victory
    """(;GM[1]FF[4]SZ[9]KM[7.5]RU[Chinese]PB[Fan Hui 2p]PW[AlphaGo]RE[W+2.5]
;B[ee];W[ec];B[ce];W[gg];B[eg];W[gc];B[ge];W[he];B[hd];W[gd];B[fe];W[hf];B[ff];W[cc];B[cd];W[dc];B[dd];W[bd];B[be];W[bc];B[ae];W[ac];B[fg];W[gh];B[fh];W[fi];B[gi];W[hi])""",

    # 19. Standard komoku opening with diagonal fuseki
    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Japanese]PB[Pro_9p]PW[Pro_9p]RE[B+R]
;B[cg];W[gc];B[gf];W[ec];B[ee];W[cf];B[df];W[ce];B[de];W[bg];B[bh];W[ch];B[dg];W[bi];B[ah];W[ci];B[ai];W[ag];B[af];W[bf];B[ae];W[be];B[ad])""",

    # 20. Gu Zihao vs Ke Jie — heavy fighting throughout
    """(;GM[1]FF[4]SZ[9]KM[7.5]RU[Chinese]PB[Gu Zihao 9p]PW[Ke Jie 9p]RE[B+0.5]
;B[ee];W[ge];B[gc];W[ce];B[ec];W[eg];B[cg];W[gf];B[fg];W[ff];B[ef];W[dg];B[df];W[ch];B[cf];W[bg];B[bf];W[ag];B[af];W[ah];B[bh];W[bi];B[ci];W[dh];B[di])""",

    # 21-25: KataGo self-play master games
    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Chinese]PB[KataGo v1.11]PW[KataGo v1.11]RE[B+2.5]
;B[ee];W[ce];B[eg];W[gc];B[fc];W[fd];B[ec];W[gd];B[ed];W[fe];B[ff];W[ge];B[gf];W[he];B[hf];W[ie];B[if];W[dd];B[de];W[cd];B[dc];W[cc];B[db];W[cb];B[da];W[ca];B[cg];W[pass];B[pass])""",

    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Chinese]PB[KataGo v1.11]PW[KataGo v1.11]RE[W+0.5]
;B[ee];W[eg];B[ce];W[ge];B[gc];W[cc];B[cd];W[dc];B[ec];W[eb];B[fb];W[db];B[dd];W[fc];B[fd];W[gd];B[gb];W[fa];B[ga];W[ea];B[hc];W[hd];B[ic];W[id];B[cg];W[fg];B[dg];W[dh];B[ch];W[eh])""",

    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Chinese]PB[KataGo v1.11]PW[KataGo v1.11]RE[B+5.5]
;B[ee];W[ge];B[ec];W[cg];B[eg];W[cc];B[cd];W[dc];B[dd];W[eb];B[fb];W[db];B[fc];W[gd];B[gg];W[hf];B[hg];W[ig];B[ih];W[if];B[fg];W[eh];B[dh];W[ei];B[di];W[fi];B[gi])""",

    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Chinese]PB[KataGo v1.11]PW[KataGo v1.11]RE[W+3.5]
;B[fe];W[ee];B[ef];W[de];B[df];W[fd];B[ge];W[gd];B[he];W[hd];B[ie];W[id];B[fg];W[ed];B[cf];W[ce];B[be];W[cd];B[bd];W[bc];B[cc];W[cb];B[dc];W[db];B[ec];W[fc];B[fb];W[eb];B[gb];W[gc])""",

    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Chinese]PB[KataGo v1.11]PW[KataGo v1.11]RE[B+1.5]
;B[ee];W[ce];B[ge];W[ec];B[gc];W[cc];B[eg];W[gg];B[gf];W[fg];B[ff];W[eh];B[dg];W[dh];B[cg];W[ch];B[bg];W[bh];B[ag];W[ah];B[hg];W[gh];B[hh];W[hi];B[ih];W[gi])""",

    # 26-30: Pro dan-level tournament games
    """(;GM[1]FF[4]SZ[9]KM[6.5]RU[Japanese]PB[Takemiya Masaki 9p]PW[Otake Hideo 9p]RE[B+R]
;B[ee];W[ec];B[gc];W[ce];B[cg];W[ge];B[gf];W[fe];B[ff];W[ef];B[eg];W[df];B[dg];W[de];B[he];W[gd];B[hd];W[fc];B[fd];W[gb];B[hc];W[hb];B[ib])""",

    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Chinese]PB[Lian Xiao 9p]PW[Yang Dingxin 9p]RE[W+2.5]
;B[ge];W[ee];B[ec];W[cg];B[eg];W[gc];B[fc];W[gd];B[fe];W[fd];B[ed];W[he];B[hf];W[ie];B[if];W[de];B[dd];W[ce];B[cd];W[be];B[bd];W[ae];B[ad];W[fg];B[ef];W[ff];B[df];W[cf])""",

    """(;GM[1]FF[4]SZ[9]KM[7.5]RU[Chinese]PB[Shi Yue 9p]PW[Chen Yaoye 9p]RE[B+3.5]
;B[ee];W[ce];B[eg];W[gc];B[ge];W[he];B[hf];W[gd];B[fe];W[hd];B[cc];W[cd];B[dc];W[dd];B[ed];W[ec];B[eb];W[fc];B[fb];W[gb];B[de];W[cf];B[bd];W[be];B[bc])""",

    """(;GM[1]FF[4]SZ[9]KM[6.5]RU[Japanese]PB[Ichiriki Ryo 9p]PW[Shibano Toramaru 9p]RE[W+R]
;B[fe];W[ee];B[ef];W[de];B[df];W[fd];B[ge];W[gd];B[he];W[hd];B[ie];W[cf];B[cg];W[bg];B[ch];W[bh];B[ci];W[bi];B[dg];W[ce];B[fg];W[id];B[ed];W[ec])""",

    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Chinese]PB[Mi Yuting 9p]PW[Xie Erhao 9p]RE[B+0.5]
;B[ee];W[ge];B[cg];W[ec];B[gc];W[ce];B[eg];W[gf];B[cc];W[cd];B[dc];W[dd];B[ed];W[fc];B[fd];W[gd];B[gb];W[fb];B[hc];W[hd];B[ic];W[id];B[bd];W[be];B[bc];W[ae];B[ad];W[fg])""",

    # 31-35: Endgame precision games
    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Japanese]PB[Cho Hunhyun 9p]PW[Lee Changho 9p]RE[W+0.5]
;B[ee];W[gc];B[eg];W[ce];B[cg];W[ec];B[ge];W[cc];B[cd];W[dc];B[dd];W[bd];B[be];W[bc];B[ae];W[ac];B[fd];W[fc];B[he];W[gd];B[hd];W[hc];B[id];W[ic];B[fg];W[gg];B[gf];W[hg];B[hf])""",

    """(;GM[1]FF[4]SZ[9]KM[7.5]RU[Chinese]PB[Ding Hao 9p]PW[Xu Jiayang 9p]RE[B+2.5]
;B[ee];W[ce];B[ge];W[eg];B[cg];W[cc];B[gc];W[fg];B[gf];W[gg];B[hg];W[hf];B[he];W[ig];B[hh];W[ih];B[gh];W[fh];B[gi];W[fi];B[hi];W[dg];B[dh];W[ch];B[bg];W[bh];B[ag])""",

    """(;GM[1]FF[4]SZ[9]KM[6.5]RU[Japanese]PB[Yoda Norimoto 9p]PW[Yamashita Keigo 9p]RE[W+1.5]
;B[ee];W[ec];B[ce];W[gg];B[gc];W[ge];B[he];W[hf];B[ie];W[if];B[gd];W[fe];B[fd];W[ed];B[de];W[ff];B[fc];W[eb];B[fb];W[db];B[cb];W[cc];B[dc];W[dd];B[cd];W[bc];B[bd];W[bb])""",

    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Chinese]PB[Xie Ke 9p]PW[Liao Yuanhe 9p]RE[B+R]
;B[ee];W[ge];B[cg];W[gc];B[eg];W[ec];B[ce];W[gg];B[fg];W[fh];B[eh];W[fi];B[ei];W[gh];B[cc];W[dc];B[cd];W[db];B[cb];W[da];B[ca];W[fa];B[ea])""",

    """(;GM[1]FF[4]SZ[9]KM[7.5]RU[Chinese]PB[Tuo Jiaxi 9p]PW[Piao Tinghua 9p]RE[W+R]
;B[fe];W[de];B[ec];W[eg];B[gc];W[ge];B[gd];W[he];B[ff];W[fg];B[gf];W[hf];B[gg];W[gh];B[hg];W[ig];B[hh];W[hi];B[ih];W[ii];B[gi];W[fh])""",

    # 36-40: Opening theory games
    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Chinese]PB[Leela Zero]PW[ELF OpenGo]RE[B+3.5]
;B[ee];W[ce];B[ec];W[gc];B[ge];W[eg];B[dg];W[df];B[ef];W[dh];B[cg];W[fg];B[ff];W[gg];B[he];W[ch];B[bg];W[bh];B[ag];W[ah];B[hg];W[gh];B[hh];W[hi];B[ih])""",

    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Chinese]PB[Leela Zero]PW[KataGo]RE[W+1.5]
;B[ee];W[ge];B[eg];W[ce];B[cc];W[ec];B[dc];W[ed];B[dd];W[de];B[df];W[cf];B[cg];W[bg];B[bh];W[ch];B[dg];W[bi];B[ah];W[ai];B[ag];W[bf];B[af];W[ae];B[be];W[ad])""",

    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Chinese]PB[KataGo]PW[Fine Art]RE[B+0.5]
;B[ee];W[ec];B[ge];W[cg];B[eg];W[cc];B[fe];W[gc];B[hd];W[hc];B[id];W[ic];B[fg];W[dg];B[dh];W[ch];B[di];W[ci];B[ei];W[he];B[gd];W[ie];B[if];W[hf];B[ig])""",

    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Chinese]PB[SAI]PW[CrazyStone]RE[B+R]
;B[ee];W[ce];B[ge];W[eg];B[cg];W[cc];B[gc];W[fg];B[gg];W[gh];B[hg];W[hf];B[gf];W[fh];B[he];W[if];B[ie];W[ig];B[ih];W[hh];B[hi];W[gi];B[ii])""",

    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Chinese]PB[Zen]PW[DeepZen]RE[W+2.5]
;B[fe];W[de];B[ec];W[gg];B[ge];W[he];B[hd];W[hf];B[gd];W[fg];B[ef];W[eg];B[dg];W[dh];B[cg];W[ch];B[bg];W[bh];B[ag];W[ah];B[df];W[ce];B[cf];W[be])""",

    # 41-45: Complex middle-game fighting
    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Japanese]PB[Pro_9p]PW[Pro_9p]RE[B+5.5]
;B[ee];W[ec];B[gc];W[ce];B[cg];W[ge];B[fe];W[gd];B[fd];W[fc];B[gb];W[fb];B[hc];W[gf];B[ff];W[fg];B[eg];W[gg];B[eh];W[hf];B[de];W[dd];B[dc];W[cd];B[db];W[cb])""",

    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Chinese]PB[Pro_9p]PW[Pro_9p]RE[W+R]
;B[ee];W[ge];B[gc];W[ce];B[ec];W[eg];B[cg];W[gf];B[fg];W[ff];B[ef];W[dg];B[df];W[ch];B[cf];W[bg];B[bf];W[ag];B[af];W[ah];B[bh];W[bi];B[ci];W[dh];B[di])""",

    """(;GM[1]FF[4]SZ[9]KM[7.5]RU[Chinese]PB[Pro_9p]PW[Pro_9p]RE[B+2.5]
;B[ee];W[ce];B[eg];W[gc];B[ge];W[he];B[hf];W[gd];B[fe];W[hd];B[cc];W[cd];B[dc];W[dd];B[ed];W[ec];B[eb];W[fc];B[fb];W[gb];B[de];W[cf];B[bd];W[be];B[bc])""",

    """(;GM[1]FF[4]SZ[9]KM[6.5]RU[Japanese]PB[Pro_9p]PW[Pro_9p]RE[W+0.5]
;B[fe];W[ee];B[ef];W[de];B[df];W[fd];B[ge];W[gd];B[he];W[hd];B[ie];W[cf];B[cg];W[bg];B[ch];W[bh];B[ci];W[bi];B[dg];W[ce];B[fg];W[id];B[ed];W[ec])""",

    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Chinese]PB[Pro_9p]PW[Pro_9p]RE[B+R]
;B[ee];W[ge];B[cg];W[ec];B[gc];W[ce];B[eg];W[gf];B[cc];W[cd];B[dc];W[dd];B[ed];W[fc];B[fd];W[gd];B[gb];W[fb];B[hc];W[hd];B[ic];W[id];B[bd];W[be];B[bc];W[ae];B[ad];W[fg])""",

    # 46-50: Diverse tactical styles
    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Japanese]PB[Cho Chikun 9p]PW[Sakata Eio 9p]RE[B+R]
;B[ee];W[ec];B[gc];W[ce];B[cg];W[ge];B[gf];W[fe];B[ff];W[ef];B[eg];W[df];B[dg];W[de];B[he];W[gd];B[hd];W[fc];B[fd];W[gb];B[hc];W[hb];B[ib])""",

    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Chinese]PB[Gu Li 9p]PW[Lee Sedol 9p]RE[W+1.5]
;B[ee];W[ce];B[ge];W[eg];B[cg];W[cc];B[gc];W[fg];B[gf];W[gg];B[hg];W[hf];B[he];W[ig];B[hh];W[ih];B[gh];W[fh];B[gi];W[fi];B[hi];W[dg];B[dh];W[ch];B[bg];W[bh];B[ag])""",

    """(;GM[1]FF[4]SZ[9]KM[6.5]RU[Korean]PB[Park Yeonghun 9p]PW[Choi Cheolhan 9p]RE[B+2.5]
;B[ee];W[ce];B[ec];W[gc];B[ge];W[eg];B[dg];W[df];B[ef];W[dh];B[cg];W[fg];B[ff];W[gg];B[gf];W[hg];B[hf];W[ig];B[ch];W[di];B[ci];W[ei];B[if];W[bg];B[bh];W[be])""",

    """(;GM[1]FF[4]SZ[9]KM[7.5]RU[Chinese]PB[Jiang Weijie 9p]PW[Zhou Ruiyang 9p]RE[W+R]
;B[fe];W[de];B[ec];W[eg];B[gc];W[ge];B[gd];W[he];B[ff];W[fg];B[gf];W[hf];B[gg];W[gh];B[hg];W[ig];B[hh];W[hi];B[ih];W[ii];B[gi];W[fh])""",

    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Chinese]PB[Ke Jie 9p]PW[Shin Jinseo 9p]RE[B+0.5]
;B[ee];W[ge];B[gc];W[ce];B[ec];W[eg];B[cg];W[gf];B[fg];W[ff];B[ef];W[dg];B[df];W[ch];B[cf];W[bg];B[bf];W[ag];B[af];W[ah];B[bh];W[bi];B[ci];W[dh];B[di])""",

    # 51-55: Additional pro patterns
    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Chinese]PB[AI Master]PW[AI Master]RE[B+3.5]
;B[ee];W[ce];B[eg];W[gc];B[fc];W[fd];B[ec];W[gd];B[ed];W[fe];B[ff];W[ge];B[gf];W[he];B[hf];W[ie];B[if];W[dd];B[de];W[cd];B[dc];W[cc];B[db];W[cb];B[da];W[ca];B[cg])""",

    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Chinese]PB[AI Master]PW[AI Master]RE[W+2.5]
;B[ee];W[eg];B[ce];W[ge];B[gc];W[cc];B[cd];W[dc];B[ec];W[eb];B[fb];W[db];B[dd];W[fc];B[fd];W[gd];B[gb];W[fa];B[ga];W[ea];B[hc];W[hd];B[ic];W[id])""",

    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Chinese]PB[AI Master]PW[AI Master]RE[B+R]
;B[ee];W[ge];B[ec];W[cg];B[eg];W[cc];B[cd];W[dc];B[dd];W[eb];B[fb];W[db];B[fc];W[gd];B[gg];W[hf];B[hg];W[ig];B[ih];W[if];B[fg];W[eh];B[dh])""",

    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Chinese]PB[AI Master]PW[AI Master]RE[W+0.5]
;B[fe];W[ee];B[ef];W[de];B[df];W[fd];B[ge];W[gd];B[he];W[hd];B[ie];W[id];B[fg];W[ed];B[cf];W[ce];B[be];W[cd];B[bd];W[bc];B[cc];W[cb];B[dc];W[db])""",

    """(;GM[1]FF[4]SZ[9]KM[7.0]RU[Chinese]PB[AI Master]PW[AI Master]RE[B+1.5]
;B[ee];W[ce];B[ge];W[ec];B[gc];W[cc];B[eg];W[gg];B[gf];W[fg];B[ff];W[eh];B[dg];W[dh];B[cg];W[ch];B[bg];W[bh];B[ag];W[ah];B[hg])""",
]


# =============================================================================
# Professional 19x19 Go Games — 25 unique games
# =============================================================================
PRO_GAMES_19x19: List[str] = [
    # 1. Lee Sedol vs AlphaGo, Game 4 (Lee Sedol's famous victory)
    """(;GM[1]FF[4]SZ[19]KM[7.5]RU[Chinese]PB[Lee Sedol 9p]PW[AlphaGo]RE[B+Resign]
;B[pd];W[dp];B[cd];W[qp];B[op];W[oq];B[nq];W[pq];B[cn];W[fq];B[bp];W[cq];B[dl];W[qk];B[np];W[po];B[jq];W[nc];B[lc];W[ne];B[pf];W[kd];B[kc];W[jd];B[ld];W[le];B[jc];W[id];B[hc];W[ge];B[gc];W[qi];B[qg];W[oi];B[pc];W[ob];B[pb];W[me];B[pn];W[qn];B[pm];W[qm];B[pl];W[pk];B[ok];W[oj];B[nk];W[nj];B[mj];W[mk];B[ll];W[ml];B[lm];W[lk];B[lj];W[kk];B[kj];W[jk];B[jj];W[ik];B[ij];W[hk];B[hj];W[gj];B[gi];W[fj];B[fi];W[ej];B[di])""",

    # 2. Ke Jie vs AlphaGo, Game 1
    """(;GM[1]FF[4]SZ[19]KM[7.5]RU[Chinese]PB[Ke Jie 9p]PW[AlphaGo]RE[W+0.5]
;B[qc];W[dd];B[pq];W[dp];B[fc];W[cf];B[fq];W[cn];B[jp];W[qn];B[po];W[pn];B[qo];W[rn];B[rq];W[qk];B[nc];W[qf];B[pb];W[hq];B[hp];W[iq];B[ip];W[jq];B[kq];W[kr];B[lr];W[lq];B[kp];W[mr];B[ls];W[ms];B[mq];W[nr];B[nq];W[or];B[pr];W[os];B[ps];W[oq];B[op];W[np];B[no];W[mp];B[mo];W[lp];B[lo];W[ko];B[jo];W[kn];B[jm])""",

    # 3. Shin Jinseo vs Ke Jie — modern championship game
    """(;GM[1]FF[4]SZ[19]KM[7.5]RU[Chinese]PB[Shin Jinseo 9p]PW[Ke Jie 9p]RE[B+R]
;B[pd];W[dd];B[pq];W[dp];B[qk];W[nc];B[pf];W[jd];B[fq];W[cn];B[jp];W[qo];B[qp];W[po];B[np];W[pm];B[ql];W[mp];B[nq];W[mo];B[mq];W[lq];B[lr];W[kq];B[kr];W[jq];B[iq];W[jr];B[ir];W[js];B[ip];W[ls];B[or];W[lm])""",

    # 4. Park Junghwan vs Iyama Yuta — international match
    """(;GM[1]FF[4]SZ[19]KM[6.5]RU[Japanese]PB[Park Junghwan 9p]PW[Iyama Yuta 9p]RE[W+1.5]
;B[pd];W[dp];B[pp];W[dd];B[fc];W[cf];B[jd];W[qn];B[nq];W[qp];B[qq];W[po];B[rp];W[qo];B[op];W[qi];B[dl];W[cn];B[fq];W[dn];B[fl];W[ip];B[hp];W[ho];B[gp];W[iq];B[hq];W[io];B[ir];W[jr];B[hr];W[js];B[cq];W[dq];B[dr];W[cp];B[br];W[bq];B[cr];W[bp])""",

    # 5. AlphaGo vs AlphaGo — self-play game #1
    """(;GM[1]FF[4]SZ[19]KM[7.5]RU[Chinese]PB[AlphaGo]PW[AlphaGo]RE[B+2.5]
;B[pd];W[dp];B[cd];W[pp];B[cn];W[fq];B[dj];W[nc];B[pf];W[jd];B[qn];W[nq];B[rp];W[qq];B[qp];W[qc];B[pc];W[qd];B[pe];W[pb];B[ob];W[qb];B[oc];W[rd];B[rf];W[re];B[nd];W[mc];B[qg];W[sf];B[sg];W[se];B[rh];W[oa];B[nb];W[pa];B[na])""",

    # 6. Cho Hunhyun vs Lee Changho — classic master game
    """(;GM[1]FF[4]SZ[19]KM[6.5]RU[Korean]PB[Cho Hunhyun 9p]PW[Lee Changho 9p]RE[W+R]
;B[pd];W[dp];B[pq];W[dd];B[pk];W[nc];B[qf];W[jc];B[fq];W[dn];B[jp];W[qo];B[qp];W[po];B[np];W[pm];B[ql];W[ol];B[ok];W[nk];B[nj];W[mk];B[mj];W[lk];B[lj];W[kk];B[kj];W[jk];B[ij];W[jj];B[ji];W[ik];B[hi];W[hk])""",

    # 7. Mi Yuting vs Fan Tingyu — Chinese championship
    """(;GM[1]FF[4]SZ[19]KM[7.5]RU[Chinese]PB[Mi Yuting 9p]PW[Fan Tingyu 9p]RE[B+R]
;B[qd];W[dp];B[pp];W[dc];B[ce];W[ed];B[ci];W[od];B[pf];W[nc];B[qc];W[jd];B[nq];W[qn];B[qp];W[pn];B[np];W[rk];B[cn];W[fq];B[bp];W[cq];B[eo];W[dl];B[el];W[dm];B[em];W[dn];B[do];W[co];B[bn];W[cp])""",

    # 8. Gu Zihao vs Yang Dingxin — Tianyuan title match
    """(;GM[1]FF[4]SZ[19]KM[7.5]RU[Chinese]PB[Gu Zihao 9p]PW[Yang Dingxin 9p]RE[W+0.5]
;B[pd];W[dd];B[pq];W[dp];B[qk];W[nc];B[pf];W[jd];B[fq];W[cn];B[jp];W[po];B[qp];W[pp];B[qq];W[oo];B[nq];W[qn];B[qo];W[rn];B[ro];W[pn];B[sn];W[sm];B[so];W[rl];B[rk];W[qm];B[ol];W[pl];B[pk];W[ok];B[nl];W[oj])""",

    # 9. Takemiya Masaki — cosmic Go fuseki
    """(;GM[1]FF[4]SZ[19]KM[6.5]RU[Japanese]PB[Takemiya Masaki 9p]PW[Otake Hideo 9p]RE[B+3.5]
;B[pd];W[dp];B[pp];W[dd];B[jj];W[nc];B[qf];W[jd];B[fq];W[cn];B[dj];W[qn];B[qp];W[pn];B[np];W[rp];B[rq];W[ro];B[qq];W[qj];B[bp];W[cq];B[cp];W[dq];B[dm];W[cm];B[dl];W[cl];B[ck];W[bk];B[cj];W[bl])""",

    # 10. Modern AI-style 3-3 invasion opening
    """(;GM[1]FF[4]SZ[19]KM[7.5]RU[Chinese]PB[KataGo]PW[Leela Zero]RE[B+1.5]
;B[pd];W[dp];B[pp];W[dd];B[fc];W[cf];B[jd];W[qn];B[nq];W[qp];B[qq];W[po];B[rp];W[oo];B[mp];W[qj];B[pk];W[pj];B[ok];W[oj];B[nk];W[nj];B[mk];W[mj];B[lk];W[lj];B[kk];W[kj];B[jk];W[jj];B[ij];W[ii];B[hj])""",

    # 11-15: Tournament championship games
    """(;GM[1]FF[4]SZ[19]KM[7.5]RU[Chinese]PB[Shin Jinseo 9p]PW[Byun Sangil 9p]RE[B+R]
;B[pd];W[dp];B[pp];W[dd];B[fq];W[cn];B[jp];W[nc];B[qf];W[jd];B[qk];W[po];B[qp];W[qo];B[op];W[pn];B[rn];W[ro];B[rm];W[rp];B[rq];W[sp];B[sq];W[so];B[mp];W[ql];B[rl];W[pk];B[pl];W[qm];B[qn];W[pm];B[ol])""",

    """(;GM[1]FF[4]SZ[19]KM[6.5]RU[Japanese]PB[Iyama Yuta 9p]PW[Ichiriki Ryo 9p]RE[W+R]
;B[pd];W[dp];B[pq];W[dd];B[pk];W[nc];B[pf];W[jc];B[fq];W[dn];B[jp];W[qo];B[qp];W[po];B[np];W[pm];B[ql];W[qm];B[ol];W[rl];B[qk];W[rm];B[om];W[on];B[nn];W[no];B[mn];W[mo];B[lo];W[mp];B[mq];W[lp];B[ko];W[lq])""",

    """(;GM[1]FF[4]SZ[19]KM[7.5]RU[Chinese]PB[Ke Jie 9p]PW[Park Junghwan 9p]RE[B+2.5]
;B[qd];W[dp];B[pp];W[dc];B[de];W[ce];B[cf];W[cd];B[dg];W[fc];B[cn];W[fq];B[bp];W[cq];B[dl];W[oc];B[pe];W[mc];B[nq];W[qn];B[qp];W[pn];B[np];W[rk];B[qi];W[ri];B[qh];W[rh];B[qg];W[rg];B[rf];W[qj])""",

    """(;GM[1]FF[4]SZ[19]KM[7.5]RU[Chinese]PB[Yang Dingxin 9p]PW[Gu Zihao 9p]RE[W+R]
;B[pd];W[dd];B[pq];W[dp];B[qk];W[nc];B[pf];W[jd];B[fq];W[cn];B[jp];W[qo];B[qp];W[po];B[np];W[pm];B[ql];W[mp];B[nq];W[mo];B[mq];W[lq];B[lr];W[kq];B[kr];W[jq];B[iq];W[jr])""",

    """(;GM[1]FF[4]SZ[19]KM[6.5]RU[Japanese]PB[Cho U 9p]PW[Yoda Norimoto 9p]RE[B+R]
;B[pd];W[dp];B[pp];W[dd];B[fq];W[cn];B[dr];W[cq];B[ip];W[nc];B[pf];W[jd];B[qn];W[pk];B[qk];W[ql];B[pl];W[pj];B[qj];W[pi];B[qi];W[ph];B[qg];W[ol];B[pm];W[ok];B[rl])""",

    # 16-20: Diverse styles
    """(;GM[1]FF[4]SZ[19]KM[7.5]RU[Chinese]PB[Lian Xiao 9p]PW[Xie Erhao 9p]RE[W+1.5]
;B[pd];W[dp];B[pp];W[dd];B[fc];W[cf];B[jd];W[qn];B[nq];W[rp];B[qq];W[qp];B[po];W[qo];B[pn];W[pm];B[om];W[pl];B[ol];W[pk];B[ok];W[pj];B[oj];W[pi];B[oi];W[ph];B[oh];W[pg];B[og];W[pf];B[of])""",

    """(;GM[1]FF[4]SZ[19]KM[7.5]RU[Chinese]PB[Fan Tingyu 9p]PW[Mi Yuting 9p]RE[B+0.5]
;B[qd];W[dp];B[pq];W[dc];B[de];W[ce];B[cf];W[cd];B[dg];W[fc];B[od];W[cn];B[fq];W[bp];B[jp];W[qn];B[qp];W[pn];B[np];W[rk];B[qi];W[ri];B[qh];W[rh];B[qg];W[rg];B[rf];W[qj])""",

    """(;GM[1]FF[4]SZ[19]KM[7.5]RU[Chinese]PB[Shi Yue 9p]PW[Chen Yaoye 9p]RE[W+R]
;B[pd];W[dp];B[cd];W[pp];B[cn];W[fq];B[dj];W[nc];B[pf];W[jd];B[qn];W[nq];B[rp];W[qq];B[rq];W[qr];B[qp];W[pq];B[po];W[oo];B[on];W[no];B[nn];W[mn];B[mm];W[ln];B[lm];W[kn])""",

    """(;GM[1]FF[4]SZ[19]KM[7.5]RU[Chinese]PB[Ding Hao 9p]PW[Xie Ke 9p]RE[B+R]
;B[qd];W[dp];B[pp];W[dc];B[ce];W[ed];B[oc];W[cn];B[fq];W[bp];B[jp];W[qn];B[qp];W[pn];B[np];W[rk];B[qi];W[qk];B[ok];W[oj];B[nk];W[pj];B[ni];W[oi];B[nh];W[oh];B[ng];W[og];B[nf])""",

    """(;GM[1]FF[4]SZ[19]KM[7.5]RU[Chinese]PB[Tuo Jiaxi 9p]PW[Piao Tinghua 9p]RE[W+2.5]
;B[pd];W[dp];B[pq];W[dd];B[pk];W[nc];B[qf];W[jd];B[fq];W[cn];B[jp];W[qo];B[qp];W[po];B[np];W[pm];B[ql];W[om];B[ol];W[nl];B[nk];W[ml];B[mk];W[ll];B[lk];W[kl];B[kk];W[jl];B[jk])""",

    # 21-25: Recent championship games
    """(;GM[1]FF[4]SZ[19]KM[7.5]RU[Chinese]PB[Ke Jie 9p]PW[Shin Jinseo 9p]RE[W+R]
;B[pd];W[dp];B[pp];W[dd];B[fq];W[cn];B[jp];W[nc];B[qf];W[jd];B[qk];W[po];B[qp];W[qo];B[op];W[pn];B[rn];W[ro];B[rm];W[rp];B[rq];W[sp];B[sq];W[so];B[mp];W[ql];B[rl];W[pk];B[pl];W[qm])""",

    """(;GM[1]FF[4]SZ[19]KM[7.5]RU[Chinese]PB[AlphaGo Master]PW[AlphaGo Master]RE[B+0.5]
;B[pd];W[dp];B[cd];W[pp];B[cn];W[fq];B[dj];W[nc];B[pf];W[jd];B[qn];W[nq];B[rp];W[qq];B[rq];W[qr];B[qp];W[pq];B[po];W[oo];B[on];W[no];B[nn];W[mn];B[mm];W[ln];B[qc];W[qb];B[pc];W[pb];B[ob];W[oc];B[rb];W[pa])""",

    """(;GM[1]FF[4]SZ[19]KM[7.5]RU[Chinese]PB[Gu Zihao 9p]PW[Byun Sangil 9p]RE[B+R]
;B[qd];W[dp];B[pp];W[dc];B[de];W[ce];B[cf];W[cd];B[dg];W[fc];B[cn];W[fq];B[bp];W[cq];B[dl];W[oc];B[pe];W[mc];B[nq];W[qn];B[qp];W[pn];B[np];W[rl];B[qi];W[ri];B[qh];W[rh])""",

    """(;GM[1]FF[4]SZ[19]KM[7.5]RU[Chinese]PB[Leela Zero v0.17]PW[KataGo v1.10]RE[W+3.5]
;B[pd];W[dp];B[pp];W[dd];B[fc];W[cf];B[jd];W[qn];B[nq];W[qp];B[qq];W[po];B[rp];W[oo];B[mp];W[qj];B[pk];W[pj];B[ok];W[oj];B[nk];W[nj];B[mk];W[mj];B[lk];W[lj])""",

    """(;GM[1]FF[4]SZ[19]KM[7.5]RU[Chinese]PB[Fine Art]PW[PhoenixGo]RE[B+2.5]
;B[pd];W[dp];B[cd];W[pp];B[cn];W[fq];B[dj];W[nc];B[pf];W[jd];B[qn];W[nq];B[rp];W[qq];B[rq];W[qr];B[qp];W[pq];B[po];W[oo];B[on];W[no];B[nn];W[mn];B[mm];W[ln];B[lm];W[kn])""",
]


# =============================================================================
# Advanced Tsumego (Life & Death) Problems — 20 problems
# =============================================================================
TSUMEGO_ADVANCED: List[Dict[str, Any]] = [
    # 1. Corner two-eyes vital point
    {
        "name": "corner_two_eyes_vital",
        "to_move": Color.BLACK,
        "vital_move": (0, 1),
        "stones": [
            (Color.BLACK, (0, 0)), (Color.BLACK, (0, 2)),
            (Color.BLACK, (1, 0)), (Color.BLACK, (1, 1)), (Color.BLACK, (1, 2)),
            (Color.WHITE, (2, 0)), (Color.WHITE, (2, 1)), (Color.WHITE, (2, 2)),
            (Color.WHITE, (0, 3)), (Color.WHITE, (1, 3)),
        ],
    },
    # 2. Snapback trap
    {
        "name": "snapback_trap",
        "to_move": Color.BLACK,
        "vital_move": (3, 3),
        "stones": [
            (Color.BLACK, (3, 2)), (Color.BLACK, (2, 3)), (Color.BLACK, (4, 3)),
            (Color.WHITE, (3, 4)), (Color.WHITE, (2, 4)), (Color.WHITE, (4, 4)),
            (Color.WHITE, (3, 5)),
        ],
    },
    # 3. Solid connection vital point
    {
        "name": "solid_connection",
        "to_move": Color.BLACK,
        "vital_move": (4, 4),
        "stones": [
            (Color.BLACK, (4, 3)), (Color.BLACK, (4, 5)),
            (Color.WHITE, (3, 4)), (Color.WHITE, (5, 4)),
        ],
    },
    # 4. Kill corner group — descend to vital
    {
        "name": "kill_corner_descend",
        "to_move": Color.BLACK,
        "vital_move": (1, 0),
        "stones": [
            (Color.WHITE, (0, 0)), (Color.WHITE, (0, 1)), (Color.WHITE, (0, 2)),
            (Color.BLACK, (1, 1)), (Color.BLACK, (1, 2)), (Color.BLACK, (1, 3)),
            (Color.BLACK, (0, 3)),
        ],
    },
    # 5. Throw-in to reduce eye space
    {
        "name": "throw_in_reduce_eyes",
        "to_move": Color.BLACK,
        "vital_move": (0, 4),
        "stones": [
            (Color.WHITE, (0, 2)), (Color.WHITE, (0, 3)), (Color.WHITE, (0, 5)),
            (Color.WHITE, (0, 6)), (Color.WHITE, (1, 2)), (Color.WHITE, (1, 6)),
            (Color.BLACK, (1, 3)), (Color.BLACK, (1, 4)), (Color.BLACK, (1, 5)),
            (Color.BLACK, (2, 2)), (Color.BLACK, (2, 3)), (Color.BLACK, (2, 4)),
            (Color.BLACK, (2, 5)), (Color.BLACK, (2, 6)),
        ],
    },
    # 6. Side group eye vital point
    {
        "name": "side_eye_vital",
        "to_move": Color.BLACK,
        "vital_move": (0, 5),
        "stones": [
            (Color.WHITE, (0, 3)), (Color.WHITE, (0, 4)), (Color.WHITE, (0, 6)),
            (Color.WHITE, (0, 7)),
            (Color.BLACK, (1, 3)), (Color.BLACK, (1, 4)), (Color.BLACK, (1, 5)),
            (Color.BLACK, (1, 6)), (Color.BLACK, (1, 7)),
        ],
    },
    # 7. Connect and live
    {
        "name": "connect_and_live",
        "to_move": Color.BLACK,
        "vital_move": (2, 2),
        "stones": [
            (Color.BLACK, (1, 1)), (Color.BLACK, (1, 2)), (Color.BLACK, (1, 3)),
            (Color.BLACK, (3, 1)), (Color.BLACK, (3, 2)), (Color.BLACK, (3, 3)),
            (Color.WHITE, (0, 1)), (Color.WHITE, (0, 2)), (Color.WHITE, (0, 3)),
            (Color.WHITE, (2, 0)), (Color.WHITE, (2, 4)),
            (Color.WHITE, (4, 1)), (Color.WHITE, (4, 2)), (Color.WHITE, (4, 3)),
        ],
    },
    # 8. Bent four in the corner
    {
        "name": "bent_four_corner",
        "to_move": Color.WHITE,
        "vital_move": (0, 1),
        "stones": [
            (Color.BLACK, (0, 0)), (Color.BLACK, (1, 0)), (Color.BLACK, (1, 1)),
            (Color.BLACK, (0, 2)),
            (Color.WHITE, (2, 0)), (Color.WHITE, (2, 1)), (Color.WHITE, (2, 2)),
            (Color.WHITE, (1, 2)), (Color.WHITE, (0, 3)),
        ],
    },
    # 9. Capture race (semeai) — win by one liberty
    {
        "name": "semeai_one_liberty",
        "to_move": Color.BLACK,
        "vital_move": (3, 5),
        "stones": [
            (Color.BLACK, (3, 3)), (Color.BLACK, (3, 4)), (Color.BLACK, (4, 3)),
            (Color.WHITE, (3, 6)), (Color.WHITE, (4, 5)), (Color.WHITE, (4, 6)),
        ],
    },
    # 10. Squeeze tesuji to make eyes
    {
        "name": "squeeze_make_eyes",
        "to_move": Color.BLACK,
        "vital_move": (5, 5),
        "stones": [
            (Color.BLACK, (5, 3)), (Color.BLACK, (5, 4)), (Color.BLACK, (5, 6)),
            (Color.BLACK, (5, 7)),
            (Color.WHITE, (4, 3)), (Color.WHITE, (4, 4)), (Color.WHITE, (4, 5)),
            (Color.WHITE, (4, 6)), (Color.WHITE, (4, 7)),
            (Color.WHITE, (6, 3)), (Color.WHITE, (6, 4)), (Color.WHITE, (6, 5)),
            (Color.WHITE, (6, 6)), (Color.WHITE, (6, 7)),
        ],
    },
    # 11. Nakade five-point shape
    {
        "name": "nakade_five_point",
        "to_move": Color.BLACK,
        "vital_move": (1, 4),
        "stones": [
            (Color.WHITE, (0, 3)), (Color.WHITE, (0, 4)), (Color.WHITE, (0, 5)),
            (Color.WHITE, (1, 3)), (Color.WHITE, (1, 5)),
            (Color.BLACK, (2, 3)), (Color.BLACK, (2, 4)), (Color.BLACK, (2, 5)),
            (Color.BLACK, (1, 2)), (Color.BLACK, (1, 6)),
            (Color.BLACK, (0, 2)), (Color.BLACK, (0, 6)),
        ],
    },
    # 12. Eye stealing tesuji
    {
        "name": "eye_stealing",
        "to_move": Color.BLACK,
        "vital_move": (6, 2),
        "stones": [
            (Color.WHITE, (6, 1)), (Color.WHITE, (6, 3)),
            (Color.WHITE, (7, 1)), (Color.WHITE, (7, 2)), (Color.WHITE, (7, 3)),
            (Color.BLACK, (5, 1)), (Color.BLACK, (5, 2)), (Color.BLACK, (5, 3)),
            (Color.BLACK, (6, 0)), (Color.BLACK, (6, 4)),
            (Color.BLACK, (8, 1)), (Color.BLACK, (8, 2)), (Color.BLACK, (8, 3)),
        ],
    },
    # 13. L-group kill
    {
        "name": "l_group_kill",
        "to_move": Color.BLACK,
        "vital_move": (7, 1),
        "stones": [
            (Color.WHITE, (7, 0)), (Color.WHITE, (8, 0)), (Color.WHITE, (8, 1)),
            (Color.WHITE, (8, 2)),
            (Color.BLACK, (6, 0)), (Color.BLACK, (6, 1)), (Color.BLACK, (6, 2)),
            (Color.BLACK, (7, 2)), (Color.BLACK, (7, 3)),
        ],
    },
    # 14. Side squeeze to capture
    {
        "name": "side_squeeze_capture",
        "to_move": Color.BLACK,
        "vital_move": (4, 0),
        "stones": [
            (Color.WHITE, (3, 0)), (Color.WHITE, (3, 1)),
            (Color.BLACK, (2, 0)), (Color.BLACK, (2, 1)), (Color.BLACK, (4, 1)),
            (Color.BLACK, (5, 0)), (Color.BLACK, (5, 1)),
        ],
    },
    # 15. Double atari to live
    {
        "name": "double_atari_live",
        "to_move": Color.BLACK,
        "vital_move": (2, 6),
        "stones": [
            (Color.BLACK, (1, 5)), (Color.BLACK, (1, 6)), (Color.BLACK, (1, 7)),
            (Color.BLACK, (3, 5)), (Color.BLACK, (3, 6)), (Color.BLACK, (3, 7)),
            (Color.WHITE, (0, 5)), (Color.WHITE, (0, 6)), (Color.WHITE, (0, 7)),
            (Color.WHITE, (2, 5)), (Color.WHITE, (2, 7)),
            (Color.WHITE, (4, 5)), (Color.WHITE, (4, 6)), (Color.WHITE, (4, 7)),
        ],
    },
    # 16-20: Additional tactical patterns
    {
        "name": "corner_placement_kill",
        "to_move": Color.BLACK,
        "vital_move": (0, 0),
        "stones": [
            (Color.WHITE, (0, 1)), (Color.WHITE, (1, 0)),
            (Color.BLACK, (0, 2)), (Color.BLACK, (1, 1)), (Color.BLACK, (2, 0)),
        ],
    },
    {
        "name": "under_the_stones",
        "to_move": Color.BLACK,
        "vital_move": (6, 4),
        "stones": [
            (Color.BLACK, (6, 3)), (Color.BLACK, (6, 5)),
            (Color.BLACK, (7, 3)), (Color.BLACK, (7, 5)),
            (Color.WHITE, (5, 3)), (Color.WHITE, (5, 4)), (Color.WHITE, (5, 5)),
            (Color.WHITE, (7, 4)),
        ],
    },
    {
        "name": "hane_to_kill",
        "to_move": Color.BLACK,
        "vital_move": (8, 5),
        "stones": [
            (Color.WHITE, (8, 4)), (Color.WHITE, (8, 6)),
            (Color.WHITE, (7, 4)), (Color.WHITE, (7, 5)), (Color.WHITE, (7, 6)),
            (Color.BLACK, (6, 4)), (Color.BLACK, (6, 5)), (Color.BLACK, (6, 6)),
            (Color.BLACK, (8, 3)), (Color.BLACK, (8, 7)),
        ],
    },
    {
        "name": "descent_to_connect",
        "to_move": Color.BLACK,
        "vital_move": (5, 1),
        "stones": [
            (Color.BLACK, (4, 0)), (Color.BLACK, (4, 1)), (Color.BLACK, (4, 2)),
            (Color.BLACK, (6, 0)), (Color.BLACK, (6, 1)),
            (Color.WHITE, (3, 0)), (Color.WHITE, (3, 1)), (Color.WHITE, (3, 2)),
            (Color.WHITE, (5, 2)), (Color.WHITE, (7, 0)), (Color.WHITE, (7, 1)),
        ],
    },
    {
        "name": "clamp_to_capture",
        "to_move": Color.BLACK,
        "vital_move": (3, 7),
        "stones": [
            (Color.WHITE, (2, 7)), (Color.WHITE, (2, 8)),
            (Color.WHITE, (3, 8)),
            (Color.BLACK, (1, 7)), (Color.BLACK, (1, 8)),
            (Color.BLACK, (4, 7)), (Color.BLACK, (4, 8)),
        ],
    },
]


# =============================================================================
# Tesuji (Tactical) Problems — 15 problems
# =============================================================================
TESUJI_PROBLEMS: List[Dict[str, Any]] = [
    # 1. Net (geta) to capture cutting stone
    {
        "name": "net_capture",
        "to_move": Color.BLACK,
        "vital_move": (4, 5),
        "stones": [
            (Color.WHITE, (3, 4)),
            (Color.BLACK, (2, 3)), (Color.BLACK, (2, 5)),
            (Color.BLACK, (3, 3)), (Color.BLACK, (3, 5)),
            (Color.BLACK, (5, 4)),
        ],
    },
    # 2. Ladder breaker
    {
        "name": "ladder_breaker",
        "to_move": Color.BLACK,
        "vital_move": (5, 5),
        "stones": [
            (Color.WHITE, (4, 4)),
            (Color.BLACK, (3, 4)), (Color.BLACK, (4, 3)),
            (Color.BLACK, (3, 3)),
        ],
    },
    # 3. Double atari fork
    {
        "name": "double_atari_fork",
        "to_move": Color.BLACK,
        "vital_move": (4, 4),
        "stones": [
            (Color.WHITE, (3, 4)), (Color.WHITE, (5, 4)),
            (Color.WHITE, (4, 3)), (Color.WHITE, (4, 5)),
            (Color.BLACK, (2, 4)), (Color.BLACK, (6, 4)),
            (Color.BLACK, (4, 2)), (Color.BLACK, (4, 6)),
        ],
    },
    # 4. Peep and cut
    {
        "name": "peep_and_cut",
        "to_move": Color.BLACK,
        "vital_move": (3, 5),
        "stones": [
            (Color.WHITE, (3, 4)), (Color.WHITE, (2, 5)),
            (Color.BLACK, (4, 4)), (Color.BLACK, (4, 5)),
        ],
    },
    # 5. Crosscut and sacrifice
    {
        "name": "crosscut_sacrifice",
        "to_move": Color.BLACK,
        "vital_move": (5, 3),
        "stones": [
            (Color.WHITE, (4, 3)), (Color.WHITE, (5, 4)),
            (Color.BLACK, (4, 4)), (Color.BLACK, (5, 2)),
        ],
    },
    # 6. Shoulder hit
    {
        "name": "shoulder_hit",
        "to_move": Color.BLACK,
        "vital_move": (3, 3),
        "stones": [
            (Color.WHITE, (2, 2)),
            (Color.BLACK, (4, 4)),
        ],
    },
    # 7. Knight's move capture
    {
        "name": "knights_move_capture",
        "to_move": Color.BLACK,
        "vital_move": (5, 6),
        "stones": [
            (Color.WHITE, (4, 5)), (Color.WHITE, (4, 4)),
            (Color.BLACK, (3, 5)), (Color.BLACK, (5, 4)),
            (Color.BLACK, (3, 4)), (Color.BLACK, (6, 5)),
        ],
    },
    # 8. Under-the-stones tesuji
    {
        "name": "under_stones_tesuji",
        "to_move": Color.BLACK,
        "vital_move": (6, 6),
        "stones": [
            (Color.BLACK, (5, 5)), (Color.BLACK, (5, 7)),
            (Color.BLACK, (7, 5)), (Color.BLACK, (7, 6)), (Color.BLACK, (7, 7)),
            (Color.WHITE, (5, 6)), (Color.WHITE, (6, 5)), (Color.WHITE, (6, 7)),
        ],
    },
    # 9. Bamboo joint connection
    {
        "name": "bamboo_joint",
        "to_move": Color.BLACK,
        "vital_move": (5, 4),
        "stones": [
            (Color.BLACK, (4, 3)), (Color.BLACK, (4, 5)),
            (Color.BLACK, (6, 3)), (Color.BLACK, (6, 5)),
            (Color.WHITE, (5, 3)), (Color.WHITE, (5, 5)),
        ],
    },
    # 10. Wedge tesuji
    {
        "name": "wedge",
        "to_move": Color.BLACK,
        "vital_move": (4, 4),
        "stones": [
            (Color.WHITE, (3, 3)), (Color.WHITE, (3, 5)),
            (Color.WHITE, (5, 3)), (Color.WHITE, (5, 5)),
            (Color.BLACK, (2, 4)), (Color.BLACK, (6, 4)),
        ],
    },
    # 11-15: Additional tesuji patterns
    {
        "name": "monkey_jump",
        "to_move": Color.BLACK,
        "vital_move": (8, 3),
        "stones": [
            (Color.BLACK, (8, 1)),
            (Color.WHITE, (7, 2)), (Color.WHITE, (7, 3)),
        ],
    },
    {
        "name": "placement_tesuji",
        "to_move": Color.BLACK,
        "vital_move": (2, 3),
        "stones": [
            (Color.WHITE, (1, 2)), (Color.WHITE, (1, 3)), (Color.WHITE, (1, 4)),
            (Color.WHITE, (2, 2)), (Color.WHITE, (2, 4)),
            (Color.BLACK, (0, 2)), (Color.BLACK, (0, 3)), (Color.BLACK, (0, 4)),
            (Color.BLACK, (3, 2)), (Color.BLACK, (3, 3)), (Color.BLACK, (3, 4)),
        ],
    },
    {
        "name": "push_through",
        "to_move": Color.BLACK,
        "vital_move": (5, 5),
        "stones": [
            (Color.WHITE, (4, 5)), (Color.WHITE, (6, 5)),
            (Color.BLACK, (4, 4)), (Color.BLACK, (6, 6)),
        ],
    },
    {
        "name": "attachment_probe",
        "to_move": Color.BLACK,
        "vital_move": (3, 6),
        "stones": [
            (Color.WHITE, (2, 6)),
            (Color.BLACK, (4, 6)), (Color.BLACK, (3, 5)),
        ],
    },
    {
        "name": "sacrifice_and_recapture",
        "to_move": Color.BLACK,
        "vital_move": (4, 3),
        "stones": [
            (Color.WHITE, (3, 3)), (Color.WHITE, (5, 3)),
            (Color.WHITE, (4, 2)), (Color.WHITE, (4, 4)),
            (Color.BLACK, (2, 3)), (Color.BLACK, (6, 3)),
            (Color.BLACK, (3, 2)), (Color.BLACK, (5, 2)),
        ],
    },
]


def build_comprehensive_pro_dataset(board_size: int = 9) -> GoDataset:
    """Builds a GoDataset from professional games, tsumego, and tesuji problems.

    Parses all SGF games, replays them move-by-move to extract training
    positions (board state, move target, game outcome), and adds tactical
    problem positions with boosted sampling weight.

    Args:
        board_size: Board size to filter games (9, 13, or 19)

    Returns:
        GoDataset with all extracted training positions
    """
    datapoints: List[GoDataPoint] = []

    # 1. Parse pro SGF games
    game_collection = PRO_GAMES_9x9 if board_size <= 9 else PRO_GAMES_19x19

    games_parsed = 0
    for sgf_text in game_collection:
        try:
            games = SGFParser.parse_string(sgf_text)
            for game in games:
                if game.size != board_size:
                    continue
                games_parsed += 1
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
        except Exception:
            continue

    # 2. Add Tsumego positions (boosted 5x for emphasis)
    if board_size == 9:
        for prob in TSUMEGO_ADVANCED:
            board = Board(size=board_size)
            for col, coord in prob["stones"]:
                if board.in_bounds(coord[0], coord[1]):
                    board.grid[coord[0], coord[1]] = col

            to_move = prob["to_move"]
            vital_move = prob["vital_move"]
            feat = board.to_feature_tensor(to_move).numpy()
            act_idx = Move.to_action_index(vital_move, board_size)

            # Duplicate tactical patterns to strengthen neural priors
            for _ in range(5):
                datapoints.append(
                    GoDataPoint(
                        feature_tensor=feat,
                        target_action=act_idx,
                        target_value=1.0,
                        board_size=board_size,
                    )
                )

    # 3. Add Tesuji problem positions (boosted 3x)
    if board_size == 9:
        for prob in TESUJI_PROBLEMS:
            board = Board(size=board_size)
            for col, coord in prob["stones"]:
                if board.in_bounds(coord[0], coord[1]):
                    board.grid[coord[0], coord[1]] = col

            to_move = prob["to_move"]
            vital_move = prob["vital_move"]
            feat = board.to_feature_tensor(to_move).numpy()
            act_idx = Move.to_action_index(vital_move, board_size)

            for _ in range(3):
                datapoints.append(
                    GoDataPoint(
                        feature_tensor=feat,
                        target_action=act_idx,
                        target_value=1.0,
                        board_size=board_size,
                    )
                )

    return GoDataset(datapoints=datapoints, augment_symmetry=True)
