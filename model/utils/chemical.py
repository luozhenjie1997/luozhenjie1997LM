import torch
import enum
import numpy as np

##
# Residue identities
"""
This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2013 Stanford University and the Authors.
Authors: Peter Eastman
Contributors:

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
non_standard_residue_substitutions = {
    '2AS':'ASP', '3AH':'HIS', '5HP':'GLU', 'ACL':'ARG', 'AGM':'ARG', 'AIB':'ALA', 'ALM':'ALA', 'ALO':'THR', 'ALY':'LYS', 'ARM':'ARG',
    'ASA':'ASP', 'ASB':'ASP', 'ASK':'ASP', 'ASL':'ASP', 'ASQ':'ASP', 'AYA':'ALA', 'BCS':'CYS', 'BHD':'ASP', 'BMT':'THR', 'BNN':'ALA',
    'BUC':'CYS', 'BUG':'LEU', 'C5C':'CYS', 'C6C':'CYS', 'CAS':'CYS', 'CCS':'CYS', 'CEA':'CYS', 'CGU':'GLU', 'CHG':'ALA', 'CLE':'LEU', 'CME':'CYS',
    'CSD':'ALA', 'CSO':'CYS', 'CSP':'CYS', 'CSS':'CYS', 'CSW':'CYS', 'CSX':'CYS', 'CXM':'MET', 'CY1':'CYS', 'CY3':'CYS', 'CYG':'CYS',
    'CYM':'CYS', 'CYQ':'CYS', 'DAH':'PHE', 'DAL':'ALA', 'DAR':'ARG', 'DAS':'ASP', 'DCY':'CYS', 'DGL':'GLU', 'DGN':'GLN', 'DHA':'ALA',
    'DHI':'HIS', 'DIL':'ILE', 'DIV':'VAL', 'DLE':'LEU', 'DLY':'LYS', 'DNP':'ALA', 'DPN':'PHE', 'DPR':'PRO', 'DSN':'SER', 'DSP':'ASP',
    'DTH':'THR', 'DTR':'TRP', 'DTY':'TYR', 'DVA':'VAL', 'EFC':'CYS', 'FLA':'ALA', 'FME':'MET', 'GGL':'GLU', 'GL3':'GLY', 'GLZ':'GLY',
    'GMA':'GLU', 'GSC':'GLY', 'HAC':'ALA', 'HAR':'ARG', 'HIC':'HIS', 'HIP':'HIS', 'HMR':'ARG', 'HPQ':'PHE', 'HTR':'TRP', 'HYP':'PRO',
    'IAS':'ASP', 'IIL':'ILE', 'IYR':'TYR', 'KCX':'LYS', 'LLP':'LYS', 'LLY':'LYS', 'LTR':'TRP', 'LYM':'LYS', 'LYZ':'LYS', 'MAA':'ALA', 'MEN':'ASN',
    'MHS':'HIS', 'MIS':'SER', 'MLE':'LEU', 'MPQ':'GLY', 'MSA':'GLY', 'MSE':'MET', 'MVA':'VAL', 'NEM':'HIS', 'NEP':'HIS', 'NLE':'LEU',
    'NLN':'LEU', 'NLP':'LEU', 'NMC':'GLY', 'OAS':'SER', 'OCS':'CYS', 'OMT':'MET', 'PAQ':'TYR', 'PCA':'GLU', 'PEC':'CYS', 'PHI':'PHE',
    'PHL':'PHE', 'PR3':'CYS', 'PRR':'ALA', 'PTR':'TYR', 'PYX':'CYS', 'SAC':'SER', 'SAR':'GLY', 'SCH':'CYS', 'SCS':'CYS', 'SCY':'CYS',
    'SEL':'SER', 'SEP':'SER', 'SET':'SER', 'SHC':'CYS', 'SHR':'LYS', 'SMC':'CYS', 'SOC':'CYS', 'STY':'TYR', 'SVA':'SER', 'TIH':'ALA',
    'TPL':'TRP', 'TPO':'THR', 'TPQ':'ALA', 'TRG':'LYS', 'TRO':'TRP', 'TYB':'TYR', 'TYI':'TYR', 'TYQ':'TYR', 'TYS':'TYR', 'TYY':'TYR'
}


"""词汇表索引"""
class AA(enum.IntEnum):
    A = 0; C = 1; D = 2; E = 3; F = 4; G = 5; H = 6; I = 7; K = 8; L = 9; M = 10;N = 11; P = 12; Q = 13; R = 14; S = 15
    T = 16; V = 17; W = 18; Y = 19; X = 20


ressymb_to_resindex = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
    'X': 20,
}

resindex_to_ressymb = {v: k for k, v in ressymb_to_resindex.items()}

"""三字母缩写词汇表索引"""
class AA3Letter(enum.IntEnum):
    ALA = 0; CYS = 1; ASP = 2; GLU = 3; PHE = 4; GLY = 5; HIS = 6; ILE = 7; LYS = 8; LEU = 9; MET = 10; ASN = 11; PRO = 12
    GLN = 13; ARG = 14; SER = 15; THR = 16; VAL = 17; TRP = 18; TYR = 19; UNK = 20; PAD = 21

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str) and len(value) == 3:      # three representation
            if value in non_standard_residue_substitutions:
                value = non_standard_residue_substitutions[value]
            if value in cls._member_names_:
                return getattr(cls, value)
        elif isinstance(value, str) and len(value) == 1:    # one representation
            if value in ressymb_to_resindex:
                return cls(ressymb_to_resindex[value])

        return super()._missing_(value)

    def __str__(self):
        return self.name

    @classmethod
    def is_aa(cls, value):
        return (value in ressymb_to_resindex) or \
            (value in non_standard_residue_substitutions) or \
            (value in cls._member_names_) or \
            (value in cls._member_map_.values())


"""主链原子位置索引"""
class BBHeavyAtom(enum.IntEnum):
    N = 0; CA = 1; C = 2; O = 3; CB = 4; OXT = 14


"""CRR区域编号，分重链和轻链"""
class CDR(enum.IntEnum):
    H1 = 1
    H2 = 2
    H3 = 3
    L1 = 4
    L2 = 5
    L3 = 6


"""IMGT编号信息"""
class IMGTCDRRange:
    H1 = (27, 38)
    H2 = (56, 65)
    H3 = (105, 117)

    L1 = (27, 38)
    L2 = (56, 65)
    L3 = (105, 117)


"""Chothia编号信息"""
class ChothiaCDRRange:
    H1 = (26, 32)
    H2 = (52, 56)
    H3 = (95, 102)

    L1 = (24, 34)
    L2 = (50, 56)
    L3 = (89, 97)

    @classmethod
    def to_cdr(cls, chain_type, resseq):
        assert chain_type in ('H', 'L')
        if chain_type == 'H':
            if cls.H1[0] <= resseq <= cls.H1[1]:
                return CDR.H1
            elif cls.H2[0] <= resseq <= cls.H2[1]:
                return CDR.H2
            elif cls.H3[0] <= resseq <= cls.H3[1]:
                return CDR.H3
        elif chain_type == 'L':
            if cls.L1[0] <= resseq <= cls.L1[1]:  # Chothia VH-CDR1
                return CDR.L1
            elif cls.L2[0] <= resseq <= cls.L2[1]:
                return CDR.L2
            elif cls.L3[0] <= resseq <= cls.L3[1]:
                return CDR.L3


conversion_amino2index = {item.name: item.value for item in AA}
aa2num = {item: key for key, item in conversion_amino2index.items()}

max_num_heavyatoms = 14  # 3个主链原子+11个侧链原子
max_num_hydrogens = 16  # 氢原子的最大数量
max_num_allatoms = max_num_heavyatoms + max_num_hydrogens

BACKBONE_FRAME = 0
OMEGA_FRAME = 1
PHI_FRAME = 2
PSI_FRAME = 3
CHI1_FRAME, CHI2_FRAME, CHI3_FRAME, CHI4_FRAME = 4, 5, 6, 7

"""
完整的氨基酸原子表示 (Nx14)，表示氨基酸的重原子，按以下顺序排列：
"""
restype_to_heavyatom_names = {
    AA3Letter.ALA: ['N', 'CA', 'C', 'O', 'CB', '',    '',    '',    '',    '',    '',    '',    '',    ''],
    AA3Letter.ARG: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'NE',  'CZ',  'NH1', 'NH2', '',    '',    ''],
    AA3Letter.ASN: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'ND2', '',    '',    '',    '',    '',    ''],
    AA3Letter.ASP: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'OD2', '',    '',    '',    '',    '',    ''],
    AA3Letter.CYS: ['N', 'CA', 'C', 'O', 'CB', 'SG',  '',    '',    '',    '',    '',    '',    '',    ''],
    AA3Letter.GLN: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'NE2', '',    '',    '',    '',    ''],
    AA3Letter.GLU: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'OE2', '',    '',    '',    '',    ''],
    AA3Letter.GLY: ['N', 'CA', 'C', 'O', '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],
    AA3Letter.HIS: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'ND1', 'CD2', 'CE1', 'NE2', '',    '',    '',    ''],
    AA3Letter.ILE: ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '',    '',    '',    '',    '',    ''],
    AA3Letter.LEU: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', '',    '',    '',    '',    '',    ''],
    AA3Letter.LYS: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'CE',  'NZ',  '',    '',    '',    '',    ''],
    AA3Letter.MET: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'SD',  'CE',  '',    '',    '',    '',    '',    ''],
    AA3Letter.PHE: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  '',    '',    ''],
    AA3Letter.PRO: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  '',    '',    '',    '',    '',    '',    ''],
    AA3Letter.SER: ['N', 'CA', 'C', 'O', 'CB', 'OG',  '',    '',    '',    '',    '',    '',    '',    ''],
    AA3Letter.THR: ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    AA3Letter.TRP: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    AA3Letter.TYR: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  'OH',  '',    ''],
    AA3Letter.VAL: ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    AA3Letter.UNK: ['',  '',   '',  '',  '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],
    AA3Letter.PAD: ['',  '',   '',  '',  '',   '',    '',    '',    '',    '',    '',    '',    '',    '']
}

# 存在的二面角，不考虑与氢原子形成的平面
chi_angles_atoms = {
    AA3Letter.ALA: [],  # ala
    AA3Letter.ARG: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'NE'], ['CG', 'CD', 'NE', 'CZ']],  # arg
    AA3Letter.ASN: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],  # asn
    AA3Letter.ASP: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],  # asp
    AA3Letter.CYS: [['N', 'CA', 'CB', 'SG']],  # cys
    AA3Letter.GLN: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'OE1']],  # gln
    AA3Letter.GLU: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'OE1']],  # glu
    AA3Letter.GLY: [],  # gly
    AA3Letter.HIS: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']],  # his (protonation handled as a pseudo-torsion)
    AA3Letter.ILE: [['N', 'CA', 'CB', 'CG1'], ['CA', 'CB', 'CG1', 'CD1']],  # ile
    AA3Letter.LEU: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],  # leu
    AA3Letter.LYS: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'CE'], ['CG', 'CD', 'CE', 'NZ']],  # lys
    AA3Letter.MET: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'SD'], ['CB', 'CG', 'SD', 'CE']],  # met
    AA3Letter.PHE: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],  # phe
    AA3Letter.PRO: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD']],  # pro
    AA3Letter.SER: [['N', 'CA', 'CB', 'OG']],  # ser
    AA3Letter.THR: [['N', 'CA', 'CB', 'OG1']],  # thr
    AA3Letter.TRP: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],  # trp
    AA3Letter.TYR: [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],  # tyr
    AA3Letter.VAL: [['N', 'CA', 'CB', 'CG1']],  # val
    AA3Letter.UNK: [],  # unk
    AA3Letter.PAD: []
}

restype_atom14_name_to_index = {
    resname: {name: index for index, name in enumerate(atoms) if name is not None}
    for resname, atoms in restype_to_heavyatom_names.items()
}

# 每个氨基酸的各个原子的理想坐标
rigid_group_heavy_atom_positions = {
    AA3Letter.ALA: [
        ['N', 0, (-0.525, 1.363, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, -0.000, -0.000)],
        ['CB', 0, (-0.529, -0.774, -1.205)],
        ['O', 3, (0.627, 1.062, 0.000)],
    ],
    AA3Letter.ARG: [
        ['N', 0, (-0.524, 1.362, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, -0.000, -0.000)],
        ['CB', 0, (-0.524, -0.778, -1.209)],
        ['O', 3, (0.626, 1.062, 0.000)],
        ['CG', 4, (0.616, 1.390, -0.000)],
        ['CD', 5, (0.564, 1.414, 0.000)],
        ['NE', 6, (0.539, 1.357, -0.000)],
        ['NH1', 7, (0.206, 2.301, 0.000)],
        ['NH2', 7, (2.078, 0.978, -0.000)],
        ['CZ', 7, (0.758, 1.093, -0.000)],
    ],
    AA3Letter.ASN: [
        ['N', 0, (-0.536, 1.357, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, -0.000, -0.000)],
        ['CB', 0, (-0.531, -0.787, -1.200)],
        ['O', 3, (0.625, 1.062, 0.000)],
        ['CG', 4, (0.584, 1.399, 0.000)],
        ['ND2', 5, (0.593, -1.188, 0.001)],
        ['OD1', 5, (0.633, 1.059, 0.000)],
    ],
    AA3Letter.ASP: [
        ['N', 0, (-0.525, 1.362, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.527, 0.000, -0.000)],
        ['CB', 0, (-0.526, -0.778, -1.208)],
        ['O', 3, (0.626, 1.062, -0.000)],
        ['CG', 4, (0.593, 1.398, -0.000)],
        ['OD1', 5, (0.610, 1.091, 0.000)],
        ['OD2', 5, (0.592, -1.101, -0.003)],
    ],
    AA3Letter.CYS: [
        ['N', 0, (-0.522, 1.362, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.524, 0.000, 0.000)],
        ['CB', 0, (-0.519, -0.773, -1.212)],
        ['O', 3, (0.625, 1.062, -0.000)],
        ['SG', 4, (0.728, 1.653, 0.000)],
    ],
    AA3Letter.GLN: [
        ['N', 0, (-0.526, 1.361, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, 0.000, 0.000)],
        ['CB', 0, (-0.525, -0.779, -1.207)],
        ['O', 3, (0.626, 1.062, -0.000)],
        ['CG', 4, (0.615, 1.393, 0.000)],
        ['CD', 5, (0.587, 1.399, -0.000)],
        ['NE2', 6, (0.593, -1.189, -0.001)],
        ['OE1', 6, (0.634, 1.060, 0.000)],
    ],
    AA3Letter.GLU: [
        ['N', 0, (-0.528, 1.361, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, -0.000, -0.000)],
        ['CB', 0, (-0.526, -0.781, -1.207)],
        ['O', 3, (0.626, 1.062, 0.000)],
        ['CG', 4, (0.615, 1.392, 0.000)],
        ['CD', 5, (0.600, 1.397, 0.000)],
        ['OE1', 6, (0.607, 1.095, -0.000)],
        ['OE2', 6, (0.589, -1.104, -0.001)],
    ],
    AA3Letter.GLY: [
        ['N', 0, (-0.572, 1.337, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.517, -0.000, -0.000)],
        ['O', 3, (0.626, 1.062, -0.000)],
    ],
    AA3Letter.HIS: [
        ['N', 0, (-0.527, 1.360, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, 0.000, 0.000)],
        ['CB', 0, (-0.525, -0.778, -1.208)],
        ['O', 3, (0.625, 1.063, 0.000)],
        ['CG', 4, (0.600, 1.370, -0.000)],
        ['CD2', 5, (0.889, -1.021, 0.003)],
        ['ND1', 5, (0.744, 1.160, -0.000)],
        ['CE1', 5, (2.030, 0.851, 0.002)],
        ['NE2', 5, (2.145, -0.466, 0.004)],
    ],
    AA3Letter.ILE: [
        ['N', 0, (-0.493, 1.373, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.527, -0.000, -0.000)],
        ['CB', 0, (-0.536, -0.793, -1.213)],
        ['O', 3, (0.627, 1.062, -0.000)],
        ['CG1', 4, (0.534, 1.437, -0.000)],
        ['CG2', 4, (0.540, -0.785, -1.199)],
        ['CD1', 5, (0.619, 1.391, 0.000)],
    ],
    AA3Letter.LEU: [
        ['N', 0, (-0.520, 1.363, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, -0.000, -0.000)],
        ['CB', 0, (-0.522, -0.773, -1.214)],
        ['O', 3, (0.625, 1.063, -0.000)],
        ['CG', 4, (0.678, 1.371, 0.000)],
        ['CD1', 5, (0.530, 1.430, -0.000)],
        ['CD2', 5, (0.535, -0.774, 1.200)],
    ],
    AA3Letter.LYS: [
        ['N', 0, (-0.526, 1.362, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, 0.000, 0.000)],
        ['CB', 0, (-0.524, -0.778, -1.208)],
        ['O', 3, (0.626, 1.062, -0.000)],
        ['CG', 4, (0.619, 1.390, 0.000)],
        ['CD', 5, (0.559, 1.417, 0.000)],
        ['CE', 6, (0.560, 1.416, 0.000)],
        ['NZ', 7, (0.554, 1.387, 0.000)],
    ],
    AA3Letter.MET: [
        ['N', 0, (-0.521, 1.364, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, 0.000, 0.000)],
        ['CB', 0, (-0.523, -0.776, -1.210)],
        ['O', 3, (0.625, 1.062, -0.000)],
        ['CG', 4, (0.613, 1.391, -0.000)],
        ['SD', 5, (0.703, 1.695, 0.000)],
        ['CE', 6, (0.320, 1.786, -0.000)],
    ],
    AA3Letter.PHE: [
        ['N', 0, (-0.518, 1.363, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.524, 0.000, -0.000)],
        ['CB', 0, (-0.525, -0.776, -1.212)],
        ['O', 3, (0.626, 1.062, -0.000)],
        ['CG', 4, (0.607, 1.377, 0.000)],
        ['CD1', 5, (0.709, 1.195, -0.000)],
        ['CD2', 5, (0.706, -1.196, 0.000)],
        ['CE1', 5, (2.102, 1.198, -0.000)],
        ['CE2', 5, (2.098, -1.201, -0.000)],
        ['CZ', 5, (2.794, -0.003, -0.001)],
    ],
    AA3Letter.PRO: [
        ['N', 0, (-0.566, 1.351, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.527, -0.000, 0.000)],
        ['CB', 0, (-0.546, -0.611, -1.293)],
        ['O', 3, (0.621, 1.066, 0.000)],
        ['CG', 4, (0.382, 1.445, 0.0)],
        # ['CD', 5, (0.427, 1.440, 0.0)],
        ['CD', 5, (0.477, 1.424, 0.0)],  # manually made angle 2 degrees larger
    ],
    AA3Letter.SER: [
        ['N', 0, (-0.529, 1.360, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, -0.000, -0.000)],
        ['CB', 0, (-0.518, -0.777, -1.211)],
        ['O', 3, (0.626, 1.062, -0.000)],
        ['OG', 4, (0.503, 1.325, 0.000)],
    ],
    AA3Letter.THR: [
        ['N', 0, (-0.517, 1.364, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, 0.000, -0.000)],
        ['CB', 0, (-0.516, -0.793, -1.215)],
        ['O', 3, (0.626, 1.062, 0.000)],
        ['CG2', 4, (0.550, -0.718, -1.228)],
        ['OG1', 4, (0.472, 1.353, 0.000)],
    ],
    AA3Letter.TRP: [
        ['N', 0, (-0.521, 1.363, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, -0.000, 0.000)],
        ['CB', 0, (-0.523, -0.776, -1.212)],
        ['O', 3, (0.627, 1.062, 0.000)],
        ['CG', 4, (0.609, 1.370, -0.000)],
        ['CD1', 5, (0.824, 1.091, 0.000)],
        ['CD2', 5, (0.854, -1.148, -0.005)],
        ['CE2', 5, (2.186, -0.678, -0.007)],
        ['CE3', 5, (0.622, -2.530, -0.007)],
        ['NE1', 5, (2.140, 0.690, -0.004)],
        ['CH2', 5, (3.028, -2.890, -0.013)],
        ['CZ2', 5, (3.283, -1.543, -0.011)],
        ['CZ3', 5, (1.715, -3.389, -0.011)],
    ],
    AA3Letter.TYR: [
        ['N', 0, (-0.522, 1.362, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.524, -0.000, -0.000)],
        ['CB', 0, (-0.522, -0.776, -1.213)],
        ['O', 3, (0.627, 1.062, -0.000)],
        ['CG', 4, (0.607, 1.382, -0.000)],
        ['CD1', 5, (0.716, 1.195, -0.000)],
        ['CD2', 5, (0.713, -1.194, -0.001)],
        ['CE1', 5, (2.107, 1.200, -0.002)],
        ['CE2', 5, (2.104, -1.201, -0.003)],
        ['OH', 5, (4.168, -0.002, -0.005)],
        ['CZ', 5, (2.791, -0.001, -0.003)],
    ],
    AA3Letter.VAL: [
        ['N', 0, (-0.494, 1.373, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.527, -0.000, -0.000)],
        ['CB', 0, (-0.533, -0.795, -1.213)],
        ['O', 3, (0.627, 1.062, -0.000)],
        ['CG1', 4, (0.540, 1.429, -0.000)],
        ['CG2', 4, (0.533, -0.776, 1.203)],
    ],
}

# 理想的 N、CA、C 初始坐标
init_N = torch.tensor([-0.5272, 1.3593, 0.000]).float()
init_CA = torch.zeros_like(init_N)
init_C = torch.tensor([1.5233, 0.000, 0.000]).float()
INIT_CRDS = torch.full((14, 3), np.nan)
INIT_CRDS[:3] = torch.stack((init_N, init_CA, init_C), dim=0)  # (3,3)
norm_N = init_N / (torch.norm(init_N, dim=-1, keepdim=True) + 1e-5)
norm_C = init_C / (torch.norm(init_C, dim=-1, keepdim=True) + 1e-5)
cos_ideal_NCAC = torch.sum(norm_N * norm_C, dim=-1)  # 理想N-CA-C键角的余弦值

class Torsion(enum.IntEnum):
    Backbone = 0
    Omega = 1
    Phi = 2
    Psi = 3
    Chi1 = 4
    Chi2 = 5
    Chi3 = 6
    Chi7 = 7

# 标记是否存在二面角chi的掩码
chi_angles_mask = {
    AA3Letter.ALA: [False, False, False, False],  # ALA
    AA3Letter.ARG: [True , True , True , True ],  # ARG
    AA3Letter.ASN: [True , True , False, False],  # ASN
    AA3Letter.ASP: [True , True , False, False],  # ASP
    AA3Letter.CYS: [True , False, False, False],  # CYS
    AA3Letter.GLN: [True , True , True , False],  # GLN
    AA3Letter.GLU: [True , True , True , False],  # GLU
    AA3Letter.GLY: [False, False, False, False],  # GLY
    AA3Letter.HIS: [True , True , False, False],  # HIS
    AA3Letter.ILE: [True , True , False, False],  # ILE
    AA3Letter.LEU: [True , True , False, False],  # LEU
    AA3Letter.LYS: [True , True , True , True ],  # LYS
    AA3Letter.MET: [True , True , True , False],  # MET
    AA3Letter.PHE: [True , True , False, False],  # PHE
    AA3Letter.PRO: [True , True , False, False],  # PRO
    AA3Letter.SER: [True , False, False, False],  # SER
    AA3Letter.THR: [True , False, False, False],  # THR
    AA3Letter.TRP: [True , True , False, False],  # TRP
    AA3Letter.TYR: [True , True , False, False],  # TYR
    AA3Letter.VAL: [True , False, False, False],  # VAL
    AA3Letter.UNK: [False, False, False, False],  # UNK
}

chi_pi_periodic = {
    AA3Letter.ALA: [False, False, False, False],  # ALA
    AA3Letter.ARG: [False, False, False, False],  # ARG
    AA3Letter.ASN: [False, False, False, False],  # ASN
    AA3Letter.ASP: [False, True , False, False],  # ASP
    AA3Letter.CYS: [False, False, False, False],  # CYS
    AA3Letter.GLN: [False, False, False, False],  # GLN
    AA3Letter.GLU: [False, False, True , False],  # GLU
    AA3Letter.GLY: [False, False, False, False],  # GLY
    AA3Letter.HIS: [False, False, False, False],  # HIS
    AA3Letter.ILE: [False, False, False, False],  # ILE
    AA3Letter.LEU: [False, False, False, False],  # LEU
    AA3Letter.LYS: [False, False, False, False],  # LYS
    AA3Letter.MET: [False, False, False, False],  # MET
    AA3Letter.PHE: [False, True , False, False],  # PHE
    AA3Letter.PRO: [False, False, False, False],  # PRO
    AA3Letter.SER: [False, False, False, False],  # SER
    AA3Letter.THR: [False, False, False, False],  # THR
    AA3Letter.TRP: [False, False, False, False],  # TRP
    AA3Letter.TYR: [False, True , False, False],  # TYR
    AA3Letter.VAL: [False, False, False, False],  # VAL
    AA3Letter.UNK: [False, False, False, False],  # UNK
}

# 各氨基酸主链理想（初始）坐标
backbone_atom_coordinates = {
    AA3Letter.ALA: [
        (-0.525, 1.363, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.526, -0.0, -0.0),  # C
    ],
    AA3Letter.ARG: [
        (-0.524, 1.362, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.525, -0.0, -0.0),  # C
    ],
    AA3Letter.ASN: [
        (-0.536, 1.357, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.526, -0.0, -0.0),  # C
    ],
    AA3Letter.ASP: [
        (-0.525, 1.362, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.527, 0.0, -0.0),  # C
    ],
    AA3Letter.CYS: [
        (-0.522, 1.362, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.524, 0.0, 0.0),  # C
    ],
    AA3Letter.GLN: [
        (-0.526, 1.361, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.526, 0.0, 0.0),  # C
    ],
    AA3Letter.GLU: [
        (-0.528, 1.361, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.526, -0.0, -0.0),  # C
    ],
    AA3Letter.GLY: [
        (-0.572, 1.337, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.517, -0.0, -0.0),  # C
    ],
    AA3Letter.HIS: [
        (-0.527, 1.36, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.525, 0.0, 0.0),  # C
    ],
    AA3Letter.ILE: [
        (-0.493, 1.373, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.527, -0.0, -0.0),  # C
    ],
    AA3Letter.LEU: [
        (-0.52, 1.363, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.525, -0.0, -0.0),  # C
    ],
    AA3Letter.LYS: [
        (-0.526, 1.362, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.526, 0.0, 0.0),  # C
    ],
    AA3Letter.MET: [
        (-0.521, 1.364, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.525, 0.0, 0.0),  # C
    ],
    AA3Letter.PHE: [
        (-0.518, 1.363, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.524, 0.0, -0.0),  # C
    ],
    AA3Letter.PRO: [
        (-0.566, 1.351, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.527, -0.0, 0.0),  # C
    ],
    AA3Letter.SER: [
        (-0.529, 1.36, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.525, -0.0, -0.0),  # C
    ],
    AA3Letter.THR: [
        (-0.517, 1.364, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.526, 0.0, -0.0),  # C
    ],
    AA3Letter.TRP: [
        (-0.521, 1.363, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.525, -0.0, 0.0),  # C
    ],
    AA3Letter.TYR: [
        (-0.522, 1.362, 0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.524, -0.0, -0.0),  # C
    ],
    AA3Letter.VAL: [
        (-0.494, 1.373, -0.0),  # N
        (0.0, 0.0, 0.0),  # CA
        (1.527, -0.0, -0.0),  # C
    ],
}

bb_oxygen_coordinate_tensor = torch.zeros([21, 3])
bb_oxygen_coordinate = {
    AA3Letter.ALA: (2.153, -1.062, 0.0),
    AA3Letter.ARG: (2.151, -1.062, 0.0),
    AA3Letter.ASN: (2.151, -1.062, 0.0),
    AA3Letter.ASP: (2.153, -1.062, 0.0),
    AA3Letter.CYS: (2.149, -1.062, 0.0),
    AA3Letter.GLN: (2.152, -1.062, 0.0),
    AA3Letter.GLU: (2.152, -1.062, 0.0),
    AA3Letter.GLY: (2.143, -1.062, 0.0),
    AA3Letter.HIS: (2.15, -1.063, 0.0),
    AA3Letter.ILE: (2.154, -1.062, 0.0),
    AA3Letter.LEU: (2.15, -1.063, 0.0),
    AA3Letter.LYS: (2.152, -1.062, 0.0),
    AA3Letter.MET: (2.15, -1.062, 0.0),
    AA3Letter.PHE: (2.15, -1.062, 0.0),
    AA3Letter.PRO: (2.148, -1.066, 0.0),
    AA3Letter.SER: (2.151, -1.062, 0.0),
    AA3Letter.THR: (2.152, -1.062, 0.0),
    AA3Letter.TRP: (2.152, -1.062, 0.0),
    AA3Letter.TYR: (2.151, -1.062, 0.0),
    AA3Letter.VAL: (2.154, -1.062, 0.0),
}

backbone_atom_coordinates_tensor = torch.zeros([21, 3, 3])
def make_coordinate_tensors():
    for restype, atom_coords in backbone_atom_coordinates.items():
        for atom_id, atom_coord in enumerate(atom_coords):
            backbone_atom_coordinates_tensor[restype][atom_id] = torch.FloatTensor(atom_coord)
        for restype, bb_oxy_coord in bb_oxygen_coordinate.items():
            bb_oxygen_coordinate_tensor[restype] = torch.FloatTensor(bb_oxy_coord)
make_coordinate_tensors()


# 下列张量由 `_make_rigid_group_constants`初始化
restype_rigid_group_rotation = torch.zeros([21, 8, 3, 3])
restype_rigid_group_translation = torch.zeros([21, 8, 3])
restype_heavyatom_to_rigid_group = torch.zeros([21, 14], dtype=torch.long)
restype_heavyatom_rigid_group_positions = torch.zeros([21, 14, 3])


def _make_rigid_group_constants():
    def _make_rotation_matrix(ex, ey):
        ex_normalized = ex / torch.linalg.norm(ex)

        # make ey perpendicular to ex
        ey_normalized = ey - torch.dot(ey, ex_normalized) * ex_normalized
        ey_normalized /= torch.linalg.norm(ey_normalized)

        eznorm = torch.cross(ex_normalized, ey_normalized)
        m = torch.stack([ex_normalized, ey_normalized, eznorm]).transpose(0, 1)  # (3, 3_index)
        return m

    for restype in AA:
        if restype == AA.X:
            continue

        atom_groups = {
            name: group
            for name, group, _ in rigid_group_heavy_atom_positions[restype]
        }
        atom_positions = {
            name: torch.FloatTensor(pos)
            for name, _, pos in rigid_group_heavy_atom_positions[restype]
        }

        # Atom 14 rigid group positions
        for atom_idx, atom_name in enumerate(restype_to_heavyatom_names[restype]):
            if (atom_name == '') or (atom_name not in atom_groups): continue
            restype_heavyatom_to_rigid_group[restype, atom_idx] = atom_groups[atom_name]
            restype_heavyatom_rigid_group_positions[restype, atom_idx, :] = atom_positions[atom_name]

        # 0: backbone to backbone
        restype_rigid_group_rotation[restype, Torsion.Backbone, :, :] = torch.eye(3)
        restype_rigid_group_translation[restype, Torsion.Backbone, :] = torch.zeros([3])

        # 1: omega-frame to backbone
        restype_rigid_group_rotation[restype, Torsion.Omega, :, :] = torch.eye(3)
        restype_rigid_group_translation[restype, Torsion.Omega, :] = torch.zeros([3])

        # 2: phi-frame to backbone
        restype_rigid_group_rotation[restype, Torsion.Phi, :, :] = _make_rotation_matrix(
            ex=atom_positions['N'] - atom_positions['CA'],
            ey=torch.FloatTensor([1., 0., 0.]),
        )
        restype_rigid_group_translation[restype, Torsion.Phi, :] = atom_positions['N']

        # 3: psi-frame to backbone
        restype_rigid_group_rotation[restype, Torsion.Psi, :, :] = _make_rotation_matrix(
            ex=atom_positions['C'] - atom_positions['CA'],
            ey=atom_positions['CA'] - atom_positions['N'],  # In accordance to the definition of psi angle
        )
        restype_rigid_group_translation[restype, Torsion.Psi, :] = atom_positions['C']

        # 4: chi1-frame to backbone
        if chi_angles_mask[restype][0]:
            base_atom_names = chi_angles_atoms[restype][0]
            base_atom_positions = [atom_positions[name] for name in base_atom_names]
            restype_rigid_group_rotation[restype, Torsion.Chi1, :, :] = _make_rotation_matrix(
                ex=base_atom_positions[2] - base_atom_positions[1],
                ey=base_atom_positions[0] - base_atom_positions[1],
            )
            restype_rigid_group_translation[restype, Torsion.Chi1, :] = base_atom_positions[2]

        # chi2-chi1
        # chi3-chi2
        # chi4-chi3
        for chi_idx in range(1, 4):
            if chi_angles_mask[restype][chi_idx]:
                axis_end_atom_name = chi_angles_atoms[restype][chi_idx][2]
                axis_end_atom_position = atom_positions[axis_end_atom_name]
                restype_rigid_group_rotation[restype, Torsion.Chi1 + chi_idx, :, :] = _make_rotation_matrix(
                    ex=axis_end_atom_position,
                    ey=torch.FloatTensor([-1., 0., 0.]),
                )
                restype_rigid_group_translation[restype, Torsion.Chi1 + chi_idx, :] = axis_end_atom_position


_make_rigid_group_constants()

restype_to_allatom_names = {
    restype: restype_to_heavyatom_names[restype]
    for restype in AA3Letter
}
