import numpy as np

audio_snips = [
    'snip01', 'snip02', 'snip03', 'snip04', 'snip05',
    'snip06', 'snip07', 'snip08', 'snip09', 'snip10',
    'snip11', 'snip12', 'snip13', 'snip14', 'snip15',
    'snip16', 'snip17', 'snip18', 'snip19', 'snip20',
    'snip21', 'snip22', 'snip23', 'snip24', 'snip25',
]

subjects = [
    'p01', 'p02', 'p03', 'p04', 'p05', 'p06', 'p07', 'p08',
    'p09', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16',
    'p17', 'p18', 'p19', 'p20', 'p21', 'p22', 'p23', 'p24',
    'p25', 'p26', 'p27', 'p28', 'p29', 'p30', 'p31', 'p32',
    'p33', 'p34', 'p35', 'p36', 'p37', 'p38', 'p39', 'p40',
    'p41', 'p42', 'pilot02', 'pilot03'
]

auditory_cluster = ['F3', 'FC1', 'FC5', 'FC6', 'FC2', 'F4']

# # Find indeces of auditory cluster in all electrodes
# indices = [electrodes.index(electrode) for electrode in auditory_cluster]

electrodes = [
    'Fp1',
    'AF3',
    'F7',
    'F3',
    'FC1',
    'FC5',
    'T7',
    'C3',
    'CP1',
    'CP5',
    'P7',
    'P3',
    'Pz',
    'PO3',
    'O1',
    'Oz',
    'O2',
    'PO4',
    'P4',
    'P8',
    'CP6',
    'CP2',
    'C4',
    'T8',
    'FC6',
    'FC2',
    'F4',
    'F8',
    'AF4',
    'Fp2',
    'Fz',
    'Cz'
]

bad_channels_dict = {
    'p01': ['T8', 'PO4', 'O2'],
    'p02': [],
    'p03': ['AF4', 'F4', 'Oz', 'PO4'],
    'p04': [],
    'p05': [],
    'p06': ['P7', 'O2', 'P8', 'F7', 'O1'],
    'p07': ['Oz', 'FC6'],
    'p08': [],
    'p09': [],
    'p10': [],
    'p11': [],
    'p12': ['PO4'],
    'p13': ['T7', 'CP6', 'P4', 'CP5'],
    'p14': ['O1'],
    'p15': [],
    'p16': [],
    'p17': [],
    'p18': ['T7', 'T8'],
    'p19': ['AF4', 'F3'],
    'p20': ['Pz', 'O2'],
    'p21': [],
    'p22': ['PO3', 'Fz'],
    'p23': ['T7', 'T8'],
    'p24': [],
    'p25': [],
    'p26': ['Fz', 'C3', 'P7'],
    'p27': ['P7', 'O1'],
    'p28': [],
    'p29': [],
    'p30': [],
    'p31': [],
    'p32': [],
    'p33': [],
    'p34': [],
    'p35': ['F4', 'EXG1'],
    'p36': [],
    'p37': [],
    'p38': ['FC5'],
    'p39': [],
    'p40': ['P4', 'Oz'],
    'p41': [],
    'p42': [],
    # pilot 01 not usable
    'pilot02': [],
    'pilot03': [],
}
