GameNASfrom collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]







GameNAS = Genotype(normal=[('max_pool_3x3', 1), ('sep_conv_5x5', 1), ('skip_connect', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('skip_connect', 0), ('avg_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('sep_conv_5x5', 4), ('dil_conv_3x3', 4)], reduce_concat=range(2, 6))

GameNAS_image = Genotype(normal=[('max_pool_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 1), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('max_pool_3x3', 1), ('dil_conv_5x5', 1)], reduce_concat=range(2, 6))





















