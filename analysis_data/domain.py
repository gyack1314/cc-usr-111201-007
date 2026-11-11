"""计算基础权重比例，展示分层和整群身份。"""
from collections import defaultdict
from .io import finite,unique
def weighted_ratio(rows):
    numerator=denominator=0.0
    for row in rows:
        w=finite(row['base_weight'])
        if w<=0 or type(row['outcome']) is not int or row['outcome'] not in (0,1):raise ValueError('基础权重或结果错误')
        numerator+=w*row['outcome'];denominator+=w
    if not denominator:raise ValueError('没有有效记录')
    return numerator/denominator
def summarize(value):
    rows=value['sample'];unique(rows,'id');groups=defaultdict(set)
    ratio=weighted_ratio(rows)
    for row in rows:groups[row['stratum']].add(row['psu'])
    return {'base_weight_ratio':ratio,'records':len(rows),'psus':{k:sorted(v) for k,v in sorted(groups.items())}}
