"""读取合成分析资料并提供原始文件指纹。"""
import hashlib
import json
import math
from pathlib import Path

def finite(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError('数值必须有限且不能是布尔值')
    return value

def load(path):
    raw=Path(path).read_bytes()
    value=json.loads(raw.decode('utf8'),parse_constant=lambda x: (_ for _ in ()).throw(ValueError('不接受非有限JSON常量')))
    if not isinstance(value,dict):raise ValueError('资料顶层必须是对象')
    return value,hashlib.sha256(raw).hexdigest()

def unique(rows,key):
    seen=set()
    for row in rows:
        value=row[key]
        if not isinstance(value,str) or not value or value in seen:raise ValueError('身份必须非空且唯一')
        seen.add(value)
    return seen
