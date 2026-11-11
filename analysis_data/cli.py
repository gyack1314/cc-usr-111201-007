"""展示已有的资料基本统计，不改变原文件。"""
import argparse
import json
from .io import load
from .domain import summarize
def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['inspect']);p.add_argument('file');a=p.parse_args()
    value,digest=load(a.file)
    print(json.dumps({'source_sha256':digest,'summary':summarize(value)},ensure_ascii=False,indent=2,allow_nan=False))
if __name__=='__main__':main()
