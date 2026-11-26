import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/x) for x in ("models","dataset_generation")]
from train import cascade_trace,cascade_predict,select
from aggregate import stats,aggregate_experiments

class Dummy:
    def __init__(self, values): self.values=values
    def predict(self,x): return np.array([self.values[i] for i in x.index])

class ReviewTests(unittest.TestCase):
    def test_trace(self):
        x=pd.DataFrame({"X":[0,1,2,3]})
        truth=["S","B","W","GHZ"]
        models={"GHZ":Dummy([0,0,1,0]),"W":Dummy([0,0,0,1]),"B":Dummy([0,1,0,0])}
        self.assertEqual(list(cascade_predict(models,x,["X"])),["S","B","GHZ","W"])
        rows={(r["stage"],r["true_class"]):r for r in cascade_trace(models,x,["X"],truth)}
        self.assertEqual(rows["GHZ","W"]["errors"],1)
        self.assertEqual(rows["W","W"]["upstream_excluded"],1)
        self.assertEqual(rows["W","GHZ"]["accepted"],1)
    def test_selection(self):
        d=pd.DataFrame({"X":[-3,-2,-1,1,2,3],"label":[0,0,0,1,1,1]})
        auc,p,m=select(d,d,["X"],"poly",2)
        self.assertEqual(p["degree"],2)
        self.assertGreaterEqual(auc,.5)
    def test_aggregation(self):
        self.assertIsNone(stats([1])["sample_std"])
        self.assertIsNone(stats([None])["mean"])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); folders=[]
            for i,v in enumerate((.25,.75)):
                p=root/str(i); (p/"ood_results").mkdir(parents=True)
                (p/"manifest.json").write_text(json.dumps({"seed":i}))
                (p/"ood_results"/"metrics.json").write_text(json.dumps({"x":{"GHZ":{"accuracy":v,"n_certified":4,"n_total":5},"B":{"accuracy":None,"n_certified":0,"n_total":5}}}))
                folders.append(p)
            results=aggregate_experiments(folders,root/"out")
            rows={r["model"]:r for r in results["ood"]}
            self.assertAlmostEqual(rows["GHZ"]["mean"],.5)
            self.assertAlmostEqual(rows["GHZ"]["sample_std"],np.sqrt(.125))
            self.assertEqual(rows["GHZ"]["certified"],8)
            self.assertEqual(rows["B"]["n_runs"],0)
            self.assertIsNone(rows["B"]["mean"])

if __name__=="__main__":
    unittest.main()
