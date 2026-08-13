"""Bundled training entry point for the four MLIP families used by MLIPFlow."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import training_chgnet, training_deepmd, training_m3gnet, training_mace
from training_common import FRAMEWORKS, OPERATIONS, TrainingError, distribution_version, load_config, seed_everything, write_result

def build_parser():
    p=argparse.ArgumentParser(description="Train/fine-tune DeepMD, M3GNet, CHGNet, or MACE")
    p.add_argument("--framework",choices=FRAMEWORKS,required=True); p.add_argument("--operation",choices=OPERATIONS,required=True)
    for name in ("config","data","output","result-manifest","device","dataset-fingerprint","config-fingerprint"): p.add_argument("--"+name,required=True)
    p.add_argument("--seed",type=int,required=True); p.add_argument("--precision",choices=("float32","float64"),required=True)
    p.add_argument("--foundation-model"); p.add_argument("--foundation-model-fingerprint"); p.add_argument("--dry-run",action="store_true")
    return p

def _paths(a):
    if a.seed<0: raise TrainingError("--seed must be non-negative")
    c=Path(a.config).expanduser().absolute(); d=Path(a.data).expanduser().absolute(); o=Path(a.output).expanduser().absolute(); r=Path(a.result_manifest).expanduser().absolute()
    if not c.is_file(): raise TrainingError(f"config does not exist: {c}")
    if not d.exists(): raise TrainingError(f"data does not exist: {d}")
    if o==r: raise TrainingError("output and result manifest must differ")
    if a.operation=="finetune":
        if not a.foundation_model or not a.foundation_model_fingerprint: raise TrainingError("finetune requires foundation model and fingerprint")
        f=Path(a.foundation_model).expanduser().absolute()
        if not f.exists(): raise TrainingError(f"foundation model does not exist: {f}")
        a.foundation_model=str(f)
    elif a.foundation_model or a.foundation_model_fingerprint: raise TrainingError("foundation arguments require finetune")
    a.config=str(c); a.data=str(d); a.output=str(o); a.result_manifest=str(r); return c,d

def _plan(a,c,cp,dp):
    if a.framework=="deepmd": return training_deepmd.plan(a,c)
    if a.framework=="mace": return training_mace.plan(a,c,cp,dp)
    if a.framework=="chgnet": return training_chgnet.plan(a,c,dp)
    return training_m3gnet.plan(a,c,dp)

def _run(a,c,cp,dp):
    if a.framework=="deepmd": return training_deepmd.run(a,c,cp,dp)
    if a.framework=="mace": return training_mace.run(a,c,cp,dp)
    if a.framework=="chgnet": return training_chgnet.run(a,c,cp,dp)
    return training_m3gnet.run(a,c,cp,dp)

def _version(f):
    return {"deepmd":lambda:distribution_version("deepmd-kit"),"m3gnet":lambda:distribution_version("matgl"),"chgnet":lambda:distribution_version("chgnet"),"mace":lambda:distribution_version("mace-torch","mace")}[f]()

def main(argv=None):
    a=build_parser().parse_args(argv)
    try:
        cp,dp=_paths(a); c=load_config(cp); plan=_plan(a,c,cp,dp)
        if a.dry_run: print(json.dumps(plan,indent=2,sort_keys=True)); return 0
        Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.result_manifest).parent.mkdir(parents=True,exist_ok=True); seed_everything(a.seed)
        media,metrics,provenance=_run(a,c,cp,dp); write_result(a,"OK",_version(a.framework),metrics,media,provenance); return 0
    except Exception as exc:
        if not a.dry_run:
            try: write_result(a,"FAIL",_version(a.framework),{},"application/octet-stream",{},str(exc))
            except Exception: pass
        print(f"ERROR: {exc}",file=sys.stderr); return 1
if __name__=="__main__": raise SystemExit(main())
