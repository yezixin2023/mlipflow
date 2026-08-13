"""CHGNet scratch training and checkpoint fine-tuning."""
import random
from pathlib import Path
from mlip_common import TrainingError, mapping, records, section, work_dir
TARGETS={"e","ef","efs","efm","efsm"}

def plan(args,config,data_path):
    cfg=section(config,"chgnet");targets=str(cfg.get("targets","ef"))
    if args.precision!="float32":raise TrainingError("CHGNet requires --precision float32")
    if targets not in TARGETS:raise TrainingError("unsupported CHGNet targets")
    return {"framework":"chgnet","operation":args.operation,"python_entrypoint":"chgnet.trainer.Trainer","targets":targets,"dataset":str(data_path),"foundation_model":args.foundation_model if args.operation=="finetune" else None,"output":args.output}
def run(args,config,config_path,data_path):
    plan(args,config,data_path);cfg=section(config,"chgnet");targets=str(cfg.get("targets","ef"))
    import torch
    from chgnet.data.dataset import StructureData, collate_graphs
    from chgnet.model.model import CHGNet
    from chgnet.trainer import Trainer
    from pymatgen.core import Structure
    from torch.utils.data import DataLoader, Subset
    structures=[];energies=[];forces=[];stresses=[];magmoms=[]
    for index,item in enumerate(records(data_path)):
        if not all(key in item for key in ("structure","energy","forces")):raise TrainingError(f"CHGNet record {index} requires structure, energy, forces")
        structure=Structure.from_dict(item["structure"]);structures.append(structure);energy=float(item["energy"]);energies.append(energy if cfg.get("energy_is_per_atom",False) else energy/len(structure));forces.append(item["forces"])
        if "s" in targets:
            if "stress" not in item:raise TrainingError(f"CHGNet record {index} requires stress")
            stresses.append(item["stress"])
        if "m" in targets:
            if "magmom" not in item:raise TrainingError(f"CHGNet record {index} requires magmom")
            magmoms.append(item["magmom"])
    if len(structures)<2:raise TrainingError("CHGNet requires at least two structures")
    dataset=StructureData(structures,energies,forces,stresses=stresses or None,magmoms=magmoms or None,shuffle=False);ids=list(range(len(dataset)));random.Random(args.seed).shuffle(ids);n=max(1,min(len(ids)-1,round(len(ids)*float(cfg.get("val_ratio",0.1)))))
    opts={"batch_size":int(cfg.get("batch_size",32)),"collate_fn":collate_graphs,"num_workers":int(cfg.get("num_workers",0))};train=DataLoader(Subset(dataset,ids[n:]),shuffle=True,generator=torch.Generator().manual_seed(args.seed),**opts);valid=DataLoader(Subset(dataset,ids[:n]),shuffle=False,**opts)
    model=CHGNet.from_file(str(Path(args.foundation_model).absolute())) if args.operation=="finetune" else CHGNet(**mapping(cfg.get("model"),"chgnet.model"));kw=mapping(cfg.get("trainer"),"chgnet.trainer")
    trainer=Trainer(model=model,targets=targets,torch_seed=args.seed,data_seed=args.seed,use_device="cuda" if args.device.lower()=="gpu" else args.device.lower(),**kw);work=work_dir(Path(args.result_manifest).absolute(),"chgnet");trainer.train(train,valid,test_loader=None,save_dir=str(work/"checkpoints"));output=Path(args.output).absolute();output.parent.mkdir(parents=True,exist_ok=True);trainer.save(filename=str(output))
    return "application/x-pytorch",{"model_size_bytes":float(output.stat().st_size)},{"samples":len(dataset),"targets":targets,"work_dir":str(work)}
