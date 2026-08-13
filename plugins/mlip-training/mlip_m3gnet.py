"""M3GNet scratch training and foundation fine-tuning through MatGL."""
import shutil
from pathlib import Path
from mlip_common import TrainingError, mapping, section, work_dir

def plan(args,config,data_path):
    if not data_path.is_file():raise TrainingError("M3GNet data must be a MatPES JSON/JSONL file")
    section(config,"m3gnet");return {"framework":"m3gnet","operation":args.operation,"python_entrypoint":"matgl.MGLPotentialTrainer","dataset":str(data_path),"foundation_model":args.foundation_model if args.operation=="finetune" else None,"output":args.output}
def run(args,config,config_path,data_path):
    plan(args,config,data_path);import matgl, torch
    from matgl import MGLDatasetLoader, MGLPotentialTrainer
    from matgl.models import M3GNet
    cfg=section(config,"m3gnet");data_cfg=mapping(cfg.get("data"),"m3gnet.data");model_cfg=mapping(cfg.get("model"),"m3gnet.model");trainer_cfg=mapping(cfg.get("trainer"),"m3gnet.trainer");matgl.set_default_dtype("float",64 if args.precision=="float64" else 32);work=work_dir(Path(args.result_manifest).absolute(),"m3gnet")
    if args.operation=="finetune":
        source=Path(args.foundation_model).absolute()
        if not source.is_dir():raise TrainingError("M3GNet foundation model must be a MatGL model directory")
        potential=matgl.load_model(str(source));model=getattr(potential,"model",potential);model=model.to(dtype=torch.float64 if args.precision=="float64" else torch.float32);elements=tuple(model.element_types);mean=getattr(potential,"data_mean",0.0);std=getattr(potential,"data_std",1.0)
    else:
        elements=tuple(model_cfg.pop("element_types",())) or None;model=None;mean=trainer_cfg.pop("data_mean",0.0);std=trainer_cfg.pop("data_std",1.0)
    dataset=MGLDatasetLoader.from_json(data_path,cutoff=float(data_cfg.get("cutoff",5.0)),element_types=elements,save_cache=False,root=str(work/"cache"),stress_unit=str(data_cfg.get("stress_unit","kbar")))
    if model is None:
        elements=elements or tuple(dataset.converter.element_types);model_cfg.setdefault("is_intensive",False);model=M3GNet(element_types=elements,**model_cfg)
    lightning=mapping(trainer_cfg.pop("trainer_kwargs",None),"m3gnet.trainer.trainer_kwargs");loader=mapping(trainer_cfg.pop("loader_kwargs",None),"m3gnet.trainer.loader_kwargs");lightning.update({"logger":False,"enable_checkpointing":False,"enable_model_summary":False});accelerator="gpu" if args.device.lower() in {"gpu","cuda"} else args.device.lower();trainer=MGLPotentialTrainer(model,seed=args.seed,accelerator=accelerator,devices=1,data_mean=mean,data_std=std,trainer_kwargs=lightning,loader_kwargs=loader,**trainer_cfg);native=work/"model";trainer.fit(dataset=dataset,save_path=native)
    output=Path(args.output).absolute();output.parent.mkdir(parents=True,exist_ok=True);base=work/"packed-model";created=Path(shutil.make_archive(str(base),"gztar",root_dir=work,base_dir="model"));shutil.move(str(created),str(output));return "application/gzip",{"model_size_bytes":float(output.stat().st_size)},{"element_types":list(model.element_types),"work_dir":str(work)}
