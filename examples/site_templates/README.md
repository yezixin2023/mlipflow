# Site template entry point

This directory is the public starting point for one MLIPFlow site's private
configuration. Copy `site.yaml.example` to `~/.mlipflow/site.yaml`, then install
only the template families supported by that site's scientific software below one
canonical `remote_template_root`.

The generic Slurm templates in this directory are the source of truth for the two
execution models:

- `slurm/mpi/{cpu,gpu}.sbatch.example` maps `CPUS` to MPI tasks;
- `slurm/single-python/{cpu,gpu}.sbatch.example` maps `CPUS` to threads for one
  Python process.

Operation templates have one maintained generic source each. Copy and customize
the relevant source for every family enabled at the site:

| Installed family | Public source template |
|---|---|
| `vasp/run.sh`, `vasp-batch/run.sh` | `vasp/run.sh.example`, `vasp/batch.run.sh.example` |
| `dft-dataset/run.sh`, `dft-dataset-reuse/run.sh` | [`../training_all_models/cluster/dft-dataset.run.sh.example`](../training_all_models/cluster/dft-dataset.run.sh.example) |
| `mlip-<framework>/run.sh`, `mlip-<framework>-publish/run.sh` | [`../training_all_models/cluster/run.sh.example`](../training_all_models/cluster/run.sh.example) |
| `ase-md-<calculator>/run.sh` and supported canonical variants | [`../ase_md_cluster/run.sh.example`](../ase_md_cluster/run.sh.example) |
| supported `lammps-<framework>-<target>/run.sh` families | [`../lammps_mlip_inputs/run.sh.example`](../lammps_mlip_inputs/run.sh.example) |
| `lasp-ssw/run.sh` | [`../lasp_random_walk/cluster/run.sh.example`](../lasp_random_walk/cluster/run.sh.example) |
| `active-learning-committee-canonical/run.sh` | [`../active_learning_validation/cluster/run.sh.example`](../active_learning_validation/cluster/run.sh.example) |

Replace every `/ABS/PATH/TO/...` value and optional site-setup comment. Preserve
all `{{...}}` placeholders. Keep SSH aliases, partitions, accounts, modules,
executables, environments, data/model roots, and work roots out of project files
and the public repository.

Do not create a second template or work root for another framework, partition, or
validation run. Add the required family under the physical site's existing roots.
Never install POTCAR data, model weights, datasets, or credentials in this public
template tree.
