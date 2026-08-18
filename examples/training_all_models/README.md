# Bundled MLIP training examples

Starter configurations for DeepMD, M3GNet/MatGL, CHGNet, and MACE. The DFT handoff tests exercise real dpdata and ASE format round-trips when the development dependencies are installed; training execution itself remains scoped to the user's reviewed compute environment.

`dft-labeling.dataset-assemble` publishes one shared split in this shape:

```text
datasets/<dataset_id>/
  canonical.json
  split.json
  deepmd/{train,valid,test}/system-*/
  m3gnet/{train,valid,test}.json
  chgnet/{train,valid,test}.json
  mace/{train,valid,test}.extxyz
  assembly-result.json
```

The same canonical record IDs populate each corresponding partition. Energies remain
total eV/configuration and forces remain eV/angstrom. DeepMD stores VASP stress as
virial in eV, M3GNet and CHGNet retain VASP kbar 3×3 tensors for their current loaders,
and MACE stores the sign-inverted ASE stress in eV/angstrom³ and ASE Voigt order.
