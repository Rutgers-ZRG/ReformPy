import warnings
from m3gnet.models import Relaxer
import ase.io

for category in (UserWarning, DeprecationWarning):
    warnings.filterwarnings("ignore", category=category, module="tensorflow")

atoms = ase.io.read('.'+'/'+'POSCAR')
ase.io.vasp.write_vasp('input.vasp', atoms, direct=True)
trajfile = 'opt.traj'

relaxer = Relaxer()  # This loads the default pre-trained model

relax_results = relaxer.relax(atoms, fmax=0.001, steps=10000, traj_file=trajfile, verbose=True)

final_structure = relax_results['final_structure']
final_energy_per_atom = float(relax_results['trajectory'].energies[-1] / len(atoms))

print(f"Relaxed lattice parameter is {final_structure.lattice.abc[0]:.6f} Å")
print(f"Final energy is {final_energy_per_atom:.6f} eV/atom")
