import numpy as np
import os
import shutil
import subprocess
from math import cos, sin, radians


def create_case_directory(base_case, case_dir):
    """Crea una nuova directory per il caso"""
    if os.path.exists(case_dir):
        shutil.rmtree(case_dir)
    shutil.copytree(base_case, case_dir)

def modify_U_file(case_dir, velocity, angle):
    """Modifica il file U con nuova velocità e angolo"""
    # Calcola le componenti della velocità
    vz = velocity * cos(radians(angle))
    vy = velocity * sin(radians(angle))
    
    U_file = os.path.join(case_dir, "0", "U")
    with open(U_file, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "internalField" in line:
            new_lines.append(line)
            new_lines.append(f"uniform (0 {vy} {vz});\n")
            i += 2  # Salta la vecchia riga del valore
        elif "inlet" in line and "value" in lines[i+2]:
            new_lines.extend(lines[i:i+2])  # Aggiungi 'inlet {' e 'type fixedValue;'
            new_lines.append(f"        value           uniform (0 {vy} {vz});\n")
            i += 4  # Salta fino alla fine del blocco inlet
        else:
            new_lines.append(line)
            i += 1
    
    with open(U_file, 'w') as f:
        f.writelines(new_lines)

def run_simulation(case_dir):
    """Esegue la simulazione OpenFOAM usando WSL"""
    print(case_dir)
    # Ottieni il percorso assoluto e rimuovi l'eventuale prefisso 'C:\' o 'c:\'
    abs_path = os.path.abspath(case_dir)
    abs_path = abs_path.replace('C:\\', '').replace('c:\\', '')
    
    # Converti il percorso nel formato WSL
    wsl_path = '/mnt/c/' + abs_path.replace('\\', '/')
    
    print(f"Percorso WSL: {wsl_path}")  # Per debug
    
    # Comando WSL
    wsl_command = f"""
    . $HOME/OpenFOAM-12/etc/bashrc && \
    cd "{wsl_path}" && \
    rhoCentralFoam
    """
    
    # Esegui il comando in WSL
    try:
        result = subprocess.run(
            ['wsl', 'bash', '-l', '-c', wsl_command],
            capture_output=True,
            text=True,
            check=True
        )
        print("Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Errore nell'esecuzione:")
        print("Output:", e.output)
        print("Error:", e.stderr)
        raise


def main():
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Parametri di simulazione
    n_simulations = 10
    base_case = os.path.join(script_directory, "../.data/referenceCase")
    
    # Range dei parametri
    velocity_range = (20, 40)  # m/s
    angle_range = (-15, 15)    # gradi
    
    # Genera combinazioni casuali di parametri
    velocities = np.random.uniform(velocity_range[0], velocity_range[1], n_simulations)
    angles = np.random.uniform(angle_range[0], angle_range[1], n_simulations)
    
    # Crea un file per tracciare i parametri
    simulation_parameter_dir = os.path.join(script_directory, "simulation_parameters.csv")
    with open(simulation_parameter_dir, "w") as f:
        f.write("case,velocity,angle\n")
    
    # Esegui le simulazioni
    for i in range(n_simulations):
        case_name = f"case_{i:03d}"
        case_dir = os.path.join(script_directory, "../.data", case_name)
        
        # Registra i parametri
        with open(simulation_parameter_dir, "a") as f:
            f.write(f"{case_name},{velocities[i]:.2f},{angles[i]:.2f}\n")
        
        print(f"\nAvvio simulazione {i+1}/{n_simulations}")
        print(f"Velocità: {velocities[i]:.2f} m/s")
        print(f"Angolo: {angles[i]:.2f} gradi")
        
        # Crea e prepara il caso
        create_case_directory(base_case, case_dir)
        modify_U_file(case_dir, velocities[i], angles[i])
        
        # Esegui la simulazione
        try:
            run_simulation(case_dir)
        except subprocess.CalledProcessError as e:
            print(f"Errore nella simulazione {case_dir}: {e}")
            continue

if __name__ == "__main__":
    main()