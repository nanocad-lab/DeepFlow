FILENAME = "./netlist.lef"

def reset_file():
    with open(FILENAME, "w") as f:
        f.write("")

def write_macro(macro_name, lines):
    with open(FILENAME, "a") as f:
        f.write("MACRO " + macro_name + "\n")
        for line in lines:
            f.write("\t" + line + "\n")
        f.write("END " + macro_name + "\n")
        f.write("\n")

def generate_system_macro(num_packages, topology="mesh"):
    macros = []
    macro_lines = []

    system_macro_lines = []
    for i in range(num_packages):
        system_macro_lines.append("package" + str(i))
    if topology=="mesh":
        for i in range(num_packages):
            for j in range(i+1, num_packages):
                system_macro_lines.append("wire(package" + str(i) + ", package" + str(j) + ")")
    
    return "system", system_macro_lines

def generate_package_macro(num_nodes, topology="mesh"):
    macros = []
    macro_lines = []

    package_macro_lines = []
    for i in range(num_nodes):
        package_macro_lines.append("node" + str(i))
    if topology=="mesh":
        for i in range(num_packages):
            for j in range(i+1, num_packages):
                package_macro_lines.append("write(node" + str(i) + ", node" + str(j) + ")")
    
    return "package", package_macro_lines

def generate_node_macro(num_hbms):
    macros = []
    macro_lines = []

    node_macro_lines = []
    node_macro_lines.append("accelerator_die")
    for i in range(num_hbms):
        node_macro_lines.append("hbm" + str(i))
        node_macro_lines.append("wire(accelerator_die, hbm" + str(i) + ")")
    
    return "node", node_macro_lines

def generate_accelerator_die_macro(num_network_io, num_sms, num_mc_phy):
    macros = []
    macro_lines = []

    accelerator_die_macro_lines = []
    for i in range(num_network_io):
        accelerator_die_macro_lines.append("network_io" + str(i))
    accelerator_die_macro_lines.append("noc")
    for i in range(num_sms):
        accelerator_die_macro_lines.append("sm" + str(i))
    for i in range(num_mc_phy):
        accelerator_die_macro_lines.append("mc_phy" + str(i))
    
    return "accelerator_die", accelerator_die_macro_lines

def generate_noc_macro():
    macros = []
    macro_lines = []

    noc_macro_lines = ["l2"]
    
    return "noc", noc_macro_lines

def generate_sm_macro():
    macros = []
    macro_lines = []

    sm_macro_lines = ["reg", "core_group", "l1", "wire(reg, core_group)", "wire(core_group, l1)"]
    
    return "sm", sm_macro_lines

def generate_core_group_macro(num_cores):
    macros = []
    macro_lines = []

    core_group_macro_lines = []
    for i in range(num_cores):
        core_group_macro_lines.append("core" + str(i))
    
    return "core_group", core_group_macro_lines

if __name__ == "__main__":
    num_packages = 4
    num_nodes = 8
    num_cores = 4
    num_network_io = 4
    num_sms = 4
    num_mc_phy = 4
    num_hbms = 4

    system_macro_name, system_macro_lines = generate_system_macro(num_packages)
    package_macro_name, package_macro_lines = generate_package_macro(num_nodes)
    node_macro_name, node_macro_lines = generate_node_macro(num_hbms)
    accelerator_die_macro_name, accelerator_die_macro_lines = generate_accelerator_die_macro(num_network_io, num_sms, num_mc_phy)
    noc_macro_name, noc_macro_lines = generate_noc_macro()
    sm_macro_name, sm_macro_lines = generate_sm_macro()
    core_group_macro_name, core_group_macro_lines = generate_core_group_macro(num_cores)

    reset_file()
    write_macro(system_macro_name, system_macro_lines)
    write_macro(package_macro_name, package_macro_lines)
    write_macro(node_macro_name, node_macro_lines)
    write_macro(accelerator_die_macro_name, accelerator_die_macro_lines)
    write_macro(noc_macro_name, noc_macro_lines)
    write_macro(sm_macro_name, sm_macro_lines)
    write_macro(core_group_macro_name, core_group_macro_lines)
