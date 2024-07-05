#####
# title: setup.py
#
# languag: setuptools
# date: 2024-02-05
# license: BSD-3-Clause
# author: Elmar Bucher
#
# description:
#   Building a setuptools based python3 C++ extension module.
# https://setuptools.pypa.io/en/latest/userguide/ext_modules.html
# https://elmjag.github.io/setuptools.html
#####


from setuptools import Extension, setup

# extract the version number from the VERSION.txt file
exec(open("./VERSION.txt").read())

setup(
    # version
    version=__version__,
    # compiler and linker ditrectives
    ext_modules=[
        Extension(
            name="embedding.physicell",  # as it would be imported # may include packages/namespaces separated by `.`
            # all sources are compiled into a single binary file
            sources=[  # straight outta PhysiCell Makefile
                # custom_modules_OBJECTS and components
                "physicellmodule.cpp",
                "../custom.cpp",
                # BioFVM_OBJECTS and components
                "BioFVM/BioFVM_agent_container.cpp",
                "BioFVM/BioFVM_basic_agent.cpp",  # bue 20240501: modified (reset_max_basic_agent_ID)
                "BioFVM/BioFVM_matlab.cpp",
                "BioFVM/BioFVM_mesh.cpp",
                "BioFVM/BioFVM_microenvironment.cpp",  # bue 20240501: modified (density exists?) [pull requested 20240616]
                "BioFVM/BioFVM_MultiCellDS.cpp",  # bue 20240509: modified (save mesh0.mat)
                "BioFVM/BioFVM_solvers.cpp",
                "BioFVM/BioFVM_utilities.cpp",
                "BioFVM/BioFVM_vector.cpp",
                "BioFVM/pugixml.cpp",
                # PhysiCell_core_OBJECTS and components
                "core/PhysiCell_basic_signaling.cpp",
                "core/PhysiCell_cell_container.cpp",
                "core/PhysiCell_cell.cpp",  # bue 20240610: and header file modified (cell_definition exists?) [pull requested 20240616]
                "core/PhysiCell_constants.cpp",
                "core/PhysiCell_custom.cpp",
                # "core/PhysiCell_digital_cell_line.cpp",
                "core/PhysiCell_phenotype.cpp",
                "core/PhysiCell_rules.cpp",
                "core/PhysiCell_signal_behavior.cpp",
                "core/PhysiCell_standard_models.cpp",
                "core/PhysiCell_utilities.cpp",
                # PhysiCell_module_OBJECTS and components
                "modules/PhysiCell_geometry.cpp",
                "modules/PhysiCell_MultiCellDS.cpp",
                "modules/PhysiCell_pathology.cpp",
                # "modules/PhysiCell_POV.cpp",
                "modules/PhysiCell_pugixml.cpp",
                "modules/PhysiCell_settings.cpp",  # bue 20240430: modified (parameter exists? and more.) [pull requested 20240616]
                "modules/PhysiCell_SVG.cpp",
                "modules/PhysiCell_various_outputs.cpp",
                # pugixml_OBJECTS and components
                # "pugixml.cpp",
            ],
            extra_compile_args=[  # straight outta PhysiCell Makefile
                "-march=native",  # ARCH
                "-O3",  # CFLAG
                "-fomit-frame-pointer",  # CFLAG
                "-mfpmath=both",  # CFLAG
                "-fopenmp",  # CFLAG
                "-m64",  # CFLAG
                "-std=c++11",  # CFLAG
                "-g",
            ],
            extra_link_args=[  # needed for openmp
                "-lgomp",
            ],
        ),
    ],
)
